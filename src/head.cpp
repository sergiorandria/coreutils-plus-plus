#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <iostream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __linux__
#  include <pthread.h>
#endif

#if defined(__AVX512BW__)
#  include <immintrin.h>
#elif defined(__AVX2__)
#  include <immintrin.h>
#elif defined(__SSE2__)
#  include <emmintrin.h>
#endif

namespace tp {

static constexpr unsigned kMaxCore     = 64u;
static constexpr std::size_t kBufSize  = 128;
static constexpr std::size_t kCacheSz  = 64;

inline const unsigned N_CORE = []() noexcept -> unsigned {
    const unsigned hw = std::thread::hardware_concurrency();
    return std::min(hw > 0u ? hw : 1u, kMaxCore);
}();

template<std::size_t N = kBufSize>
class task_wrapper {
    alignas(std::max_align_t) char buf_[N];
    void (*inv_) (void*)        = nullptr;
    void (*dtor_)(void*)        = nullptr;
    void (*mov_) (void*, void*) = nullptr;

public:
    task_wrapper() = default;

    template<typename F,
             typename = std::enable_if_t<std::is_invocable_r_v<void, std::decay_t<F>&>>>
    
    explicit task_wrapper(F&& f) {
        using D = std::decay_t<F>;
        static_assert(sizeof(D)  <= N,                          "Task too large for SBO buffer");
        static_assert(alignof(D) <= alignof(std::max_align_t), "Task alignment exceeds max_align_t");
        new (buf_) D(std::forward<F>(f));
        inv_  = [](void* p) { (*static_cast<D*>(p))(); };
        dtor_ = [](void* p) {   static_cast<D*>(p)->~D(); };
        mov_  = [](void* s, void* d) {
            new (d) D(std::move(*static_cast<D*>(s)));
            static_cast<D*>(s)->~D();
        };
    }

    task_wrapper(task_wrapper&& o) noexcept {
        if (o.inv_) {
            o.mov_(o.buf_, buf_);
            inv_ = o.inv_; dtor_ = o.dtor_; mov_ = o.mov_;
            o.inv_ = o.dtor_ = nullptr; o.mov_ = nullptr;
        }
    }

    task_wrapper& operator=(task_wrapper&& o) noexcept {
        if (this != &o) {
            if (dtor_) dtor_(buf_);
            if (o.inv_) {
                o.mov_(o.buf_, buf_);
                inv_ = o.inv_; dtor_ = o.dtor_; mov_ = o.mov_;
                o.inv_ = o.dtor_ = nullptr; o.mov_ = nullptr;
            } else {
                inv_ = dtor_ = nullptr; mov_ = nullptr;
            }
        }
        return *this;
    }

    void operator()() noexcept { if (inv_) inv_(buf_); }
    explicit operator bool() const noexcept { return inv_ != nullptr; }
    ~task_wrapper() { if (dtor_) dtor_(buf_); }

    task_wrapper(const task_wrapper&)            = delete;
    task_wrapper& operator=(const task_wrapper&) = delete;
};

struct alignas(kCacheSz) aligned_task_queue {
    std::queue<task_wrapper<>> tasks;
    std::mutex                 mtx;

    aligned_task_queue() = default;
    aligned_task_queue(aligned_task_queue&& o) noexcept : tasks(std::move(o.tasks)) {}
    aligned_task_queue& operator=(aligned_task_queue&& o) noexcept {
        if (this != &o) tasks = std::move(o.tasks);
        return *this;
    }
    aligned_task_queue(const aligned_task_queue&)            = delete;
    aligned_task_queue& operator=(const aligned_task_queue&) = delete;
};

class __wc_thread_pool {
public:
    __wc_thread_pool(const __wc_thread_pool&)            = delete;
    __wc_thread_pool& operator=(const __wc_thread_pool&) = delete;

    [[nodiscard]] static __wc_thread_pool* Instance() noexcept {
        std::call_once(init_, [] { inst_.reset(new __wc_thread_pool()); });
        return inst_.get();
    }

    template<typename F, typename... A>
    [[nodiscard]] auto submit(F&& fn, A&&... args)
        -> std::future<std::invoke_result_t<F, A...>>
    {
        using R = std::invoke_result_t<F, A...>;
        auto pkg = std::make_shared<std::packaged_task<R()>>(
            [f = std::forward<F>(fn), ...a = std::forward<A>(args)]() mutable -> R {
                return f(std::move(a)...);
            });
        auto fut = pkg->get_future();
        push([pkg] { try { (*pkg)(); } catch (...) {} });
        return fut;
    }

    void enqueue(std::function<void()> fn) {
        push([f = std::move(fn)] { try { f(); } catch (...) {} });
    }

    template<typename It>
    void enqueue_batch(It first, It last) {
        std::size_t cnt = 0;
        {
            std::lock_guard<std::mutex> lk(smtx_);
            if (stop_.load(std::memory_order_acquire))
                throw std::runtime_error("enqueue_batch: pool stopped");
            for (auto it = first; it != last; ++it, ++cnt) {
                const std::size_t q = nxt_.fetch_add(1, std::memory_order_relaxed) % N_CORE;
                std::lock_guard<std::mutex> ql(qs_[q].mtx);
                qs_[q].tasks.emplace([t = *it] { try { t(); } catch (...) {} });
            }
            active_.fetch_add(cnt, std::memory_order_release);
        }
        cnt == 1 ? cv_.notify_one() : cv_.notify_all();
    }

    void shutdown(std::chrono::milliseconds tmo = std::chrono::milliseconds(5'000)) {
        {
            std::lock_guard<std::mutex> lk(smtx_);
            stop_.store(true, std::memory_order_release);
        }
        cv_.notify_all();
        const auto dl = std::chrono::steady_clock::now() + tmo;
        for (auto& t : thr_) {
            if (!t.joinable()) continue;
            std::chrono::steady_clock::now() < dl ? t.join() : t.detach();
        }
    }

    [[nodiscard]] std::size_t thread_count() const noexcept { return N_CORE; }
    [[nodiscard]] std::size_t active_tasks()  const noexcept {
        return active_.load(std::memory_order_acquire);
    }
    void wait_all() const noexcept {
        while (active_.load(std::memory_order_acquire) > 0)
            std::this_thread::yield();
    }

    ~__wc_thread_pool() { shutdown(); }

private:
    __wc_thread_pool() : qs_(N_CORE) {
        cpu_ = std::thread::hardware_concurrency();
        thr_.reserve(N_CORE);
        for (std::size_t i = 0; i < N_CORE; ++i) {
            thr_.emplace_back([this, i] { worker(i); });
#ifdef __linux__
            cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(i % cpu_, &cs);
            pthread_setaffinity_np(thr_.back().native_handle(), sizeof(cs), &cs);
#endif
        }
    }

    template<typename F>
    void push(F&& fn) {
        {
            std::lock_guard<std::mutex> lk(smtx_);
            if (stop_.load(std::memory_order_acquire))
                throw std::runtime_error("push: pool stopped");
            const std::size_t q = nxt_.fetch_add(1, std::memory_order_relaxed) % N_CORE;
            std::lock_guard<std::mutex> ql(qs_[q].mtx);
            qs_[q].tasks.emplace(std::forward<F>(fn));
            active_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_one();
    }

    void worker(std::size_t id) {
        while (true) {
            task_wrapper<> task;
            bool found = false;

            {
                std::lock_guard<std::mutex> lk(qs_[id].mtx);
                if (!qs_[id].tasks.empty()) {
                    task  = std::move(qs_[id].tasks.front());
                    qs_[id].tasks.pop();
                    found = true;
                }
            }

            if (!found) {
                for (std::size_t i = 1; i <= N_CORE; ++i) {
                    const std::size_t s = (id + i) % N_CORE;
                    std::unique_lock<std::mutex> lk(qs_[s].mtx, std::try_to_lock);
                    if (lk.owns_lock() && !qs_[s].tasks.empty()) {
                        task  = std::move(qs_[s].tasks.front());
                        qs_[s].tasks.pop();
                        found = true;
                        break;
                    }
                }
            }

            if (found) {
                task();
                active_.fetch_sub(1, std::memory_order_release);
                continue;
            }

            {
                std::unique_lock<std::mutex> lk(wmtx_);
                cv_.wait(lk, [this] {
                    return stop_.load(std::memory_order_acquire)
                        || active_.load(std::memory_order_acquire) > 0;
                });
            }

            if (stop_.load(std::memory_order_acquire)) {
                while (true) {
                    task_wrapper<> drain;
                    {
                        std::lock_guard<std::mutex> lk(qs_[id].mtx);
                        if (qs_[id].tasks.empty()) break;
                        drain = std::move(qs_[id].tasks.front());
                        qs_[id].tasks.pop();
                    }
                    drain();
                    active_.fetch_sub(1, std::memory_order_release);
                }
                return;
            }
        }
    }

    static std::unique_ptr<__wc_thread_pool> inst_;
    static std::once_flag                    init_;

    unsigned                        cpu_ = 0;
    std::vector<std::thread>        thr_;
    std::vector<aligned_task_queue> qs_;
    std::mutex                      smtx_;
    std::mutex                      wmtx_;
    std::condition_variable         cv_;
    std::atomic<bool>               stop_{false};
    std::atomic<std::size_t>        active_{0};
    std::atomic<std::size_t>        nxt_{0};
};

std::unique_ptr<__wc_thread_pool> __wc_thread_pool::inst_;
std::once_flag                    __wc_thread_pool::init_;

using thread_pool = __wc_thread_pool;

} // namespace tp

namespace detail {

class Logger {
public:
    [[nodiscard]] static Logger* logger() noexcept {
        if (!log_) [[unlikely]] {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!log_) log_ = new Logger();
        }
        return log_;
    }

    Logger(const Logger&) = delete;
    Logger(Logger&&)      = delete;
    ~Logger()             = default;

    [[noreturn]] void display_error(const char* msg = "Error") noexcept {
        std::fprintf(stderr, "%s\n", msg);
        _exit(1);
    }

    [[noreturn]] void die(const char* prog, const char* fmt, ...) noexcept {
        std::fprintf(stderr, "%s: ", prog);
        va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
        fputc('\n', stderr);
        _exit(1);
    }

    [[nodiscard]] bool warn(const char* prog, const char* path) noexcept {
        std::fprintf(stderr, "%s: %s: %s\n", prog, path, strerror(errno));
        return false;
    }

    void write_all(int fd, const void* buf, std::size_t len) noexcept {
        const char* p = static_cast<const char*>(buf);
        while (len > 0) {
            const ssize_t w = ::write(fd, p, len);
            if (w <= 0) [[unlikely]] { perror("fasthead: write"); _exit(1); }
            p   += static_cast<std::size_t>(w);
            len -= static_cast<std::size_t>(w);
        }
    }

private:
    Logger() = default;
    static Logger*    log_;
    static std::mutex mtx_;
};

Logger*    Logger::log_ = nullptr;
std::mutex Logger::mtx_;

inline Logger* logger = Logger::logger();

[[nodiscard]] inline std::string_view file_extension(std::string_view path) noexcept {
    const auto d = path.rfind('.');
    return d == std::string_view::npos ? "" : path.substr(d + 1);
}

struct ParsedCount { long long value; bool negative; };

[[nodiscard]] ParsedCount
parse_count(const char* s, bool is_bytes, const char* prog) noexcept
{
    const char* p = s;
    bool neg = false;
    if (*p == '-')      { neg = true; ++p; }
    else if (*p == '+')   ++p;

    if (!*p || !std::isdigit(static_cast<unsigned char>(*p)))
        logger->die(prog, "invalid count '%s'", s);

    char* end;
    long long v = std::strtoll(p, &end, 10);
    if (v < 0) logger->die(prog, "count too large '%s'", s);

    static constexpr struct { const char* sfx; long long m; bool bytes_only; } T[] = {
        {"b",   512LL,                  true },
        {"kB",  1000LL,                 false},
        {"k",   1024LL,                 false},
        {"K",   1024LL,                 false},
        {"KiB", 1024LL,                 false},
        {"MB",  1000000LL,              false},
        {"M",   1048576LL,              false},
        {"MiB", 1048576LL,              false},
        {"GB",  1000000000LL,           false},
        {"G",   1073741824LL,           false},
        {"GiB", 1073741824LL,           false},
        {"TB",  1000000000000LL,        false},
        {"T",   1099511627776LL,        false},
        {"TiB", 1099511627776LL,        false},
        {"PB",  1000000000000000LL,     false},
        {"P",   1125899906842624LL,     false},
        {"PiB", 1125899906842624LL,     false},
        {"EB",  1000000000000000000LL,  false},
        {"E",   1152921504606846976LL,  false},
        {"EiB", 1152921504606846976LL,  false},
        {nullptr, 0LL, false}
    };

    if (*end) {
        bool ok = false;
        for (int i = 0; T[i].sfx; ++i) {
            if ((!T[i].bytes_only || is_bytes) && std::strcmp(end, T[i].sfx) == 0) {
                v *= T[i].m; ok = true; break;
            }
        }
        if (!ok) logger->die(prog, "invalid suffix in '%s'", s);
    }
    return {v, neg};
}

[[nodiscard]] std::vector<char> read_stdin() noexcept {
    std::vector<char> buf;
    buf.reserve(65536);
    char tmp[65536]; ssize_t n;
    while ((n = ::read(STDIN_FILENO, tmp, sizeof(tmp))) > 0)
        buf.insert(buf.end(), tmp, tmp + n);
    return buf;
}

[[nodiscard]] static const char*
find_nth_delimiter(const char* __restrict__ data, std::size_t size,
                   char delim, long long n) noexcept
{
    if (n <= 0) [[unlikely]] return data;

    const char* pos = data;
    const char* end = data + size;

#if defined(__AVX512BW__)
    {
        const __m512i dv = _mm512_set1_epi8(delim);
        while (pos + 64 <= end) {
            const uint64_t mask = _mm512_cmpeq_epi8_mask(
                _mm512_loadu_si512(reinterpret_cast<const __m512i*>(pos)), dv);
            const int cnt = __builtin_popcountll(mask);
            if (cnt < n) [[likely]] { n -= cnt; pos += 64; continue; }
            uint64_t m = mask;
            while (m) {
                const int idx = __builtin_ctzll(m);
                if (--n == 0) [[unlikely]] return pos + idx + 1;
                m &= m - 1;
            }
        }
    }
#endif

#if defined(__AVX2__)
    {
        const __m256i dv = _mm256_set1_epi8(delim);
        while (pos + 32 <= end) {
            const unsigned mask = static_cast<unsigned>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pos)), dv)));
            const int cnt = __builtin_popcount(mask);
            if (cnt < n) [[likely]] { n -= cnt; pos += 32; continue; }
            unsigned m = mask;
            while (m) {
                const int idx = __builtin_ctz(m);
                if (--n == 0) [[unlikely]] return pos + idx + 1;
                m &= m - 1;
            }
        }
    }
#elif defined(__SSE2__)
    {
        const __m128i dv = _mm_set1_epi8(delim);
        while (pos + 16 <= end) {
            const unsigned mask = static_cast<unsigned>(
                _mm_movemask_epi8(_mm_cmpeq_epi8(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(pos)), dv)));
            const int cnt = __builtin_popcount(mask);
            if (cnt < n) [[likely]] { n -= cnt; pos += 16; continue; }
            unsigned m = mask;
            while (m) {
                const int idx = __builtin_ctz(m);
                if (--n == 0) [[unlikely]] return pos + idx + 1;
                m &= m - 1;
            }
        }
    }
#endif

    while (pos < end) {
        if (*pos == delim) [[unlikely]]
            if (--n == 0) return pos + 1;
        ++pos;
    }
    return nullptr;
}

[[nodiscard]] static std::size_t
count_delimiters(const char* __restrict__ data, std::size_t size, char delim) noexcept
{
    const char* pos = data;
    const char* end = data + size;
    std::size_t cnt = 0;

#if defined(__AVX512BW__)
    {
        const __m512i dv = _mm512_set1_epi8(delim);
        while (pos + 64 <= end) {
            cnt += static_cast<std::size_t>(__builtin_popcountll(
                _mm512_cmpeq_epi8_mask(
                    _mm512_loadu_si512(reinterpret_cast<const __m512i*>(pos)), dv)));
            pos += 64;
        }
    }
#endif

#if defined(__AVX2__)
    {
        const __m256i dv = _mm256_set1_epi8(delim);
        while (pos + 32 <= end) {
            cnt += static_cast<std::size_t>(__builtin_popcount(static_cast<unsigned>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pos)), dv)))));
            pos += 32;
        }
    }
#elif defined(__SSE2__)
    {
        const __m128i dv = _mm_set1_epi8(delim);
        while (pos + 16 <= end) {
            cnt += static_cast<std::size_t>(__builtin_popcount(static_cast<unsigned>(
                _mm_movemask_epi8(_mm_cmpeq_epi8(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(pos)), dv)))));
            pos += 16;
        }
    }
#endif

    while (pos < end) { cnt += (*pos++ == delim); }
    return cnt;
}

} // namespace detail

namespace core {

namespace fs {

enum class FileType { html, bash, python, txt, cpp, c, csharp, javascript, css, rust };

// Interface for file types 
class VirtualFileType {
public:
    virtual std::string get_headers() = 0;
    virtual ~VirtualFileType() = default;

protected: 
    std::string headers_; 
};

class __html        : public VirtualFileType { std::string get_headers() override { return this->headers_; } };
class __bash        : public VirtualFileType { std::string get_headers() override { return this->headers_; } };
class __python      : public VirtualFileType { std::string get_headers() override { return this->headers_; } };
class __txt         : public VirtualFileType { std::string get_headers() override { return this->headers_; } };
class __cpp         : public VirtualFileType { std::string get_headers() override { return this->headers_; } };
class __c           : public VirtualFileType { std::string get_headers() override { return this->headers_; } };
class __csharp      : public VirtualFileType { std::string get_headers() override { return this->headers_; } };
class __javascript  : public VirtualFileType { std::string get_headers() override { return this->headers_; } };
class __css         : public VirtualFileType { std::string get_headers() override { return this->headers_; } };
class __rust        : public VirtualFileType { std::string get_headers() override { return this->headers_; } };

[[nodiscard]] inline FileType detect_file_type(std::string_view path) noexcept {
    const auto ext = detail::file_extension(path);
    if (ext == "html" || ext == "htm") return FileType::html;
    if (ext == "sh")                   return FileType::bash;
    if (ext == "py")                   return FileType::python;
    if (ext == "txt")                  return FileType::txt; 
    if (ext == "cpp")                  return FileType::cpp;
    if (ext == "c")                    return FileType::c; 
    if (ext == "csharp")               return FileType::csharp; 
    if (ext == "javascript")           return FileType::javascript;
    if (ext == "css")                  return FileType::css; 
    if (ext == "rust")                 return FileType::rust; 
    return FileType::txt;
}

struct alignas(16) RawFile {
    char*       data_        = nullptr;
    std::size_t size_        = 0;
    std::size_t lines_count_ = 0;
    int         fd           = -1;
    bool        is_mapped    = false;
    std::size_t mapped_size  = 0;

    [[nodiscard]] std::size_t size()            const noexcept { return size_; }
    [[nodiscard]] std::size_t get_lines_count() const noexcept { return lines_count_; }

    RawFile() = default;

    explicit RawFile(std::string_view filename) noexcept {
        fd = open(filename.data(), O_RDONLY | O_CLOEXEC);
        if (fd == -1) [[unlikely]] return;

        struct stat st{};
        if (fstat(fd, &st) == -1) [[unlikely]] { close(fd); fd = -1; return; }

        size_ = static_cast<std::size_t>(st.st_size);
        if (size_ == 0) return;

        const auto pg = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        mapped_size   = (size_ + pg - 1) & ~(pg - 1);

        void* ptr = mmap(nullptr, mapped_size, PROT_READ,
                         MAP_PRIVATE | MAP_POPULATE, fd, 0);
        if (ptr == MAP_FAILED) [[unlikely]] { is_mapped = false; return; }

        madvise(ptr, mapped_size, MADV_SEQUENTIAL | MADV_WILLNEED);
        data_     = static_cast<char*>(ptr);
        is_mapped = true;
    }

    ~RawFile() noexcept {
        if (is_mapped && data_) { munmap(data_, mapped_size); data_ = nullptr; is_mapped = false; }
        if (fd != -1)            { close(fd); fd = -1; }
    }

    RawFile(const RawFile&)            = delete;
    RawFile& operator=(const RawFile&) = delete;

    RawFile(RawFile&& o) noexcept
        : data_(o.data_), size_(o.size_), lines_count_(o.lines_count_),
          fd(o.fd), is_mapped(o.is_mapped), mapped_size(o.mapped_size)
    { o.data_ = nullptr; o.fd = -1; o.is_mapped = false; }
};

struct alignas(64) FileChunk {
    const char*         start      = nullptr;
    const char*         end        = nullptr;
    std::size_t         line_count = 0;
    std::size_t         chunk_id   = 0;
    std::vector<size_t> line_positions;
};

} // namespace fs

struct ProcessConfig {
    long long count      = 10;
    bool      negative   = false;
    bool      bytes_mode = false;
    char      delimiter  = '\n';
    bool      verbose    = false;
    bool      quiet      = false;
};

[[nodiscard]] inline fs::RawFile memory_map_file(const std::string& fn) noexcept {
    return fs::RawFile(fn);
}

[[nodiscard]] std::vector<fs::FileChunk>
create_chunks(const char* data, std::size_t size,
              std::size_t n_threads, char delim) noexcept
{
    if (n_threads == 0) n_threads = 1;
    if (size == 0)      return {};

    const auto pg = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    n_threads = std::min(n_threads, std::max<std::size_t>(1, size / pg));

    std::vector<fs::FileChunk> chunks;
    chunks.reserve(n_threads);

    const std::size_t  slice    = size / n_threads;
    const char*        file_end = data + size;
    const char*        cursor   = data;

    for (std::size_t id = 0; id < n_threads && cursor < file_end; ++id) {
        fs::FileChunk c;
        c.chunk_id = id;
        c.start    = cursor;

        if (id == n_threads - 1) {
            c.end = file_end;
        } else {
            const char* nom = cursor + slice;
            if (nom >= file_end) {
                c.end = file_end;
            } else {
                const char* d = static_cast<const char*>(
                    memchr(nom, delim, static_cast<std::size_t>(file_end - nom)));
                c.end = d ? d + 1 : file_end;
            }
        }
        cursor = c.end;
        chunks.push_back(std::move(c));
    }
    return chunks;
}

class CoreProcessor {
public:
    explicit CoreProcessor(
        std::size_t num_lines   = 10,
        std::size_t num_threads = std::thread::hardware_concurrency()) noexcept
        : m_lines  (num_lines   == 0 ? 10 : num_lines)
        , m_threads(num_threads == 0 ?  1 : num_threads)
    {}

    [[nodiscard]] std::string process(const std::string& filename);

    bool process_file(const std::string& filename, const ProcessConfig& cfg,
                      bool show_header, bool blank_before_header);

    ~CoreProcessor() = default;
    CoreProcessor(const CoreProcessor&) = delete;
    CoreProcessor(CoreProcessor&&)      = delete;

private:
    std::size_t m_lines;
    std::size_t m_threads;
    char        m_delim_ = '\n';

    [[nodiscard]] fs::FileChunk find_lines_in_chunk(const fs::FileChunk& chunk) {
        fs::FileChunk r = chunk;
        r.line_count    = detail::count_delimiters(
            chunk.start,
            static_cast<std::size_t>(chunk.end - chunk.start),
            m_delim_);
        return r;
    }

    [[nodiscard]] std::size_t
    count_delimiters_parallel(const char* data, std::size_t size, char delim) {
        m_delim_      = delim;
        auto  chunks  = create_chunks(data, size, m_threads, delim);
        auto* pool    = tp::thread_pool::Instance();

        std::vector<std::future<fs::FileChunk>> futs;
        futs.reserve(chunks.size());
        for (auto& c : chunks)
            futs.push_back(pool->submit([this, c]() mutable {
                return find_lines_in_chunk(c);
            }));

        std::size_t total = 0;
        for (auto& f : futs) total += f.get().line_count;
        return total;
    }

    void write_head(const char* data, std::size_t size, const ProcessConfig& cfg) {
        if (size == 0) return;

        if (cfg.negative && cfg.count == 0) {
            detail::logger->write_all(STDOUT_FILENO, data, size);
            return;
        }

        if (cfg.bytes_mode) {
            if (!cfg.negative) {
                const std::size_t n = std::min(static_cast<std::size_t>(cfg.count), size);
                if (n) detail::logger->write_all(STDOUT_FILENO, data, n);
            } else {
                const std::size_t skip = static_cast<std::size_t>(cfg.count);
                if (skip < size)
                    detail::logger->write_all(STDOUT_FILENO, data, size - skip);
            }
            return;
        }

        if (!cfg.negative) {
            if (cfg.count == 0) return;
            const char* ep = detail::find_nth_delimiter(data, size, cfg.delimiter, cfg.count);
            detail::logger->write_all(STDOUT_FILENO, data,
                ep ? static_cast<std::size_t>(ep - data) : size);
        } else {
            const std::size_t total = count_delimiters_parallel(data, size, cfg.delimiter);
            if (total > static_cast<std::size_t>(cfg.count)) {
                const long long target = static_cast<long long>(total) - cfg.count;
                const char*     ep     = detail::find_nth_delimiter(
                    data, size, cfg.delimiter, target);
                if (ep)
                    detail::logger->write_all(STDOUT_FILENO, data,
                        static_cast<std::size_t>(ep - data));
            }
        }
    }
};

bool CoreProcessor::process_file(const std::string& filename, const ProcessConfig& cfg,
                                  bool show_header, bool blank_before_header)
{
    auto emit_header = [&](const char* name) noexcept {
        if (blank_before_header)
            detail::logger->write_all(STDOUT_FILENO, "\n", 1);
        char hdr[4096 + 12];
        const int n = std::snprintf(hdr, sizeof(hdr), "==> %s <==\n", name);
        if (n > 0) detail::logger->write_all(STDOUT_FILENO, hdr, static_cast<std::size_t>(n));
    };

    if (filename == "-") {
        if (show_header) emit_header("standard input");
        auto buf = detail::read_stdin();
        write_head(buf.data(), buf.size(), cfg);
        return true;
    }

    fs::RawFile mf(filename);
    if (mf.fd == -1) [[unlikely]]
        return detail::logger->warn("fasthead", filename.c_str());

    if (!mf.is_mapped && mf.size_ > 0) [[unlikely]] {
        const int tmpfd = open(filename.c_str(), O_RDONLY | O_CLOEXEC);
        if (tmpfd == -1) return detail::logger->warn("fasthead", filename.c_str());
        char tmp[65536]; ssize_t n;
        std::vector<char> fbuf;
        fbuf.reserve(mf.size_);
        while ((n = ::read(tmpfd, tmp, sizeof(tmp))) > 0)
            fbuf.insert(fbuf.end(), tmp, tmp + n);
        ::close(tmpfd);
        if (show_header) emit_header(filename.c_str());
        write_head(fbuf.data(), fbuf.size(), cfg);
        return true;
    }

    if (show_header) emit_header(filename.c_str());
    if (mf.size_ == 0) return true;
    write_head(mf.data_, mf.size_, cfg);
    return true;
}

std::string CoreProcessor::process(const std::string& filename)
{
    fs::RawFile mf(filename);
    if (mf.fd == -1 || mf.size_ == 0 || !mf.is_mapped) return {};
    const char* ep = detail::find_nth_delimiter(
        mf.data_, mf.size_, '\n', static_cast<long long>(m_lines));
    const std::size_t len = ep ? static_cast<std::size_t>(ep - mf.data_) : mf.size_;
    return std::string(mf.data_, len);
}

namespace cldecl {

template<typename FileType>
class __head_internal_class {
public:
    [[nodiscard]] static __head_internal_class* Instance() noexcept {
        if (!inst_) [[unlikely]] {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!inst_) inst_ = new __head_internal_class();
        }
        return inst_;
    }

    __head_internal_class(const __head_internal_class&) = delete;
    __head_internal_class(__head_internal_class&&)      = delete;

    [[nodiscard]] constexpr fs::FileType return_file_type() const noexcept {
        if constexpr (std::is_same_v<FileType, fs::__html>)   return fs::FileType::html;
        if constexpr (std::is_same_v<FileType, fs::__bash>)   return fs::FileType::bash;
        if constexpr (std::is_same_v<FileType, fs::__python>) return fs::FileType::python;
        return fs::FileType::txt;
    }

private:
    __head_internal_class() = default;
    static __head_internal_class* inst_;
    static std::mutex              mtx_;
};

template<typename T> __head_internal_class<T>* __head_internal_class<T>::inst_ = nullptr;
template<typename T> std::mutex                 __head_internal_class<T>::mtx_;

} // namespace cldecl

} // namespace core

using namespace core::cldecl;
using namespace core::fs;

int main(int argc, const char* argv[])
{
    std::ios_base::sync_with_stdio(false);

    core::ProcessConfig      cfg;
    std::vector<std::string> files;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);

        if ((arg == "-n" || arg == "--lines") && i + 1 < argc) {
            const auto pc = detail::parse_count(argv[++i], false, "fasthead");
            cfg.count    = pc.value;
            cfg.negative = pc.negative;
        } else if ((arg == "-c" || arg == "--bytes") && i + 1 < argc) {
            const auto pc = detail::parse_count(argv[++i], true, "fasthead");
            cfg.count      = pc.value;
            cfg.negative   = pc.negative;
            cfg.bytes_mode = true;
        } else if (arg == "-q" || arg == "--quiet" || arg == "--silent") {
            cfg.quiet = true; cfg.verbose = false;
        } else if (arg == "-v" || arg == "--verbose") {
            cfg.verbose = true; cfg.quiet = false;
        } else if (arg == "--") {
            for (int j = i + 1; j < argc; ++j) files.emplace_back(argv[j]);
            break;
        } else if (!arg.empty() && arg[0] == '-') {
            detail::logger->die("fasthead", "unknown option '%s'", argv[i]);
        } else {
            files.emplace_back(argv[i]);
        }
    }

    if (files.empty()) 
        files.emplace_back("-");

    const bool show_header = files.size() > 1 ? !cfg.quiet : cfg.verbose;

    auto hObj = __head_internal_class<core::fs::__html>::Instance();
    (void)hObj;

    core::CoreProcessor proc;
    bool ok          = true;
    bool blank_before = false;

    for (const auto& fn : files) {
        if (!proc.process_file(fn, cfg, show_header, blank_before))
            ok = false;
        blank_before = show_header;
    }

    tp::thread_pool::Instance()->shutdown();
    return ok ? 0 : 1;
}
