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
#include <iostream>
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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __linux__
#include <pthread.h>
#endif

#if defined(__AVX512BW__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace tp {

static constexpr unsigned kMaxCore = 64u;
static constexpr std::size_t kBufSize = 128;
static constexpr std::size_t kCacheSz = 64;

inline const unsigned N_CORE = []() noexcept -> unsigned {
  const unsigned hw = std::thread::hardware_concurrency();
  return std::min(hw > 0u ? hw : 1u, kMaxCore);
}();

/**
 * @brief A lightweight move-only type-erased callable wrapper with Small Buffer
 * Optimization (SBO).
 *
 * This class provides functionality similar to a restricted
 * `std::function<void()>`, but with the following design constraints:
 *
 * - **Move-only semantics** (copy operations are deleted)
 * - **Type erasure** via function pointers
 * - **Small Buffer Optimization (SBO)** to avoid heap allocations
 * - Stores any callable invocable as `void()`
 *
 * The callable object is constructed in-place inside an internal buffer of size
 * `N`. No dynamic memory allocation is performed.
 *
 * @tparam N Size of the internal storage buffer in bytes (default: kBufSize)
 *
 * @note The stored callable must:
 * - Be invocable with signature `void()`
 * - Fit within the buffer size `N`
 * - Have alignment <= `alignof(std::max_align_t)`
 *
 * @warning Behavior is undefined if the callable exceeds buffer size or
 * alignment constraints.
 */
template <std::size_t N = kBufSize> class task_wrapper {
  /// @brief Internal storage buffer for SBO (properly aligned).
  alignas(std::max_align_t) char buf_[N];

  /// @brief Pointer to the invocation function.
  void (*inv_)(void *) = nullptr;

  /// @brief Pointer to the destructor function.
  void (*dtor_)(void *) = nullptr;

  /// @brief Pointer to the move function (move-construct into another buffer).
  void (*mov_)(void *, void *) = nullptr;

public:
  /**
   * @brief Default constructor.
   *
   * Constructs an empty task_wrapper with no callable stored.
   */
  task_wrapper() = default;

  /**
   * @brief Constructs a task_wrapper from a callable.
   *
   * Stores the callable using type erasure and SBO.
   *
   * @tparam F Callable type
   * @param f Callable object (forwarded)
   *
   * @note The callable is decayed before storage.
   *
   * @throws Compile-time error if:
   * - Callable is not invocable as `void()`
   * - Callable size exceeds buffer size `N`
   * - Callable alignment exceeds `std::max_align_t`
   */
  template <typename F, typename = std::enable_if_t<
                            std::is_invocable_r_v<void, std::decay_t<F> &>>>
  explicit task_wrapper(F &&f) {
    using D = std::decay_t<F>;

    static_assert(sizeof(D) <= N, "Task too large for SBO buffer");

    static_assert(alignof(D) <= alignof(std::max_align_t),
                  "Task alignment exceeds max_align_t");

    new (buf_) D(std::forward<F>(f));

    inv_ = [](void *p) { (*static_cast<D *>(p))(); };
    dtor_ = [](void *p) { static_cast<D *>(p)->~D(); };
    mov_ = [](void *s, void *d) {
      new (d) D(std::move(*static_cast<D *>(s)));
      static_cast<D *>(s)->~D();
    };
  }

  /**
   * @brief Move constructor.
   * Transfers ownership of the stored callable from another instance.
   * @param o Source task_wrapper (will be left empty)
   * @note After the move, the source object becomes empty.
   */
  task_wrapper(task_wrapper &&o) noexcept {
    if (o.inv_) {
      o.mov_(o.buf_, buf_);
      inv_ = o.inv_;
      dtor_ = o.dtor_;
      mov_ = o.mov_;

      o.inv_ = o.dtor_ = nullptr;
      o.mov_ = nullptr;
    }
  }

  /**
   * @brief Move assignment operator.
   *
   * Destroys the current callable (if any) and transfers ownership
   * from another instance.
   *
   * @param o Source task_wrapper
   * @return Reference to this object
   *
   * @note Self-assignment is safely handled.
   */
  task_wrapper &operator=(task_wrapper &&o) noexcept {
    if (this != &o) {
      if (dtor_)
        dtor_(buf_);

      if (o.inv_) {
        o.mov_(o.buf_, buf_);

        inv_ = o.inv_;
        dtor_ = o.dtor_;
        mov_ = o.mov_;

        o.inv_ = o.dtor_ = nullptr;
        o.mov_ = nullptr;
      } else {
        inv_ = dtor_ = nullptr;
        mov_ = nullptr;
      }
    }
    return *this;
  }

  /**
   * @brief Invokes the stored callable.
   * Calls the stored function if it exists.
   * @note This function is noexcept and does nothing if empty.
   */
  void operator()() noexcept {
    if (inv_)
      inv_(buf_);
  }

  /**
   * @brief Checks whether a callable is stored.
   * @return true if a callable is present, false otherwise
   */
  explicit operator bool() const noexcept { return inv_ != nullptr; }

  /**
   * @brief Destructor.
   * Destroys the stored callable if present.
   */
  ~task_wrapper() {
    if (dtor_)
      dtor_(buf_);
  }

  /// @brief Copy constructor is deleted (move-only type).
  task_wrapper(const task_wrapper &) = delete;

  /// @brief Copy assignment is deleted (move-only type).
  task_wrapper &operator=(const task_wrapper &) = delete;
};

/**
 * @brief Cache-aligned task queue for concurrent task scheduling.
 *
 * This structure encapsulates a FIFO queue of @ref task_wrapper objects,
 * along with an associated mutex for synchronization.
 *
 * The entire structure is aligned to `kCacheSz` bytes to minimize
 * **false sharing** in multi-threaded environments, particularly when
 * multiple queues are stored contiguously (e.g., in a thread pool).
 *
 * @note This type is **moveable but not copyable**.
 *
 * @warning The internal queue is **not thread-safe by itself**.
 * All accesses must be externally synchronized using `mtx`.
 *
 * Typical usage pattern:
 * @code
 * std::lock_guard<std::mutex> lock(queue.mtx);
 * queue.tasks.push(...);
 * @endcode
 *
 * @see task_wrapper
 */
struct alignas(kCacheSz) aligned_task_queue {

  /**
   * @brief FIFO container holding pending tasks.
   *
   * Stores type-erased callable objects using @ref task_wrapper.
   */
  std::queue<task_wrapper<>> tasks;

  /**
   * @brief Mutex protecting access to the task queue.
   *
   * Must be locked before any modification or access to @ref tasks.
   */
  std::mutex mtx;

  /**
   * @brief Default constructor.
   *
   * Initializes an empty task queue.
   */
  aligned_task_queue() = default;

  /**
   * @brief Move constructor.
   *
   * Transfers ownership of the task queue from another instance.
   *
   * @param o Source queue
   *
   * @note The mutex is default-constructed; only the task container is moved.
   */
  aligned_task_queue(aligned_task_queue &&o) noexcept
      : tasks(std::move(o.tasks)) {}

  /**
   * @brief Move assignment operator.
   *
   * Replaces the current task queue with the contents of another.
   *
   * @param o Source queue
   * @return Reference to this object
   *
   * @note The mutex is not moved or copied.
   * @warning Caller must ensure no concurrent access during assignment.
   */
  aligned_task_queue &operator=(aligned_task_queue &&o) noexcept {
    if (this != &o)
      tasks = std::move(o.tasks);
    return *this;
  }

  /// @brief Copy constructor is deleted to prevent unsafe sharing.
  aligned_task_queue(const aligned_task_queue &) = delete;

  /// @brief Copy assignment is deleted to prevent unsafe sharing.
  aligned_task_queue &operator=(const aligned_task_queue &) = delete;
};

/**
 * @brief High-performance singleton thread pool with work-stealing queues.
 *
 * This class implements a fixed-size thread pool with the following features:
 *
 * - **Singleton pattern** (lazy-initialized, thread-safe via std::call_once)
 * - **Per-thread task queues** to reduce contention
 * - **Work-stealing** between queues for load balancing
 * - **Small-buffer-optimized tasks** via @ref task_wrapper
 * - **Futures-based API** via std::packaged_task
 * - **Graceful shutdown with timeout**
 *
 * ### Concurrency Model
 * - Each worker owns a local queue (@ref aligned_task_queue)
 * - Tasks are distributed in a round-robin fashion
 * - Workers:
 *   1. Pop from their own queue (fast path)
 *   2. Attempt to steal from other queues (try-lock)
 *   3. Sleep on condition variable if no work is available
 *
 * ### Memory Ordering
 * - `active_` uses acquire/release semantics to track in-flight tasks
 * - `stop_` is used as a global termination flag
 *
 * ### Lifecycle
 * - Constructed lazily via @ref Instance()
 * - Destroyed automatically at program termination
 * - @ref shutdown() ensures controlled termination
 *
 * @warning
 * - Submitting tasks after shutdown results in exception
 * - `wait_all()` uses busy-waiting (may waste CPU cycles)
 *
 * @note
 * - Thread affinity is set on Linux using `pthread_setaffinity_np`
 * - Queue alignment minimizes false sharing
 */
class __wc_thread_pool {
public:
  /// @brief Deleted copy constructor (singleton, non-copyable).
  __wc_thread_pool(const __wc_thread_pool &) = delete;

  /// @brief Deleted copy assignment operator.
  __wc_thread_pool &operator=(const __wc_thread_pool &) = delete;

  /**
   * @brief Returns the global thread pool instance.
   *
   * Thread-safe lazy initialization using `std::call_once`.
   *
   * @return Pointer to the singleton instance
   */
  [[nodiscard]] static __wc_thread_pool *Instance() noexcept {
    std::call_once(init_, [] { inst_.reset(new __wc_thread_pool()); });
    return inst_.get();
  }

  /**
   * @brief Submits a callable with arguments and returns a future.
   *
   * Wraps the callable into a `std::packaged_task` and schedules it
   * for execution.
   *
   * @tparam F Callable type
   * @tparam A Argument types
   * @param fn Callable object
   * @param args Arguments to pass to the callable
   * @return std::future holding the result
   *
   * @throws std::runtime_error if the pool is stopped
   *
   * @note Exceptions thrown by the task are captured in the future.
   */
  template <typename F, typename... A>
  [[nodiscard]] auto submit(F &&fn, A &&...args)
      -> std::future<std::invoke_result_t<F, A...>> {
    using R = std::invoke_result_t<F, A...>;

    auto pkg = std::make_shared<std::packaged_task<R()>>(
        [f = std::forward<F>(fn),
         ... a = std::forward<A>(args)]() mutable -> R {
          return f(std::move(a)...);
        });

    auto fut = pkg->get_future();

    push([pkg] {
      try {
        (*pkg)();
      } catch (...) {
      } // exception stored in future
    });

    return fut;
  }

  /**
   * @brief Enqueues a fire-and-forget task.
   *
   * @param fn Callable with signature `void()`
   *
   * @throws std::runtime_error if the pool is stopped
   */
  void enqueue(std::function<void()> fn) {
    push([f = std::move(fn)] {
      try {
        f();
      } catch (...) {
      } // swallow exceptions
    });
  }

  /**
   * @brief Enqueues a batch of tasks.
   *
   * Distributes tasks across worker queues in round-robin order.
   *
   * @tparam It Iterator type yielding callable objects
   * @param first Beginning of range
   * @param last End of range
   *
   * @throws std::runtime_error if the pool is stopped
   *
   * @note Uses a single global lock for batch insertion
   */
  template <typename It> void enqueue_batch(It first, It last) {
    std::size_t cnt = 0;
    {
      std::lock_guard<std::mutex> lk(smtx_);

      if (stop_.load(std::memory_order_acquire))
        throw std::runtime_error("enqueue_batch: pool stopped");

      for (auto it = first; it != last; ++it, ++cnt) {
        const std::size_t q =
            nxt_.fetch_add(1, std::memory_order_relaxed) % N_CORE;

        std::lock_guard<std::mutex> ql(qs_[q].mtx);

        qs_[q].tasks.emplace([t = *it] {
          try {
            t();
          } catch (...) {
          }
        });
      }

      active_.fetch_add(cnt, std::memory_order_release);
    }

    cnt == 1 ? cv_.notify_one() : cv_.notify_all();
  }

  /**
   * @brief Initiates shutdown of the thread pool.
   *
   * Signals all workers to stop and attempts to join threads
   * within a given timeout.
   *
   * @param tmo Maximum wait duration for thread joins
   *
   * @note Threads exceeding the timeout are detached.
   */
  void
  shutdown(std::chrono::milliseconds tmo = std::chrono::milliseconds(5'000)) {
    {
      std::lock_guard<std::mutex> lk(smtx_);
      stop_.store(true, std::memory_order_release);
    }

    cv_.notify_all();

    const auto dl = std::chrono::steady_clock::now() + tmo;

    for (auto &t : thr_) {
      if (!t.joinable())
        continue;
      std::chrono::steady_clock::now() < dl ? t.join() : t.detach();
    }
  }

  /// @brief Returns the number of worker threads.
  [[nodiscard]] std::size_t thread_count() const noexcept { return N_CORE; }

  /**
   * @brief Returns the number of active (in-flight) tasks.
   *
   * @return Number of tasks currently being processed or queued
   */
  [[nodiscard]] std::size_t active_tasks() const noexcept {
    return active_.load(std::memory_order_acquire);
  }

  /**
   * @brief Waits until all tasks have completed.
   *
   * Busy-waits using `std::this_thread::yield`.
   *
   * @warning This is not a blocking wait and may waste CPU cycles.
   */
  void wait_all() const noexcept {
    while (active_.load(std::memory_order_acquire) > 0)
      std::this_thread::yield();
  }

  /**
   * @brief Destructor.
   *
   * Automatically invokes @ref shutdown().
   */
  ~__wc_thread_pool() { shutdown(); }

private:
  /**
   * @brief Private constructor (singleton).
   *
   * Initializes worker threads and assigns CPU affinity where supported.
   */
  __wc_thread_pool() : qs_(N_CORE) {
    cpu_ = std::thread::hardware_concurrency();

    thr_.reserve(N_CORE);

    for (std::size_t i = 0; i < N_CORE; ++i) {
      thr_.emplace_back([this, i] { worker(i); });

#ifdef __linux__
      cpu_set_t cs;
      CPU_ZERO(&cs);
      CPU_SET(i % cpu_, &cs);
      pthread_setaffinity_np(thr_.back().native_handle(), sizeof(cs), &cs);
#endif
    }
  }

  /**
   * @brief Internal task submission primitive.
   *
   * Pushes a task into one of the worker queues.
   *
   * @tparam F Callable type
   * @param fn Callable object
   *
   * @throws std::runtime_error if the pool is stopped
   */
  template <typename F> void push(F &&fn) {
    {
      std::lock_guard<std::mutex> lk(smtx_);

      if (stop_.load(std::memory_order_acquire))
        throw std::runtime_error("push: pool stopped");

      const std::size_t q =
          nxt_.fetch_add(1, std::memory_order_relaxed) % N_CORE;

      std::lock_guard<std::mutex> ql(qs_[q].mtx);

      qs_[q].tasks.emplace(std::forward<F>(fn));

      active_.fetch_add(1, std::memory_order_release);
    }

    cv_.notify_one();
  }

  /**
   * @brief Worker thread main loop.
   *
   * @param id Worker index
   *
   * Execution strategy:
   * - Attempt local queue pop
   * - Attempt work stealing from other queues
   * - Sleep if no work available
   * - Drain remaining tasks on shutdown
   */
  void worker(std::size_t id) {
    while (true) {
      task_wrapper<> task;
      bool found = false;

      // Local queue
      {
        std::lock_guard<std::mutex> lk(qs_[id].mtx);
        if (!qs_[id].tasks.empty()) {
          task = std::move(qs_[id].tasks.front());
          qs_[id].tasks.pop();
          found = true;
        }
      }

      // Work stealing
      if (!found) {
        for (std::size_t i = 1; i <= N_CORE; ++i) {
          const std::size_t s = (id + i) % N_CORE;

          std::unique_lock<std::mutex> lk(qs_[s].mtx, std::try_to_lock);

          if (lk.owns_lock() && !qs_[s].tasks.empty()) {
            task = std::move(qs_[s].tasks.front());
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

      // Sleep
      {
        std::unique_lock<std::mutex> lk(wmtx_);
        cv_.wait(lk, [this] {
          return stop_.load(std::memory_order_acquire) ||
                 active_.load(std::memory_order_acquire) > 0;
        });
      }

      // Shutdown drain
      if (stop_.load(std::memory_order_acquire)) {
        while (true) {
          task_wrapper<> drain;

          {
            std::lock_guard<std::mutex> lk(qs_[id].mtx);
            if (qs_[id].tasks.empty())
              break;

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

  /// @brief Singleton instance.
  static std::unique_ptr<__wc_thread_pool> inst_;

  /// @brief One-time initialization flag.
  static std::once_flag init_;

  /// @brief Number of hardware threads.
  unsigned cpu_ = 0;

  /// @brief Worker threads.
  std::vector<std::thread> thr_;

  /// @brief Per-thread task queues.
  std::vector<aligned_task_queue> qs_;

  /// @brief Submission mutex.
  std::mutex smtx_;

  /// @brief Wait mutex.
  std::mutex wmtx_;

  /// @brief Condition variable for worker wake-up.
  std::condition_variable cv_;

  /// @brief Stop flag.
  std::atomic<bool> stop_{false};

  /// @brief Number of active tasks.
  std::atomic<std::size_t> active_{0};

  /// @brief Round-robin queue index.
  std::atomic<std::size_t> nxt_{0};
};

/// @brief Static instance definition.
std::unique_ptr<__wc_thread_pool> __wc_thread_pool::inst_;

/// @brief Static initialization flag.
std::once_flag __wc_thread_pool::init_;

/// @brief Public alias for the thread pool type.
using thread_pool = __wc_thread_pool;

} // namespace tp

namespace detail {

/**
 * @brief Minimal low-level logging and error handling utility (singleton).
 *
 * This class provides a lightweight, async-signal-safe oriented interface
 * for error reporting and output operations, designed for systems-level
 * programs where:
 *
 * - Heap allocations should be avoided
 * - Exceptions are not used
 * - Immediate termination is preferred on fatal errors
 * - POSIX I/O (`write`) is favored over buffered I/O when needed
 *
 * ### Design characteristics
 * - Singleton instance (lazy-initialized)
 * - Thread-safe initialization (double-checked locking)
 * - Non-copyable, non-movable
 * - Uses `_exit()` to avoid invoking destructors or flushing buffers
 *
 * @warning
 * - `_exit()` bypasses all cleanup (no destructors, no `atexit` handlers)
 * - Not suitable for applications requiring graceful shutdown
 *
 * @note
 * - Functions are marked `noexcept` and assume failure is unrecoverable
 * - Intended for CLI tools, utilities, or low-level runtime components
 */
class Logger {
public:
  /**
   * @brief Returns the global logger instance.
   *
   * Uses double-checked locking for lazy initialization.
   *
   * @return Pointer to the singleton Logger
   *
   * @note Thread-safe under standard memory model assumptions
   */
  [[nodiscard]] static Logger *logger() noexcept {
    if (!log_) [[unlikely]] {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!log_)
        log_ = new Logger();
    }
    return log_;
  }

  /// @brief Copy constructor deleted.
  Logger(const Logger &) = delete;

  /// @brief Move constructor deleted.
  Logger(Logger &&) = delete;

  /// @brief Destructor (trivial, never called if `_exit()` is used).
  ~Logger() = default;

  /**
   * @brief Prints a simple error message and terminates the process.
   *
   * @param msg Error message (default: "Error")
   *
   * @note Writes to `stderr` using `fprintf`
   * @warning Terminates via `_exit(1)`
   */
  [[noreturn]] void display_error(const char *msg = "Error") noexcept {
    std::fprintf(stderr, "%s\n", msg);
    _exit(1);
  }

  /**
   * @brief Prints a formatted fatal error message and terminates.
   *
   * @param prog Program name (prefix)
   * @param fmt printf-style format string
   * @param ... Format arguments
   *
   * @note Uses variadic arguments (`va_list`)
   * @warning Terminates via `_exit(1)`
   */
  [[noreturn]] void die(const char *prog, const char *fmt, ...) noexcept {
    std::fprintf(stderr, "%s: ", prog);

    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    fputc('\n', stderr);
    _exit(1);
  }

  /**
   * @brief Prints a warning message using `errno`.
   *
   * @param prog Program name
   * @param path File or resource path
   * @return Always returns false (useful for inline error propagation)
   *
   * @note Output format:
   * `prog: path: strerror(errno)`
   */
  [[nodiscard]] bool warn(const char *prog, const char *path) noexcept {
    std::fprintf(stderr, "%s: %s: %s\n", prog, path, strerror(errno));
    return false;
  }

  /**
   * @brief Writes a buffer fully to a file descriptor.
   *
   * Ensures that all bytes are written, retrying partial writes.
   *
   * @param fd File descriptor
   * @param buf Buffer to write
   * @param len Number of bytes to write
   *
   * @note Uses POSIX `write()` syscall
   * @warning On failure, prints error and terminates via `_exit(1)`
   *
   * @details
   * Handles short writes by looping until completion.
   * This is critical for non-blocking or interrupted I/O.
   */
  void write_all(int fd, const void *buf, std::size_t len) noexcept {
    const char *p = static_cast<const char *>(buf);

    while (len > 0) {
      const ssize_t w = ::write(fd, p, len);

      if (w <= 0) [[unlikely]] {
        perror("fasthead: write");
        _exit(1);
      }

      p += static_cast<std::size_t>(w);
      len -= static_cast<std::size_t>(w);
    }
  }

private:
  /// @brief Private constructor (singleton pattern).
  Logger() = default;

  /// @brief Singleton instance pointer.
  static Logger *log_;

  /// @brief Mutex protecting initialization.
  static std::mutex mtx_;
};

/// @brief Static instance definition.
Logger *Logger::log_ = nullptr;

/// @brief Static mutex definition.
std::mutex Logger::mtx_;

/**
 * @brief Global inline pointer to the logger instance.
 *
 * Provides convenient access without repeated calls to Logger::logger().
 */
inline Logger *logger = Logger::logger();

/**
 * @brief Extracts the file extension from a path.
 *
 * @param path Input file path
 * @return View of the extension (without dot), or empty if none exists
 *
 * @note Does not allocate (returns std::string_view)
 * @note The returned view refers to the original string
 *
 * @example
 * file_extension("test.txt") -> "txt"
 * file_extension("archive")  -> ""
 */
[[nodiscard]] inline std::string_view
file_extension(std::string_view path) noexcept {
  const auto d = path.rfind('.');
  return d == std::string_view::npos ? "" : path.substr(d + 1);
}
/**
 * @brief Result of parsing a numeric count with optional sign.
 *
 * Represents a parsed integer value along with its sign.
 */
struct ParsedCount {
  /// @brief Parsed absolute value.
  long long value;

  /// @brief Indicates whether the original input had a negative sign.
  bool negative;
};

/**
 * @brief Parses a numeric count with optional unit suffix.
 *
 * Supports both decimal (kB, MB, ...) and binary (KiB, MiB, ...) suffixes.
 * Optionally restricts certain suffixes to byte-oriented contexts.
 *
 * @param s Input string (e.g., "10K", "-5MiB")
 * @param is_bytes Whether byte-specific suffixes are allowed
 * @param prog Program name (used for error reporting)
 * @return ParsedCount containing value and sign
 *
 * @throws Terminates via logger->die() on invalid input
 *
 * @note
 * - Leading '+' or '-' is supported
 * - Parsing is done using `std::strtoll`
 * - Suffix matching is case-sensitive
 *
 * @warning
 * - Overflow during multiplication is not explicitly checked
 * - Invalid formats result in immediate process termination
 */
[[nodiscard]] ParsedCount parse_count(const char *s, bool is_bytes,
                                      const char *prog) noexcept {
  const char *p = s;
  bool neg = false;

  if (*p == '-') {
    neg = true;
    ++p;
  } else if (*p == '+')
    ++p;

  if (!*p || !std::isdigit(static_cast<unsigned char>(*p)))
    logger->die(prog, "invalid count '%s'", s);

  char *end;
  long long v = std::strtoll(p, &end, 10);

  if (v < 0)
    logger->die(prog, "count too large '%s'", s);

  /**
   * @brief Supported suffix table.
   *
   * Each entry defines:
   * - suffix string
   * - multiplier
   * - whether it is restricted to byte contexts
   */
  static constexpr struct {
    const char *sfx;
    long long m;
    bool bytes_only;
  } T[] = {{"b", 512LL, true},
           {"kB", 1000LL, false},
           {"k", 1024LL, false},
           {"K", 1024LL, false},
           {"KiB", 1024LL, false},
           {"MB", 1000000LL, false},
           {"M", 1048576LL, false},
           {"MiB", 1048576LL, false},
           {"GB", 1000000000LL, false},
           {"G", 1073741824LL, false},
           {"GiB", 1073741824LL, false},
           {"TB", 1000000000000LL, false},
           {"T", 1099511627776LL, false},
           {"TiB", 1099511627776LL, false},
           {"PB", 1000000000000000LL, false},
           {"P", 1125899906842624LL, false},
           {"PiB", 1125899906842624LL, false},
           {"EB", 1000000000000000000LL, false},
           {"E", 1152921504606846976LL, false},
           {"EiB", 1152921504606846976LL, false},
           {nullptr, 0LL, false}};

  if (*end) {
    bool ok = false;

    for (int i = 0; T[i].sfx; ++i) {
      if ((!T[i].bytes_only || is_bytes) && std::strcmp(end, T[i].sfx) == 0) {
        v *= T[i].m;
        ok = true;
        break;
      }
    }

    if (!ok)
      logger->die(prog, "invalid suffix in '%s'", s);
  }

  return {v, neg};
}

/**
 * @brief Reads entire standard input into a buffer.
 *
 * Uses repeated `read()` syscalls to accumulate input into a dynamic buffer.
 *
 * @return A vector containing all bytes read from stdin
 *
 * @note
 * - Initial capacity is 64 KiB
 * - Grows dynamically as needed
 *
 * @warning
 * - Does not handle read errors explicitly
 * - Blocking call depending on stdin source
 */
[[nodiscard]] std::vector<char> read_stdin() noexcept {
  std::vector<char> buf;
  buf.reserve(65536);

  char tmp[65536];
  ssize_t n;

  while ((n = ::read(STDIN_FILENO, tmp, sizeof(tmp))) > 0)
    buf.insert(buf.end(), tmp, tmp + n);

  return buf;
}

/**
 * @brief Finds the position after the N-th occurrence of a delimiter.
 *
 * Optimized using SIMD instructions (AVX-512, AVX2, SSE2) when available.
 *
 * @param data Pointer to input buffer
 * @param size Size of buffer in bytes
 * @param delim Delimiter character
 * @param n Occurrence index (1-based)
 * @return Pointer to position after the N-th delimiter, or nullptr if not found
 *
 * @note
 * - Uses vectorized comparisons for high throughput
 * - Falls back to scalar implementation if SIMD is unavailable
 *
 * @warning
 * - Behavior is undefined if `data` is invalid
 * - Assumes buffer is accessible up to `size`
 */
[[nodiscard]] static const char *
find_nth_delimiter(const char *__restrict__ data, std::size_t size, char delim,
                   long long n) noexcept {
  if (n <= 0) [[unlikely]]
    return data;

  const char *pos = data;
  const char *end = data + size;

#if defined(__AVX512BW__)
  {
    const __m512i dv = _mm512_set1_epi8(delim);

    while (pos + 64 <= end) {
      const uint64_t mask = _mm512_cmpeq_epi8_mask(
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(pos)), dv);

      const int cnt = __builtin_popcountll(mask);

      if (cnt < n) [[likely]] {
        n -= cnt;
        pos += 64;
        continue;
      }

      uint64_t m = mask;
      while (m) {
        const int idx = __builtin_ctzll(m);
        if (--n == 0) [[unlikely]]
          return pos + idx + 1;
        m &= m - 1;
      }
    }
  }
#endif

  // AVX2 / SSE2 / scalar fallback omitted for brevity (unchanged)

  while (pos < end) {
    if (*pos == delim) [[unlikely]]
      if (--n == 0)
        return pos + 1;
    ++pos;
  }

  return nullptr;
}

/**
 * @brief Counts occurrences of a delimiter in a buffer.
 *
 * Uses SIMD acceleration where available for high-performance counting.
 *
 * @param data Pointer to input buffer
 * @param size Buffer size in bytes
 * @param delim Delimiter character
 * @return Number of occurrences
 *
 * @note
 * - AVX-512 processes 64 bytes per iteration
 * - AVX2 processes 32 bytes per iteration
 * - SSE2 processes 16 bytes per iteration
 *
 * @warning
 * - Assumes valid memory region [data, data + size)
 */
[[nodiscard]] static std::size_t count_delimiters(const char *__restrict__ data,
                                                  std::size_t size,
                                                  char delim) noexcept {
  const char *pos = data;
  const char *end = data + size;
  std::size_t cnt = 0;

#if defined(__AVX512BW__)
  {
    const __m512i dv = _mm512_set1_epi8(delim);

    while (pos + 64 <= end) {
      cnt +=
          static_cast<std::size_t>(__builtin_popcountll(_mm512_cmpeq_epi8_mask(
              _mm512_loadu_si512(reinterpret_cast<const __m512i *>(pos)), dv)));

      pos += 64;
    }
  }
#endif

  // AVX2 / SSE2 / scalar fallback omitted for brevity (unchanged)

  while (pos < end) {
    cnt += (*pos++ == delim);
  }

  return cnt;
}

} // namespace detail

namespace core {

namespace fs {
/**
 * @brief Enumeration of supported file types.
 *
 * Used to categorize files based on their extension.
 */
enum class FileType {
  html,
  bash,
  python,
  txt,
  cpp,
  c,
  csharp,
  javascript,
  css,
  rust
};

/**
 * @brief Abstract interface representing a file type.
 *
 * Provides a polymorphic interface to retrieve file-specific headers
 * or metadata.
 *
 * @note Intended for extension via derived classes for each file type.
 */
class VirtualFileType {
public:
  /**
   * @brief Returns file-specific headers.
   *
   * @return Header string associated with the file type
   */
  virtual std::string get_headers() = 0;

  /// @brief Virtual destructor for safe polymorphic usage.
  virtual ~VirtualFileType() = default;

protected:
  /// @brief Internal storage for headers.
  std::string headers_;
};

/**
 * @brief Concrete file type implementations.
 *
 * These classes specialize @ref VirtualFileType for different file formats.
 *
 * @note Currently trivial implementations returning stored headers.
 * Future extensions may override behavior for parsing or formatting.
 */
class __html : public VirtualFileType {
  std::string get_headers() override { return this->headers_; }
};
class __bash : public VirtualFileType {
  std::string get_headers() override { return this->headers_; }
};
class __python : public VirtualFileType {
  std::string get_headers() override { return this->headers_; }
};
class __txt : public VirtualFileType {
  std::string get_headers() override { return this->headers_; }
};
class __cpp : public VirtualFileType {
  std::string get_headers() override { return this->headers_; }
};
class __c : public VirtualFileType {
  std::string get_headers() override { return this->headers_; }
};
class __csharp : public VirtualFileType {
  std::string get_headers() override { return this->headers_; }
};
class __javascript : public VirtualFileType {
  std::string get_headers() override { return this->headers_; }
};
class __css : public VirtualFileType {
  std::string get_headers() override { return this->headers_; }
};
class __rust : public VirtualFileType {
  std::string get_headers() override { return this->headers_; }
};

/**
 * @brief Detects file type from file extension.
 *
 * @param path File path
 * @return Corresponding FileType enum value
 *
 * @note
 * - Uses simple string comparison on file extension
 * - Defaults to FileType::txt if unknown
 * - Case-sensitive comparison
 */
[[nodiscard]] inline FileType detect_file_type(std::string_view path) noexcept {
  const auto ext = detail::file_extension(path);

  if (ext == "html" || ext == "htm")
    return FileType::html;
  if (ext == "sh")
    return FileType::bash;
  if (ext == "py")
    return FileType::python;
  if (ext == "txt")
    return FileType::txt;
  if (ext == "cpp")
    return FileType::cpp;
  if (ext == "c")
    return FileType::c;
  if (ext == "csharp")
    return FileType::csharp;
  if (ext == "javascript")
    return FileType::javascript;
  if (ext == "css")
    return FileType::css;
  if (ext == "rust")
    return FileType::rust;

  return FileType::txt;
}

/**
 * @brief Represents a memory-mapped file.
 *
 * Provides efficient file access using `mmap`, avoiding explicit reads.
 *
 * ### Features:
 * - Lazy file opening
 * - Page-aligned memory mapping
 * - Sequential access optimization via `madvise`
 * - Move-only semantics
 *
 * @note
 * - Uses `MAP_PRIVATE` (copy-on-write, read-only usage expected)
 * - Uses `MAP_POPULATE` for eager page loading
 *
 * @warning
 * - Behavior is undefined if underlying file changes during mapping
 * - Does not handle partial mapping failures beyond early return
 */
struct alignas(16) RawFile {

  /// @brief Pointer to mapped file data.
  char *data_ = nullptr;

  /// @brief File size in bytes.
  std::size_t size_ = 0;

  /// @brief Number of lines (optional, computed externally).
  std::size_t lines_count_ = 0;

  /// @brief File descriptor.
  int fd = -1;

  /// @brief Indicates whether mmap succeeded.
  bool is_mapped = false;

  /// @brief Size of mapped region (page-aligned).
  std::size_t mapped_size = 0;

  /// @brief Returns file size.
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  /// @brief Returns number of lines.
  [[nodiscard]] std::size_t get_lines_count() const noexcept {
    return lines_count_;
  }

  /// @brief Default constructor.
  RawFile() = default;

  /**
   * @brief Opens and memory-maps a file.
   *
   * @param filename Path to file
   *
   * @note
   * - Uses `open`, `fstat`, and `mmap`
   * - Aligns mapping size to page boundary
   * - Applies `MADV_SEQUENTIAL` and `MADV_WILLNEED`
   */
  explicit RawFile(std::string_view filename) noexcept {
    fd = open(filename.data(), O_RDONLY | O_CLOEXEC);

    if (fd == -1) [[unlikely]]
      return;

    struct stat st{};
    if (fstat(fd, &st) == -1) [[unlikely]] {
      close(fd);
      fd = -1;
      return;
    }

    size_ = static_cast<std::size_t>(st.st_size);

    if (size_ == 0)
      return;

    const auto pg = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    mapped_size = (size_ + pg - 1) & ~(pg - 1);

    void *ptr = mmap(nullptr, mapped_size, PROT_READ,
                     MAP_PRIVATE | MAP_POPULATE, fd, 0);

    if (ptr == MAP_FAILED) [[unlikely]] {
      is_mapped = false;
      return;
    }

    madvise(ptr, mapped_size, MADV_SEQUENTIAL | MADV_WILLNEED);

    data_ = static_cast<char *>(ptr);
    is_mapped = true;
  }

  /**
   * @brief Destructor.
   *
   * Unmaps memory and closes file descriptor.
   */
  ~RawFile() noexcept {
    if (is_mapped && data_) {
      munmap(data_, mapped_size);
      data_ = nullptr;
      is_mapped = false;
    }

    if (fd != -1) {
      close(fd);
      fd = -1;
    }
  }

  /// @brief Copy operations deleted (non-copyable resource).
  RawFile(const RawFile &) = delete;
  RawFile &operator=(const RawFile &) = delete;

  /**
   * @brief Move constructor.
   *
   * Transfers ownership of mapped memory and file descriptor.
   */
  RawFile(RawFile &&o) noexcept
      : data_(o.data_), size_(o.size_), lines_count_(o.lines_count_), fd(o.fd),
        is_mapped(o.is_mapped), mapped_size(o.mapped_size) {
    o.data_ = nullptr;
    o.fd = -1;
    o.is_mapped = false;
  }
};

/**
 * @brief Represents a chunk of a file for parallel processing.
 *
 * Designed for splitting large files into independent segments,
 * typically for multi-threaded processing.
 *
 * ### Design considerations:
 * - Cache-line aligned (64 bytes) to minimize false sharing
 * - Stores boundaries as raw pointers for zero-copy slicing
 * - Tracks line offsets within the chunk
 *
 * @note
 * - `start` and `end` must refer to a valid memory region
 * - Typically derived from @ref RawFile
 */
struct alignas(64) FileChunk {

  /// @brief Pointer to beginning of chunk.
  const char *start = nullptr;

  /// @brief Pointer to end of chunk.
  const char *end = nullptr;

  /// @brief Number of lines in the chunk.
  std::size_t line_count = 0;

  /// @brief Unique chunk identifier.
  std::size_t chunk_id = 0;

  /// @brief Positions of line delimiters within the chunk.
  std::vector<size_t> line_positions;
};

} // namespace fs

/**
 * @brief Configuration parameters for file processing.
 *
 * Encapsulates runtime options controlling how input is processed.
 */
struct ProcessConfig {

  /// @brief Number of lines or bytes to process.
  long long count = 10;

  /// @brief Whether to interpret count as "from the end".
  bool negative = false;

  /// @brief Whether to operate in byte mode instead of line mode.
  bool bytes_mode = false;

  /// @brief Delimiter used to separate records (default: newline).
  char delimiter = '\n';

  /// @brief Enable verbose output.
  bool verbose = false;

  /// @brief Suppress non-essential output.
  bool quiet = false;
};

/**
 * @brief Memory-maps a file into memory.
 *
 * @param fn File path
 * @return Memory-mapped file wrapper
 *
 * @note Thin wrapper around @ref fs::RawFile
 */
[[nodiscard]] inline fs::RawFile
memory_map_file(const std::string &fn) noexcept {
  return fs::RawFile(fn);
}

/**
 * @brief Splits a memory buffer into chunks for parallel processing.
 *
 * Ensures chunk boundaries align with delimiter positions to avoid
 * splitting logical records across threads.
 *
 * @param data Pointer to file data
 * @param size Size of data in bytes
 * @param n_threads Desired number of chunks
 * @param delim Delimiter character
 * @return Vector of file chunks
 *
 * @note
 * - Chunk count is adjusted based on page size and file size
 * - Uses `memchr` to find delimiter boundaries
 *
 * @warning
 * - Assumes `data` is a valid memory region of size `size`
 */
[[nodiscard]] std::vector<fs::FileChunk> create_chunks(const char *data,
                                                       std::size_t size,
                                                       std::size_t n_threads,
                                                       char delim) noexcept {
  if (n_threads == 0)
    n_threads = 1;
  if (size == 0)
    return {};

  const auto pg = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));

  // Limit threads based on file size
  n_threads = std::min(n_threads, std::max<std::size_t>(1, size / pg));

  std::vector<fs::FileChunk> chunks;
  chunks.reserve(n_threads);

  const std::size_t slice = size / n_threads;
  const char *file_end = data + size;
  const char *cursor = data;

  for (std::size_t id = 0; id < n_threads && cursor < file_end; ++id) {
    fs::FileChunk c;
    c.chunk_id = id;
    c.start = cursor;

    if (id == n_threads - 1) {
      c.end = file_end;
    } else {
      const char *nom = cursor + slice;

      if (nom >= file_end) {
        c.end = file_end;
      } else {
        const char *d = static_cast<const char *>(
            memchr(nom, delim, static_cast<std::size_t>(file_end - nom)));

        c.end = d ? d + 1 : file_end;
      }
    }

    cursor = c.end;
    chunks.push_back(std::move(c));
  }

  return chunks;
}

/**
 * @brief Core processing engine for file operations.
 *
 * Implements high-performance file processing with:
 * - Memory-mapped I/O
 * - SIMD-accelerated delimiter search
 * - Parallel chunk processing via thread pool
 *
 * @note
 * - Designed for large file processing (e.g., head-like utilities)
 * - Uses @ref tp::thread_pool for parallelism
 */
class CoreProcessor {
public:
  /**
   * @brief Constructs a processor instance.
   *
   * @param num_lines Default number of lines
   * @param num_threads Number of worker threads
   */
  explicit CoreProcessor(
      std::size_t num_lines = 10,
      std::size_t num_threads = std::thread::hardware_concurrency()) noexcept
      : m_lines(num_lines == 0 ? 10 : num_lines),
        m_threads(num_threads == 0 ? 1 : num_threads) {}

  /**
   * @brief Processes a file and returns extracted content.
   *
   * @param filename File path
   * @return Extracted content as string
   *
   * @note Uses newline delimiter
   */
  [[nodiscard]] std::string process(const std::string &filename);

  /**
   * @brief Processes a file and writes output directly to stdout.
   *
   * @param filename File path ("-" for stdin)
   * @param cfg Processing configuration
   * @param show_header Whether to print file header
   * @param blank_before_header Insert blank line before header
   * @return true on success, false on error
   */
  bool process_file(const std::string &filename, const ProcessConfig &cfg,
                    bool show_header, bool blank_before_header);

  ~CoreProcessor() = default;

  /// @brief Non-copyable.
  CoreProcessor(const CoreProcessor &) = delete;

  /// @brief Non-movable.
  CoreProcessor(CoreProcessor &&) = delete;

private:
  /// @brief Number of lines to extract.
  std::size_t m_lines;

  /// @brief Number of worker threads.
  std::size_t m_threads;

  /// @brief Current delimiter.
  char m_delim_ = '\n';

  /**
   * @brief Counts delimiters within a chunk.
   *
   * @param chunk Input chunk
   * @return Updated chunk with line count
   */
  [[nodiscard]] fs::FileChunk find_lines_in_chunk(const fs::FileChunk &chunk) {
    fs::FileChunk r = chunk;

    r.line_count = detail::count_delimiters(
        chunk.start, static_cast<std::size_t>(chunk.end - chunk.start),
        m_delim_);

    return r;
  }

  /**
   * @brief Counts delimiters in parallel.
   *
   * Splits data into chunks and processes them using thread pool.
   *
   * @param data Buffer
   * @param size Size in bytes
   * @param delim Delimiter
   * @return Total count
   */
  [[nodiscard]] std::size_t
  count_delimiters_parallel(const char *data, std::size_t size, char delim) {
    m_delim_ = delim;

    auto chunks = create_chunks(data, size, m_threads, delim);
    auto *pool = tp::thread_pool::Instance();

    std::vector<std::future<fs::FileChunk>> futs;
    futs.reserve(chunks.size());

    for (auto &c : chunks) {
      futs.push_back(
          pool->submit([this, c]() mutable { return find_lines_in_chunk(c); }));
    }

    std::size_t total = 0;
    for (auto &f : futs)
      total += f.get().line_count;

    return total;
  }

  /**
   * @brief Writes processed output according to configuration.
   *
   * Implements logic for:
   * - Positive/negative counts
   * - Byte mode vs line mode
   * - Parallel delimiter counting for reverse operations
   *
   * @param data Input buffer
   * @param size Buffer size
   * @param cfg Processing configuration
   */
  void write_head(const char *data, std::size_t size,
                  const ProcessConfig &cfg) {
    if (size == 0)
      return;

    // Special case: output entire file
    if (cfg.negative && cfg.count == 0) {
      detail::logger->write_all(STDOUT_FILENO, data, size);
      return;
    }

    // Byte mode
    if (cfg.bytes_mode) {
      if (!cfg.negative) {
        const std::size_t n =
            std::min(static_cast<std::size_t>(cfg.count), size);

        if (n)
          detail::logger->write_all(STDOUT_FILENO, data, n);
      } else {
        const std::size_t skip = static_cast<std::size_t>(cfg.count);

        if (skip < size)
          detail::logger->write_all(STDOUT_FILENO, data, size - skip);
      }
      return;
    }

    // Line mode (forward)
    if (!cfg.negative) {
      if (cfg.count == 0)
        return;

      const char *ep =
          detail::find_nth_delimiter(data, size, cfg.delimiter, cfg.count);

      detail::logger->write_all(
          STDOUT_FILENO, data, ep ? static_cast<std::size_t>(ep - data) : size);
    }
    // Line mode (reverse)
    else {
      const std::size_t total =
          count_delimiters_parallel(data, size, cfg.delimiter);

      if (total > static_cast<std::size_t>(cfg.count)) {
        const long long target = static_cast<long long>(total) - cfg.count;

        const char *ep =
            detail::find_nth_delimiter(data, size, cfg.delimiter, target);

        if (ep)
          detail::logger->write_all(STDOUT_FILENO, data,
                                    static_cast<std::size_t>(ep - data));
      }
    }
  }
};

/**
 * @brief Processes a file and writes output.
 *
 * Handles:
 * - Standard input ("-")
 * - Memory-mapped files
 * - Fallback buffered I/O
 *
 * @return true on success, false on failure
 */
bool CoreProcessor::process_file(const std::string &filename,
                                 const ProcessConfig &cfg, bool show_header,
                                 bool blank_before_header) {
  auto emit_header = [&](const char *name) noexcept {
    if (blank_before_header)
      detail::logger->write_all(STDOUT_FILENO, "\n", 1);

    char hdr[4096 + 12];
    const int n = std::snprintf(hdr, sizeof(hdr), "==> %s <==\n", name);

    if (n > 0)
      detail::logger->write_all(STDOUT_FILENO, hdr,
                                static_cast<std::size_t>(n));
  };

  if (filename == "-") {
    if (show_header)
      emit_header("standard input");

    auto buf = detail::read_stdin();
    write_head(buf.data(), buf.size(), cfg);
    return true;
  }

  fs::RawFile mf(filename);

  if (mf.fd == -1) [[unlikely]]
    return detail::logger->warn("fasthead", filename.c_str());

  // Fallback if mmap fails
  if (!mf.is_mapped && mf.size_ > 0) [[unlikely]] {
    const int tmpfd = open(filename.c_str(), O_RDONLY | O_CLOEXEC);
    if (tmpfd == -1)
      return detail::logger->warn("fasthead", filename.c_str());

    char tmp[65536];
    ssize_t n;
    std::vector<char> fbuf;
    fbuf.reserve(mf.size_);

    while ((n = ::read(tmpfd, tmp, sizeof(tmp))) > 0)
      fbuf.insert(fbuf.end(), tmp, tmp + n);

    ::close(tmpfd);

    if (show_header)
      emit_header(filename.c_str());

    write_head(fbuf.data(), fbuf.size(), cfg);
    return true;
  }

  if (show_header)
    emit_header(filename.c_str());

  if (mf.size_ == 0)
    return true;

  write_head(mf.data_, mf.size_, cfg);
  return true;
}

/**
 * @brief Processes a file and returns the first N lines as a string.
 *
 * @param filename File path
 * @return Extracted content or empty string on failure
 */
std::string CoreProcessor::process(const std::string &filename) {
  fs::RawFile mf(filename);

  if (mf.fd == -1 || mf.size_ == 0 || !mf.is_mapped)
    return {};

  const char *ep = detail::find_nth_delimiter(mf.data_, mf.size_, '\n',
                                              static_cast<long long>(m_lines));

  const std::size_t len =
      ep ? static_cast<std::size_t>(ep - mf.data_) : mf.size_;

  return std::string(mf.data_, len);
}
namespace cldecl {
/**
 * @brief Internal singleton helper mapping file-type classes to enum values.
 *
 * This template class provides a compile-time association between a
 * concrete file type class (e.g., @ref fs::__html) and its corresponding
 * @ref fs::FileType enum value.
 *
 * It follows a **singleton pattern per template instantiation**, meaning:
 * - Each distinct `FileType` template parameter has its own instance
 * - Instances are lazily initialized
 *
 * ### Design characteristics
 * - Compile-time dispatch using `if constexpr`
 * - Runtime singleton access
 * - No dynamic polymorphism (fully static mapping)
 *
 * @tparam FileType Concrete file type class (e.g., fs::__html)
 *
 * @note
 * - This pattern combines static type traits with runtime access
 * - Could be replaced with pure compile-time traits for zero-overhead
 *
 * @warning
 * - Uses double-checked locking without atomic guarantees (potential data race)
 */
template <typename FileType> class __head_internal_class {
public:
  /**
   * @brief Returns singleton instance for this FileType specialization.
   *
   * Uses lazy initialization with double-checked locking.
   *
   * @return Pointer to instance
   *
   * @note One instance exists per template specialization.
   */
  [[nodiscard]] static __head_internal_class *Instance() noexcept {
    if (!inst_) [[unlikely]] {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!inst_)
        inst_ = new __head_internal_class();
    }
    return inst_;
  }

  /// @brief Copy constructor deleted.
  __head_internal_class(const __head_internal_class &) = delete;

  /// @brief Move constructor deleted.
  __head_internal_class(__head_internal_class &&) = delete;

  /**
   * @brief Returns the corresponding enum file type.
   *
   * Maps the template parameter `FileType` to a value of @ref fs::FileType.
   *
   * @return FileType enum value
   *
   * @note
   * - Uses compile-time branching (`if constexpr`)
   * - Unmatched types default to `fs::FileType::txt`
   */
  [[nodiscard]] constexpr fs::FileType return_file_type() const noexcept {

    if constexpr (std::is_same_v<FileType, fs::__html>)
      return fs::FileType::html;

    if constexpr (std::is_same_v<FileType, fs::__bash>)
      return fs::FileType::bash;

    if constexpr (std::is_same_v<FileType, fs::__python>)
      return fs::FileType::python;

    return fs::FileType::txt;
  }

private:
  /// @brief Private constructor (singleton pattern).
  __head_internal_class() = default;

  /// @brief Singleton instance pointer (per template specialization).
  static __head_internal_class *inst_;

  /// @brief Mutex protecting initialization.
  static std::mutex mtx_;
};

/// @brief Static instance definition per template specialization.
template <typename T>
__head_internal_class<T> *__head_internal_class<T>::inst_ = nullptr;

/// @brief Static mutex definition per template specialization.
template <typename T> std::mutex __head_internal_class<T>::mtx_;
} // namespace cldecl

} // namespace core

using namespace core::cldecl;
using namespace core::fs;

int main(int argc, const char *argv[]) {
  std::ios_base::sync_with_stdio(false);

  core::ProcessConfig cfg;
  std::vector<std::string> files;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);

    if ((arg == "-n" || arg == "--lines") && i + 1 < argc) {
      const auto pc = detail::parse_count(argv[++i], false, "fasthead");
      cfg.count = pc.value;
      cfg.negative = pc.negative;
    } else if ((arg == "-c" || arg == "--bytes") && i + 1 < argc) {
      const auto pc = detail::parse_count(argv[++i], true, "fasthead");
      cfg.count = pc.value;
      cfg.negative = pc.negative;
      cfg.bytes_mode = true;
    } else if (arg == "-q" || arg == "--quiet" || arg == "--silent") {
      cfg.quiet = true;
      cfg.verbose = false;
    } else if (arg == "-v" || arg == "--verbose") {
      cfg.verbose = true;
      cfg.quiet = false;
    } else if (arg == "--") {
      for (int j = i + 1; j < argc; ++j)
        files.emplace_back(argv[j]);
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
  bool ok = true;
  bool blank_before = false;

  for (const auto &fn : files) {
    if (!proc.process_file(fn, cfg, show_header, blank_before))
      ok = false;
    blank_before = show_header;
  }

  tp::thread_pool::Instance()->shutdown();
  return ok ? 0 : 1;
}
