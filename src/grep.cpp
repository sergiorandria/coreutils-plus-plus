// This implementation is in progress ...

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <locale>
#endif

#include <climits>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#if __cplusplus <= 201703L
#error "Need to be compiled with C++20"
#endif // __cplusplus

namespace detail {
int hashFvn1(std::string_view __v) { return INT_MAX; }

} // namespace detail

namespace fs {
struct RawFile {
  char *data_;
  size_t size_;
  int fd = -1;

  RawFile() {}
  ~RawFile() {}
};

} // namespace fs

namespace tp {

#ifdef LIMIT_HARDWARE_USAGE
#ifndef BUILD_MAX_CORE
#define BUILD_MAX_CORE                                                         \
  64   // Limit max cpu to 64 for
       // processor with more than that
#endif // BUILD_MAX_CORE

#endif // LIMIT_HARDWARE_USAGE

// For small buffer optimization
#ifndef TASK_STACK_SIZE
#define TASK_STACK_SIZE 1024
#endif // TASK_STACK_SIZE

#define TRANSFER_OWNERSHIP(obj1, obj2)                                         \
  static_assert(typeid((obj1)) == typeid((obj2)));                             \
  (obj1).invoke_fn = (obj2).invoke_fn;                                         \
  (obj1).destroy_fn = (obj2).destroy_fn;                                       \
  (obj1).move_fn = (obj2).move_fn;                                             \
  (obj2).invoke_fn = nullptr;                                                  \
  (obj2).destroy_fn = nullptr;                                                 \
  (obj2).move_fn = nullptr

#ifndef N_CORE
// The explanation of this choice is inside fast-wc.cpp tp namespace.
// The main difference is that we can now compile grep.cpp with
// N_CORE defined at compile-time.
inline unsigned int N_CORE = std::thread::hardware_concurrency();
#endif // N_CORE

template <size_t stackSize = TASK_STACK_SIZE> class task_worker {
  // This class follows a multithreaded design pattern called
  // thread pool pattern. A task can only be performed three operations:
  // destroy, moved, invoke
public:
  using Priority = std::int8_t;

  task_worker() = default;

  // Create worker at compile time
  template <typename F> constexpr task_worker(F &&f) {
    using DecayF = std::decay_t<F>;

    if constexpr (is_small_v<DecayF>())
      new (stack) DecayF(std::forward<F>(f));
    else
      data = new char[stkSize + 1];

    invoke_fn = [](void *p) { (*static_cast<DecayF *>(p))(); };
    destroy_fn = [](void *p) { static_cast<DecayF *>(p)->~DecayF(); };
    move_fn = [](void *src, void *dst) {
      new (dst) DecayF(*static_cast<DecayF *>(src));
      static_cast<DecayF *>(src)->~DecayF();
    };

    has_work_policy = true;
  }

  task_worker(task_worker &&__o) {
    if (this->has_work_policy) {
      TRANSFER_OWNERSHIP(*this, __o);
    }
  }

  task_worker &operator=(task_worker &&__o) {
    static_assert(__o.has_work_policy == true,
                  "task_worker() other object is not fully initialized!");

    if (this->has_work_policy) {
      this->destroy_fn(stack);

      __o.move_fn(__o.stack, stack);
      TRANSFER_OWNERSHIP(*this, __o);
    }
  }

  template <typename T = std::function<void(void *)>>
  inline bool is_invoked_fn_invokable() const {
    return static_cast<T>(invoke_fn) != nullptr;
  }

  // Invocation operator
  void operator()() {
    if (is_invoked_fn_invokable<>())
      (void)invoke_fn(stack);
  }

  ~task_worker() {
    if (has_work_policy) {
      if (is_small_v<void>())
        destroy_fn(stack);
      else
        delete[] data;
    }
  }

  task_worker(const task_worker &&) = default;

  task_worker(const task_worker &) = delete;
  task_worker &operator=(const task_worker &) = delete;

private:
  Priority priority;
  size_t stkSize = stackSize;
  bool has_work_policy = false;

  std::function<void(void *)> invoke_fn = nullptr;
  std::function<void(void *)> destroy_fn = nullptr;
  std::function<void(void *)> move_fn = nullptr;

  template <typename T> constexpr bool is_small_v() const {
    return stkSize <= TASK_STACK_SIZE;
  }

  union {
    alignas(std::max_align_t) char stack[stackSize]; // Inlined buffer for SBO
    alignas(std::max_align_t) char *data; // Dynamically allocated buffer
  };
};

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif // CACHE_LINE_SIZE

// Any operations with the Pool queue is just
// a direct manipulation of std::queue tasks
// CACHE_LINE_SIZE = TASK_STACK_SIZE * 4,
// We should reconsider this value.
template <bool EnablePriorityScheduling = true>
struct alignas(CACHE_LINE_SIZE) aligned_task_queue {
  using TypeCondition = std::conditional<EnablePriorityScheduling,
                                         std::priority_queue<task_worker<>>,
                                         std::queue<task_worker<>>>;

  aligned_task_queue(aligned_task_queue &&__o) : tasks(std::move(__o.tasks)) {}

  aligned_task_queue &operator=(aligned_task_queue &&other) {
    tasks = std::move(other.tasks);
    return *this;
  }

  aligned_task_queue() = default;
  aligned_task_queue(const aligned_task_queue &) = delete;

  // Depends on compile-time flags.
  // std::priority_queue can be a little bit slower
  // on older computer.
  TypeCondition tasks;
  std::mutex atqMutex;
};

// If EnablePriorityScheduling is true,
// the task scheduling is enabled, and task A which
// have more priority than task B will run first.
template <bool EnablePriorityScheduling = true> class ThreadPool {
public:
  using Priority = std::int8_t;

#ifndef OPTIMIZE_SINGLETON_BUILD_TP
#warning "Singleton object build is expensive"
#endif // OPTIMIZE_SINGLETON_BUILD_TP

  static ThreadPool *Instance() {
    std::call_once(__iflag, []() { instance.reset(new ThreadPool); });
  }

  ThreadPool &operator=(ThreadPool &mmAlloc) {
    // Should be changed to something else
    return *this;
  }

  template <typename F, typename... Args>
    requires std::invocable<F, Args...>
  std::future<std::invoke_result_t<F, Args...>> submit_task(F &&f,
                                                            Args &&...args) {
    return _Tp_submit_task_helper(0, std::forward<F>(f),
                                  std::forward<Args>(args)...);
  }

  template <typename F, typename... Args>
    requires std::invocable<F, Args...>
  std::future<std::invoke_result_t<F>, Args...>
  submit_priority_task(Priority p, F &&f, Args &&...args) {
    static_assert(EnablePriorityScheduling,
                  "Priority scheduling should be enabled");

    return _Tp_submit_task_helper(p, std::forward<F>(f),
                                  std::forward<Args>(args)...);
  }

  ~ThreadPool() {}

private:
  static std::unique_ptr<ThreadPool<EnablePriorityScheduling>> instance;
  static std::once_flag __iflag;
  std::uint8_t cpu_core;
  std::mutex submitMutex;
  std::atomic<bool> __stop{false};
  std::vector<std::thread> threads;
  std::vector<aligned_task_queue<>> queues;
  std::condition_variable cv;
  std::mutex queueMutex;

  constexpr ThreadPool() : queues(N_CORE) {
    cpu_core = std::min(N_CORE, std::thread::hardware_concurrency());
    threads.reserve(cpu_core);

    for (int i = 0; i < N_CORE; ++i) {
      threads.emplace_back([this, i]() { worker_thread(i); });

// Pin threads to CPU cores,
// to gain performance with a
// better cache locality
#ifdef __linux__
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(i % cpu_core, &cpuset);

      pthread_setaffinity_np(threads.back().native_handle(), sizeof(cpuset),
                             &cpuset);
    }
#endif // __linux__
  }

  void worker_thread(size_t thread_id) {}

  template <typename F, typename... Args>
    requires std::invocable<F, Args...>
  std::future<std::invoke_result_t<F, Args...>>
  _Tp_submit_task_helper(F &&fn, Args &&...args) {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        [fn = std::forward<F>(fn),
         ... args = std::forward<Args>(args)]() mutable -> return_type {
          return fn(std::move(args)...);
        });

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(submitMutex);
      if (__stop.load(std::memory_order_acquire))
        throw std::runtime_error("Cannot submit to stopped thread pool");
    }
  }
};

template <bool EnablePriorityScheduling>
std::once_flag ThreadPool<EnablePriorityScheduling>::__iflag;

template <bool EnablePriorityScheduling>
std::unique_ptr<ThreadPool<EnablePriorityScheduling>>
    ThreadPool<EnablePriorityScheduling>::instance = nullptr;
} // namespace tp

namespace core {

class __grep_internal_class {
public:
  static __grep_internal_class *Instance() {

#ifdef OPTIMIZE_SINGLETON_BUILD
    // This was used for research purpose about time and space
    // complexity (will be deleted in the first stable version)
    static __grep_internal_class instance;
    return &instance;
#else
      if (instance == nullptr) {
        std::lock_guard<std::mutex> lock(gMutex);
        if (instance == nullptr) {
          instance = new __grep_internal_class();
        }
      }

      return instance;
#endif // OPTIMIZE_SINGLETON_BUILD
  }

  void process() const {}

  __grep_internal_class(const __grep_internal_class &gObj) = delete;
  __grep_internal_class(const __grep_internal_class &&gObj) = delete;

private:
  __grep_internal_class() {}

  static __grep_internal_class *instance;
  static std::mutex gMutex;
};

__grep_internal_class *__grep_internal_class::instance = nullptr;
std::mutex __grep_internal_class::gMutex;

} // namespace core

int main(int argc, const char *argv[]) {
  std::ios_base::sync_with_stdio(false);

  return 0;
}
