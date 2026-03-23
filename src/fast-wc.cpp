/*
 *  Copyright (C) 2026, Sergio Randriamihoatra
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <type_traits>
#endif // _GNU_SOURCE

#ifdef _WIN32
#include <windows.h>
#endif // _WIN32

#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <immintrin.h>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#if defined(_POSIX_VERSION) && _POSIX_VERSION >= 200809L

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#else
#error "POSIX version is inferior than 200809L"
#endif // _POSIX_VERSION

// For __wc_internal_class implementation
#define __wc_lib_use_std_atomic

// Exit code
#define WC_EXIT_CODE_FAILURE -1
#define WC_EXIT_CODE_SUCCESS 0
#define WC_EXIT_CODE_ERROR 1
#define WC_EXIT_CODE_UNKNOWN 2

#define LIKELY(x) (__builtin_expect(!!(x), 1))
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))

#define __FORCE_INLINE __attribute__((always_inline)) inline

#if not defined(__wc_lib_thread_pool)
#define __wc_lib_thread_pool

#ifdef _WIN32

#include <cstdio>
#include <strsafe.h>
#include <tchar.h>

// For logging

/**
 * @brief Displays the last Windows system error for a given function.
 *
 * This function retrieves the last error code using `GetLastError()`,
 * formats the corresponding system error message, and prints it along
 * with the provided function name.
 *
 * @param lpszFunction The name of the function that failed.
 */
void display_error(LPCTSTR lpszFunction)
// Routine Description:
// Retrieve and output the system error message for the last-error code
{
  LPVOID lpMsgBuf;
  LPVOID lpDisplayBuf;
  DWORD dw = GetLastError();
  FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                (LPTSTR)&lpMsgBuf, 0, NULL);
  lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
                                    (lstrlen((LPCTSTR)lpMsgBuf) +
                                     lstrlen((LPCTSTR)lpszFunction) +
                                     40) // account for format string
                                        * sizeof(TCHAR));
  if (FAILED(StringCchPrintf(
          (LPTSTR)lpDisplayBuf, LocalSize(lpDisplayBuf) / sizeof(TCHAR),
          TEXT("%s failed with error code %d as follows:\n%s"), lpszFunction,
          dw, lpMsgBuf))) {
    printf("FATAL ERROR: Unable to output error code.\n");
  }
  _tprintf(TEXT("ERROR: %s\n"), (LPCTSTR)lpDisplayBuf);
  LocalFree(lpMsgBuf);
  LocalFree(lpDisplayBuf);
}

#endif // _WIN32

namespace tp {

#define BUILD_MAX_CORE 64

// This constexpr template function is used for
// logging and type debugging
template <typename T, typename U>
constexpr bool is_decay_equal = std::is_same_v<std::decay_t<T>, U>;

#if defined(__linux) || defined(__linux__) || defined(__gnu_linux__)
// Tried to define it with constexpr but it didn't work
// That's why the type of N_CORE is different from the
// declaration.
//
// Note: A CPU can create more threads than the value returned
// by std::thread::hardware_concurrency(), but many threads can considerably
// slow down the CPU performance (The CPU is overwhelmed). So we have to
// moderate the number of threads a process can generate. On linux systems, the
// value of hardware_concurrency() can be obtained using the command : $ nproc
// --all
inline unsigned int N_CORE = std::thread::hardware_concurrency();

// It cannot be evaluated in a compile time expression.
// Unless we take the risk to oversubscribe threads, overallocate buffers and
// break scheduling assumptions if the N_CORE of the build machine is
// different from the runtime of the target machine. So, runtime detection is
// almost always the correct way.
// constexpr unsigned int N_CORE = N_CORE;

#elif defined(_WIN32)
namespace thread {

// Windows systems do not have default hardware_concurrency()
// implementation. There are many API and system calls which claim
// to return the number of core on the target machine, but their results
// are all incoherent. Here is an effort for a correct implementation of
// std::thread::hardware_concurrency() for msdos systems.
size_t hardware_concurrency() {
  size_t concurrency = 0;
  DWORD length = 0;

  if (GetLogicalProcessorInformationEx(RelationAll, nullptr, &length) !=
      FALSE) {
    return concurrency;
  }

  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    return concurrency;
  }

  using BufferType = std::unique_ptr<void, void (*)(void *)>;
  BufferType buffer(std::malloc(length, std::free));

  if (!buffer) {
    return concurrency;
  }

  unsigned char *mem = reinterpret_cast<unsigned char *>(buffer.get());
  if (GetLogicalProcessorInformationEx(
          RelationAll,
          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(mem),
          &length) == false) {
    return concurrency;
  }

  for (DWORD i = 0; i < length;) {
    auto *proc =
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(mem + i);
    if (proc->Relationship == RelationProcessorCore) {
      // Some documentation says that this scope can
      // be simplified to:
      // concurrency++
      //
      // This part of code is not tested yet
      for (WORD group = 0; group < proc->Processor.GroupCount; ++group) {
        for (KAFFINITY mask = proc->Processor.GroupMask[group].Mask; mask != 0;
             mask >>= 1) {
          concurrency += mask & 1;
        }
      }
    }

    i += proc->Size;
  }

  return concurrency;
}

// Then we define N_CORE  (The number of core the machine)
inline unsigned int _N_CORE = thread::hardware_concurrency();

#endif // __linux__

inline constexpr int BUFFER_SIZE = 128;
inline constexpr std::size_t CACHE_LINE_SIZE = 64;

// Small buffer optimization for task storage and caching
template <size_t BufferSize = BUFFER_SIZE> class task_wrapper {
private:
  alignas(std::max_align_t) char buffer[BufferSize];

  // Initializing pointer function to nullptr. A task can be
  // either invoked (and to the appropriate action according to invoked),
  // either destroyed, or moved (for memory optimization).
  void (*invoke_fn)(void *) = nullptr;
  void (*destroy_fn)(void *) = nullptr;
  void (*move_fn)(void *, void *) = nullptr;

public:
  task_wrapper() = default;

  // Construct a type-erased callable stored in an internal fixed-size buffer.
  // The callable must be invocable with signature void().
  // Compile-time checks ensure size, alignment, and callability constraints.
  // Stores function pointers for invocation, destruction, and move operations.
  template <typename F> task_wrapper(F &&f) {
    using DecayF = std::decay_t<F>;

    static_assert(sizeof(DecayF) <= BufferSize, "Task too large for buffer");
    static_assert(alignof(DecayF) <= alignof(std::max_align_t),
                  "Task alignment too large");

    static_assert(std::is_invocable_v<F &>, "Task must be callable");
    static_assert(std::is_invocable_r_v<void, F &>,
                  "Task must be callable with no args");

    new (buffer) DecayF(std::forward<F>(f));
    invoke_fn = [](void *p) { (*static_cast<DecayF *>(p))(); };

    destroy_fn = [](void *p) { static_cast<DecayF *>(p)->~DecayF(); };

    move_fn = [](void *src, void *dst) {
      new (dst) DecayF(std::move(*static_cast<DecayF *>(src)));
      static_cast<DecayF *>(src)->~DecayF();
    };
  }

  // Move constructor.
  // Transfers ownership of the stored callable from another wrapper.
  // Leaves the source wrapper in an empty state.
  task_wrapper(task_wrapper &&other) noexcept {
    if (other.invoke_fn) {
      other.move_fn(other.buffer, buffer);
      invoke_fn = other.invoke_fn;
      destroy_fn = other.destroy_fn;
      move_fn = other.move_fn;
      other.invoke_fn = nullptr;
      other.destroy_fn = nullptr;
      other.move_fn = nullptr;
    }
  }

  // Move assignment operator.
  // Destroys current callable if present, then moves the callable from source.
  // Ensures proper cleanup and leaves source empty.
  task_wrapper &operator=(task_wrapper &&other) noexcept {
    if (this != &other) {
      if (destroy_fn) {
        destroy_fn(buffer);
      }

      if (other.invoke_fn) {
        other.move_fn(other.buffer, buffer);
        invoke_fn = other.invoke_fn;
        destroy_fn = other.destroy_fn;
        move_fn = other.move_fn;
        other.invoke_fn = nullptr;
        other.destroy_fn = nullptr;
        other.move_fn = nullptr;
      } else {
        invoke_fn = nullptr;
        destroy_fn = nullptr;
        move_fn = nullptr;
      }
    }

    return *this;
  }

  // Invoke the stored callable if present
  void operator()() {
    if (invoke_fn) {
      invoke_fn(buffer);
    }
  }

  // Check if the wrapper currently holds a valid callable or not.
  explicit operator bool() const { return invoke_fn != nullptr; }

  ~task_wrapper() {
    if (destroy_fn) {
      destroy_fn(buffer);
    }
  }

  // Delete copy constructor
  task_wrapper(const task_wrapper &) = delete;
  task_wrapper &operator=(const task_wrapper &) = delete;
};

// Per-thread queue with cache line alignment
//
// `aligned_task_queue` encapsulates a FIFO queue of `task_wrapper<>`
// instances protected by a `std::mutex` for thread-safe access.
// The structure is explicitly aligned to `CACHE_LINE_SIZE` in order
// to minimize false sharing when multiple queues are placed in
// contiguous memory (e.g., thread pools or work-stealing schedulers).
//
// Move construction and move assignment transfer ownership of the
// underlying task queue, while the mutex is default-constructed
// since `std::mutex` is non-movable. Copy operations are disabled
// to preserve synchronization correctness and avoid accidental
// duplication of shared state.
struct alignas(CACHE_LINE_SIZE) aligned_task_queue {
  std::queue<task_wrapper<>> tasks;
  std::mutex mutex;

  aligned_task_queue(aligned_task_queue &&other) noexcept
      : tasks(std::move(other.tasks)) {
    // Note: mutex cannot be moved, so we just default construct it
  }

  aligned_task_queue &operator=(aligned_task_queue &&other) noexcept {
    if (this != &other) {
      tasks = std::move(other.tasks);
    }
    return *this;
  }

  aligned_task_queue() = default;
  aligned_task_queue(const aligned_task_queue &) = delete;
  aligned_task_queue &operator=(const aligned_task_queue &) = delete;
};

// Optimized thread pool with work stealing
class __wc_thread_pool {
public:
  __wc_thread_pool(const __wc_thread_pool &) = delete;
  __wc_thread_pool &operator=(const __wc_thread_pool &) = delete;

  // Singleton implementation, primary class are all
  // unique and private instance can be get via the public method
  // T* Instance() { ... };
  static __wc_thread_pool *Instance() {
    std::call_once(__init_flag,
                   []() { __instance.reset(new __wc_thread_pool()); });

    return __instance.get();
  }

  /**
   * @brief Submits a callable to the thread pool and returns a future to its
   * result.
   *
   * @tparam F Callable type.
   * @tparam Args Argument types.
   * @param fn Callable object to execute.
   * @param args Arguments forwarded to the callable.
   * @return std::future holding the result of the asynchronous computation.
   *
   * @details
   * Wraps the callable into a `std::packaged_task`, enqueues it using
   * round-robin distribution across aligned task queues, and notifies
   * one worker thread. Throws `std::runtime_error` if submission occurs
   * after the pool has been stopped.
   *
   * @note Exceptions thrown during task execution are caught internally
   * to prevent worker thread termination.
   */
  template <typename F, typename... Args>
  auto submit(F &&fn, Args &&...args)
      -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    // Use lambda instead of std::bind for better performance
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        [fn = std::forward<F>(fn),
         ... args = std::forward<Args>(args)]() mutable -> return_type {
          return fn(std::move(args)...);
        });

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(__submit_mutex);
      if (__stop.load(std::memory_order_acquire)) {
        throw std::runtime_error("Cannot submit to stopped thread pool");
      }

      // Round-robin task distribution to reduce contention
      size_t target_queue =
          __next_queue.fetch_add(1, std::memory_order_relaxed) % N_CORE;
      {
        std::unique_lock<std::mutex> queue_lock(__queues[target_queue].mutex);
        __queues[target_queue].tasks.emplace([task]() {
          try {
            (*task)();
          } catch (...) {
            // Prevent thread death from exceptions
          }
        });
      }
      __active_tasks.fetch_add(1, std::memory_order_release);
    }
    __tp_cv.notify_one();
    return res;
  }

  /**
   * @brief Enqueues a void task into the thread pool.
   *
   * @param task Callable object with signature `void()`.
   *
   * @details
   * Inserts the task into one of the internal queues using a round-robin
   * strategy to reduce contention between worker threads. The task is
   * wrapped in a lambda that catches all exceptions to prevent worker
   * termination. A worker thread is notified after successful insertion.
   *
   * @throws std::runtime_error if the thread pool has already been stopped.
   *
   * @note Uses atomic counters and per-queue mutexes to maintain
   * thread-safe submission and accurate tracking of active tasks.
   */
  void enqueue(std::function<void()> task) {
    {
      std::unique_lock<std::mutex> lock(__submit_mutex);
      if (__stop.load(std::memory_order_acquire)) {
        throw std::runtime_error("Cannot enqueue to stopped thread pool");
      }
      size_t target_queue =
          __next_queue.fetch_add(1, std::memory_order_relaxed) % N_CORE;
      {
        std::unique_lock<std::mutex> queue_lock(__queues[target_queue].mutex);
        __queues[target_queue].tasks.emplace([task = std::move(task)]() {
          try {
            task();
          } catch (...) {
            // Prevent thread death from exceptions
          }
        });
      }
      __active_tasks.fetch_add(1, std::memory_order_release);
    }
    __tp_cv.notify_one();
  }

  /**
   * @brief Enqueues a batch of void tasks into the thread pool.
   *
   * @tparam Iterator Iterator type pointing to callable objects
   *                  with signature `void()`.
   * @param begin Iterator to the first task in the batch.
   * @param end   Iterator to one-past-the-last task.
   *
   * @details
   * Inserts multiple tasks in a single submission phase to reduce
   * synchronization overhead. Tasks are distributed across internal
   * queues using a round-robin strategy to minimize contention.
   * Each task is wrapped in a lambda that catches all exceptions
   * to avoid worker thread termination.
   *
   * The number of active tasks is updated atomically once per batch.
   * Worker threads are notified using `notify_one()` for a single
   * task or `notify_all()` when multiple tasks are added.
   *
   * @throws std::runtime_error if the thread pool has been stopped.
   *
   * @note Submission is protected by a global submit mutex while
   * individual queues use dedicated mutexes for localized locking.
   */
  template <typename Iterator>
  void enqueue_batch(Iterator begin, Iterator end) {
    size_t count = 0;
    {
      std::unique_lock<std::mutex> lock(__submit_mutex);

      if (__stop.load(std::memory_order_acquire)) {
        throw std::runtime_error("Cannot enqueue to stopped thread pool");
      }

      for (auto it = begin; it != end; ++it) {
        size_t target_queue =
            __next_queue.fetch_add(1, std::memory_order_relaxed) % N_CORE;
        {
          std::unique_lock<std::mutex> queue_lock(__queues[target_queue].mutex);
          __queues[target_queue].tasks.emplace([task = *it]() {
            try {
              task();
            } catch (...) {
              // Prevent thread death from exceptions
            }
          });
        }
        ++count;
      }
      __active_tasks.fetch_add(count, std::memory_order_release);
    }
    if (count == 1) {
      __tp_cv.notify_one();
    } else {
      __tp_cv.notify_all();
    }
  }

  // Gracefully stops the thread pool and waits for worker threads to finish.
  //
  // Sets the stop flag to prevent new task submissions, then wakes all
  // worker threads so they can exit once pending work is done.
  //
  // The caller can specify a timeout duration (default: 5 seconds).
  // Threads are joined while time remains before the deadline.
  // If the timeout is exceeded, remaining joinable threads are detached
  // to avoid blocking shutdown indefinitely.
  //
  // Note:
  // - Detaching threads means they may continue running in background.
  // - Safe shutdown assumes workers periodically check the stop flag.
  void shutdown(
      std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
    {
      std::unique_lock<std::mutex> lock(__submit_mutex);
      __stop.store(true, std::memory_order_release);
    }
    __tp_cv.notify_all();

    // If a thread do not respond anything after deadline seconds,
    // detach it.
    auto deadline = std::chrono::steady_clock::now() + timeout;

    for (auto &thread : __threads) {
      if (thread.joinable()) {
        auto remaining = deadline - std::chrono::steady_clock::now();

        if (remaining > std::chrono::milliseconds(0)) {
          thread.join();
        } else {
          // Timeout reached, detach remaining threads
          thread.detach();
        }
      }
    }
  }

  ~__wc_thread_pool() { shutdown(); }

  // Returns the number of worker threads configured in the pool.
  //
  // This reflects the compile-time or fixed configuration (N_CORE),
  // not necessarily the number of currently running OS threads.
  size_t thread_count() const { return N_CORE; }

  // Returns the number of tasks currently marked as active.
  //
  // The value is read atomically with acquire semantics to ensure
  // visibility of task state across threads.
  size_t active_tasks() const {
    return __active_tasks.load(std::memory_order_acquire);
  }

  // Blocks until all active tasks have completed.
  //
  // Uses a spin-wait loop with std::this_thread::yield() to reduce
  // contention while waiting. Intended for short waits; for longer
  // waits a condition variable would be more efficient.
  void wait_all() {
    while (__active_tasks.load(std::memory_order_acquire) > 0) {
      std::this_thread::yield();
    }
  }

private:
  // Constructs the thread pool and spawns worker threads.
  //
  // Initializes the per-core task queues and determines the number
  // of hardware threads available on the system.
  //
  // Reserves space for worker thread objects to avoid reallocations,
  // then launches N_CORE worker threads. Each thread executes the
  // internal worker loop bound to its logical queue index.
  //
  // Assumptions:
  // - N_CORE represents the number of worker queues/threads desired.
  // - worker_thread(size_t id) handles task fetching and execution.
  __wc_thread_pool() : __queues(N_CORE) {
    __cpu_core = std::thread::hardware_concurrency();
    __threads.reserve(N_CORE);
    for (size_t i = 0; i < N_CORE; ++i) {
      __threads.emplace_back([this, i]() { worker_thread(i); });
#ifdef __linux__
      // Pin threads to CPU cores for better cache locality
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(i % __cpu_core, &cpuset);
      pthread_setaffinity_np(__threads.back().native_handle(), sizeof(cpuset),
                             &cpuset);
#endif
    }
  }

  // Main worker loop executed by each thread in the pool.
  //
  // Execution model:
  //  - Each worker primarily consumes tasks from its own queue to
  //    preserve cache locality and reduce contention.
  //  - If its queue is empty, the worker attempts work stealing
  //    from other queues using a non-blocking try_to_lock strategy.
  //  - When no work is available, the thread blocks on a condition
  //    variable until new tasks arrive or shutdown is requested.
  //
  // Task lifecycle:
  //  - Tasks are executed via task_wrapper<> abstraction.
  //  - After execution, the global active task counter is decremented
  //    using release semantics to synchronize with waiters.
  //
  // Shutdown behavior:
  //  - When stop flag is set, the worker drains remaining tasks from
  //    its own queue before exiting to ensure graceful termination.
  //  - No further stealing is performed during shutdown phase.
  //
  // Concurrency notes:
  //  - Queue-level mutexes minimize contention between workers.
  //  - Work stealing improves load balancing under uneven workloads.
  //  - Condition variable prevents busy spinning when the system is idle.
  void worker_thread(size_t thread_id) {
    while (true) {
      task_wrapper<> task;
      bool found_task = false;
      // Try to get task from own queue first
      {
        std::unique_lock<std::mutex> lock(__queues[thread_id].mutex);
        if (!__queues[thread_id].tasks.empty()) {
          task = std::move(__queues[thread_id].tasks.front());
          __queues[thread_id].tasks.pop();
          found_task = true;
        }
      }

      // If own queue is empty, try work stealing
      if (!found_task) {
        for (size_t i = 1; i <= N_CORE; ++i) {
          size_t steal_from = (thread_id + i) % N_CORE;
          std::unique_lock<std::mutex> lock(__queues[steal_from].mutex,
                                            std::try_to_lock);

          if (lock.owns_lock() && !__queues[steal_from].tasks.empty()) {
            task = std::move(__queues[steal_from].tasks.front());
            __queues[steal_from].tasks.pop();
            found_task = true;
            break;
          }
        }
      }

      if (found_task) {
        task();
        __active_tasks.fetch_sub(1, std::memory_order_release);
      } else {
        // No task found, wait for notification
        std::unique_lock<std::mutex> lock(__wait_mutex);
        __tp_cv.wait(lock, [this]() {
          return __stop.load(std::memory_order_acquire) ||
                 __active_tasks.load(std::memory_order_acquire) > 0;
        });

        // Check if we should stop
        if (__stop.load(std::memory_order_acquire)) {
          // Drain remaining tasks before exiting
          bool has_tasks = true;

          while (has_tasks) {
            has_tasks = false;
            std::unique_lock<std::mutex> lock(__queues[thread_id].mutex);

            if (!__queues[thread_id].tasks.empty()) {
              task = std::move(__queues[thread_id].tasks.front());
              __queues[thread_id].tasks.pop();
              has_tasks = true;
              lock.unlock();
              task();
              __active_tasks.fetch_sub(1, std::memory_order_release);
            }
          }
          return;
        }
      }
    }
  }

  static std::unique_ptr<__wc_thread_pool> __instance;
  static std::once_flag __init_flag;

  unsigned int __cpu_core;
  std::vector<std::thread> __threads;

  // Per-thread queues for reduced contention
  std::vector<aligned_task_queue> __queues;

  // Separate mutexes to reduce lock contention
  std::mutex __submit_mutex;
  std::mutex __wait_mutex;

  std::condition_variable __tp_cv;
  std::atomic<bool> __stop{false};
  std::atomic<size_t> __active_tasks{0};
  std::atomic<size_t> __next_queue{0};
};

// Static member initialization
std::unique_ptr<__wc_thread_pool> __wc_thread_pool::__instance;

std::once_flag __wc_thread_pool::__init_flag;

// Convenience alias
using thread_pool = __wc_thread_pool;

} // namespace tp

#endif // __wc_lib_thread_pool

#if not defined(__wc_argv_parse)
#define __wc_argv_parse

namespace detail {
inline constexpr int MAX_OPTIONS = 32;

/**
 * @brief Computes a 32-bit FNV-1a hash for a byte sequence.
 *
 * Implements the Fowler–Noll–Vo hash function (FNV-1a variant) over the
 * bytes contained in the given `std::string_view`. The implementation
 * uses manual loop unrolling to process four bytes per iteration in order
 * to reduce loop overhead and improve instruction-level parallelism on
 * modern CPUs. Remaining bytes are processed sequentially.
 *
 * The function is `constexpr`, enabling compile-time hashing when invoked
 * with constant expressions.
 *
 * @param __s The input string view representing the byte sequence to hash.
 *            The data is treated as raw bytes; no encoding assumptions are
 *            enforced beyond byte-wise processing.
 *
 * @return uint32_t The resulting 32-bit FNV-1a hash value.
 *
 * @pre
 * - `__s.data()` must reference a valid contiguous memory region of
 *   at least `__s.size()` bytes.
 *
 * @note
 * - Time complexity is O(n), where n is `__s.size()`.
 * - Deterministic across platforms given identical byte sequences.
 * - Uses standard FNV offset basis (2166136261u) and FNV prime (16777619u).
 * - Manual unrolling improves throughput mainly for medium to large inputs.
 *
 * @warning
 * - Not suitable for cryptographic purposes.
 * - Collisions are possible due to the 32-bit output size.
 *
 * @exception noexcept Guaranteed not to throw exceptions.
 */
constexpr uint32_t hash_fnv1a(std::string_view __s) noexcept {
  uint32_t hash = 2166136261u;
  const char *ptr = __s.data();
  size_t len = __s.size();

  // Process 4 bytes at a time for better performance,
  // It can be increased but doesn't affect too much the
  // runtime of the function.
  while (len >= 4) {
    hash ^= static_cast<uint8_t>(ptr[0]);
    hash *= 16777619u;
    hash ^= static_cast<uint8_t>(ptr[1]);
    hash *= 16777619u;
    hash ^= static_cast<uint8_t>(ptr[2]);
    hash *= 16777619u;
    hash ^= static_cast<uint8_t>(ptr[3]);
    hash *= 16777619u;
    ptr += 4;
    len -= 4;
  }

  // Handle remainder
  while (len--) {
    hash ^= static_cast<uint8_t>(*ptr++);
    hash *= 16777619u;
  }

  return hash;
}

/**
 * @brief Fast branch-minimized ASCII to integer conversion.
 *
 * Converts a null-terminated C-string representing a signed decimal integer
 * into an `int`. The implementation is optimized for performance using
 * branchless sign handling and manual loop unrolling to reduce control-flow
 * overhead in the common case of short numeric sequences.
 *
 * Parsing stops at the first non-digit character. Only base-10 digits are
 * supported. No locale handling, whitespace skipping, or overflow checks
 * are performed.
 *
 * @param __s Pointer to a null-terminated character sequence containing an
 *            optional leading '+' or '-' followed by decimal digits.
 *
 * @return int The parsed integer value with sign applied.
 *
 * @pre
 * - `__s` must be a valid, readable, null-terminated string.
 * - Input must represent a value within the range of `int`; otherwise
 *   signed overflow results in undefined behavior.
 *
 * @post
 * - The input pointer itself is not modified externally.
 * - Parsing stops at the first non-decimal character.
 *
 * @note
 * - Time complexity is O(n), where n is the number of processed digits.
 * - Designed for hot paths where input is well-formed and short.
 * - Not a drop-in replacement for `std::atoi` or `std::from_chars`.
 *
 * @warning
 * - No overflow detection.
 * - No validation of empty strings or non-numeric input beyond digit checks.
 * - Behavior is undefined if `__s` is null.
 *
 * @exception noexcept Guaranteed not to throw exceptions.
 */
constexpr int __fast_atoi(const char *__s) noexcept {
  int res = 0;
  int sign = 1;

  // Branchless sign handling
  sign = 1 - 2 * (*__s == '-');
  __s += (*__s == '-' || *__s == '+');

  // Unroll loop for common case (up to 8 digits)
  // Process 4 digits at once using SWAR (SIMD Within A Register)
  while (*__s >= '0' && *__s <= '9') {
    // Manual unrolling for 4 iterations
    res = res * 10 + (*__s++ - '0');
    if (!(*__s >= '0' && *__s <= '9')) {
      break;
    }

    res = res * 10 + (*__s++ - '0');
    if (!(*__s >= '0' && *__s <= '9')) {
      break;
    }

    res = res * 10 + (*__s++ - '0');
    if (!(*__s >= '0' && *__s <= '9')) {
      break;
    }

    res = res * 10 + (*__s++ - '0');
  }

  return res * sign;
}

/**
 * @brief Computes the number of decimal digits required to represent an
 * integer.
 *
 * Returns the base-10 width (digit count) of the given unsigned integer
 * without performing division or logarithmic operations. The implementation
 * uses a sequence of comparisons against powers of 10, which is typically
 * faster than `log10` or iterative division for small and medium-sized
 * integers in hot paths.
 *
 * The function is intended to be aggressively inlined (`__FORCE_INLINE`)
 * and is suitable for performance-critical formatting or serialization
 * routines.
 *
 * @param __n The non-negative integer whose decimal digit width is computed.
 *
 * @return size_t The number of decimal digits required to represent `__n`.
 *                Returns 1 when `__n == 0`.
 *
 * @note
 * - Time complexity is O(1) with a bounded number of comparisons.
 * - Covers the full range of 64-bit unsigned values (up to 20 digits).
 * - Avoids floating-point operations and division instructions.
 *
 * @warning
 * - Assumes `size_t` can represent values up to at least 10^19 if used
 *   on 64-bit platforms; behavior remains correct on smaller platforms
 *   but upper branches may be unreachable.
 *
 * @exception noexcept Guaranteed not to throw exceptions.
 */
__FORCE_INLINE size_t __int_width(size_t __n) noexcept {
  if (__n == 0) {
    return 1;
  }

  // Use binary search on powers of 10
  // It's faster than log10 for small numbers.
  if (__n < 10) {
    return 1;
  }

  if (__n < 100) {
    return 2;
  }

  if (__n < 1000) {
    return 3;
  }

  if (__n < 10000) {
    return 4;
  }

  if (__n < 100000) {
    return 5;
  }

  if (__n < 1000000) {
    return 6;
  }

  if (__n < 10000000) {
    return 7;
  }

  if (__n < 100000000) {
    return 8;
  }

  if (__n < 1000000000) {
    return 9;
  }

  if (__n < 10000000000ULL) {
    return 10;
  }

  if (__n < 100000000000ULL) {
    return 11;
  }

  if (__n < 1000000000000ULL) {
    return 12;
  }

  if (__n < 10000000000000ULL) {
    return 13;
  }

  if (__n < 100000000000000ULL) {
    return 14;
  }

  if (__n < 1000000000000000ULL) {
    return 15;
  }

  if (__n < 10000000000000000ULL) {
    return 16;
  }

  if (__n < 100000000000000000ULL) {
    return 17;
  }

  if (__n < 1000000000000000000ULL) {
    return 18;
  }

  if (__n < 10000000000000000000ULL) {
    return 19;
  }

  return 20;
}

// Every option with the wc is a OptionType::flag,
// the value val is just there for knowledge. The
// is_multi_time_appear_true_v is for std::logic() error, and option
// should be construct once.
enum class OptionType : uint8_t { flag, val, is_multi_time_appear_true_v };

/**
 * @brief Represents a single command-line option for a word-count-like utility.
 *
 * This templated structure defines a command-line option with a short
 * name, long name, type, and default value. It enforces uniqueness and
 * prevents multiple instantiations or unintended copies/moves to reduce
 * logic errors and memory overhead.
 *
 * @tparam __DataType The type of the option's value (e.g., `int`, `bool`).
 *
 * @note
 * - The constructor checks `OptionType::is_multi_time_appear_true_v` and
 *   throws a `std::logic_error` if an option is constructed multiple times.
 * - Copying and moving of this object is explicitly deleted to avoid
 *   accidental duplication or memory issues.
 * - Designed to be `constexpr` where possible, allowing compile-time
 *   initialization of options.
 *
 * @warning
 * - Throws `std::logic_error` if an option marked as unique is constructed
 *   more than once.
 * - Does not perform deep validation of `long_name` or `short_name`.
 */
template <class __DataType> struct __wc_option {
  char short_name;
  std::string_view long_name;
  OptionType __type;
  __DataType __default_value;

  constexpr __wc_option(char __s, std::string_view __l, OptionType __t,
                        __DataType def = __DataType{})
      : short_name(__s), long_name(__l), __type(__t), __default_value(def) {
    // Every option should be unique
    if (OptionType::is_multi_time_appear_true_v == __t)
      throw std::logic_error("An option was construct multiple times!");
  }

  constexpr ~__wc_option() = default;

  // For memory optimization, and to prevent any
  // logic error also.
  __wc_option(__wc_option &) = delete;
  __wc_option(__wc_option &&) = delete;

  __wc_option(const __wc_option &) = delete;
  __wc_option(const __wc_option &&) = delete;
};

#if not defined __wc_argparser_class
#define __wc_argparser_class

// Compact option storage with better cache locality
template <size_t __max_opt = MAX_OPTIONS> class __wc_argparser {
public:
  template <typename _Tp>
  constexpr void add_opt(const __wc_option<_Tp> &__opt) {
    if (UNLIKELY(opt_count >= __max_opt)) [[unlikely]]
      return;

    options[opt_count++] = {__opt.short_name,
                            static_cast<uint8_t>(__opt.long_name.size()),
                            __opt.__type,
                            0, // padding
                            detail::hash_fnv1a(__opt.long_name),
                            __opt.long_name.data()};
  }

  __wc_argparser() = default;
  ~__wc_argparser() = default;

  /**
   * @brief Parses command-line arguments for short and long options.
   *
   * Iterates over `argv`, skipping non-option arguments, and dispatches
   * each option to the appropriate parser (`__parse_short_options` or
   * `__parse_long_options`). Designed for minimal overhead in hot paths.
   *
   * @param argc Number of command-line arguments.
   * @param argv Array of argument strings.
   *
   * @note noexcept guarantees no exceptions are thrown.
   */
  void parse(int argc, const char **argv) noexcept {
    pcount = 0;

    for (int i = 1; i < argc; ++i) {
      const char *__arg = argv[i];

      // Early exit for non-options
      if (__arg[0] != '-') {
        continue;
      }

      // Branchless option type detection
      bool is_long = (__arg[1] == '-');
      const char *opt_start = __arg + 1 + is_long;

      if (*opt_start == '\0') {
        continue;
      }

      if (is_long) {
        __parse_long_options(opt_start, i, argc, argv);
      }

      else {
        __parse_short_options(*opt_start, i, argc, argv);
      }
    }
  }

  // Optimized lookup using SIMD-style comparison where possible
  [[nodiscard]]
  std::optional<const char *> get(char short_name) const noexcept {
    // Linear search is actually faster than hash for small arrays
    // due to better cache locality and branch prediction
    for (size_t i = 0; i < pcount; ++i) {
      if (options[parsed[i].__idx].short_name == short_name) {
        return parsed[i].__v;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]]
  std::optional<const char *> get(std::string_view long_name) const noexcept {
    const uint32_t hash = detail::hash_fnv1a(long_name);

    // Use hash for long option lookup (worth it for string comparison)
    for (size_t i = 0; i < pcount; ++i) {
      const auto &opt = options[parsed[i].__idx];
      if (opt.hash == hash) { // Fast hash comparison first
        return parsed[i].__v;
      }
    }

    return std::nullopt;
  }

  /**
   * @brief Checks if an option with the given short name exists.
   * @param __sn Short option character.
   * @return true if the option is present, false otherwise.
   */
  [[nodiscard]]
  __FORCE_INLINE bool has(char __sn) const noexcept {
    return get(__sn).has_value();
  }

  /**
   * @brief Checks if an option with the given long name exists.
   * @param __ln Long option string.
   * @return true if the option is present, false otherwise.
   */
  [[nodiscard]]
  __FORCE_INLINE bool has(std::string_view __ln) const noexcept {
    return get(__ln).has_value();
  }

  /**
   * @brief Retrieves the integer value of an option by short name.
   * @param __sn Short option character.
   * @param __default_value Value returned if the option is absent (default 0).
   * @return The parsed integer value or `__default_value` if not set.
   */
  [[nodiscard]]
  __FORCE_INLINE int get_int(char __sn,
                             int __default_value = 0) const noexcept {
    auto val = get(__sn);
    return val ? detail::__fast_atoi(*val) : __default_value;
  }

private:
  struct __parsed_option {
    uint8_t __idx;   // The index of the option in argv
    const char *__v; // The raw content of the option value
  };

  /**
   * @brief Memory-aligned representation of a command-line option.
   *
   * Stores both short and long names, the option type, and a precomputed
   * hash for fast lookup. Aligned to 32 bytes for cache efficiency.
   *
   * Members:
   * - `short_name` : Single-character short option.
   * - `long_name_len` : Length of the long option string.
   * - `type` : Option type (e.g., flag, value, etc.).
   * - `_padding` : Padding for alignment.
   * - `hash` : Precomputed hash of the long option name.
   * - `long_name` : Pointer to the long option string.
   */
  struct alignas(32) option_entry {
    char short_name;
    uint8_t long_name_len;
    OptionType type;
    uint8_t _padding;
    uint32_t hash;
    const char *long_name;
  };

  // Use stack arrays for small, fixed-size data
  std::array<__parsed_option, __max_opt> parsed; // Correct parsed option
  uint8_t pcount = 0;                            // parsed option count

  std::array<option_entry, __max_opt> options; // options from argv
  uint8_t opt_count = 0;                       // option count

  /**
   * @brief Parses a single short option and dispatches it.
   *
   * Checks the registered short options for a match with `opt`. If found,
   * calls `__add_parsed_option` to process it. Optimized for the first
   * few options to avoid looping.
   *
   * @param opt The short option character to parse (e.g., 'h').
   * @param idx Reference to the current argument index in `argv`.
   * @param argc Total number of command-line arguments.
   * @param argv Array of argument strings.
   *
   * @note noexcept guarantees that this function does not throw exceptions.
   */
  void __parse_short_options(char opt, int &idx, int argc,
                             const char **argv) noexcept {
    if (opt_count > 0 && options[0].short_name == opt) {
      __add_parsed_option(0, idx, argc, argv);
      return;
    }
    if (opt_count > 1 && options[1].short_name == opt) {
      __add_parsed_option(1, idx, argc, argv);
      return;
    }
    if (opt_count > 2 && options[2].short_name == opt) {
      __add_parsed_option(2, idx, argc, argv);
      return;
    }
    if (opt_count > 3 && options[3].short_name == opt) {
      __add_parsed_option(3, idx, argc, argv);
      return;
    }
    for (size_t i = 4; i < opt_count; ++i) {
      if (options[i].short_name == opt) {
        __add_parsed_option(i, idx, argc, argv);
        return;
      }
    }
  }

  /**
   * @brief Parses a single long option by its name.
   *
   * Computes the FNV-1a hash of `opt` and searches registered options
   * for a matching hash. If found, delegates processing to
   * `__add_parsed_option`.
   *
   * @param opt Pointer to the long option string (without leading "--").
   * @param idx Reference to the current argument index in `argv`.
   * @param argc Total number of command-line arguments.
   * @param argv Array of argument strings.
   *
   * @note noexcept guarantees this function does not throw exceptions.
   */
  void __parse_long_options(const char *opt, int &idx, int argc,
                            const char **argv) noexcept {
    const uint32_t hash = detail::hash_fnv1a(opt);

    for (size_t i = 0; i < opt_count; ++i) {
      if (options[i].hash == hash) {
        __add_parsed_option(i, idx, argc, argv);
        return;
      }
    }
  }

  /**
   * @brief Parses a single long option by its name.
   *
   * Computes the FNV-1a hash of `opt` and searches registered options
   * for a matching hash. If found, delegates processing to
   * `__add_parsed_option`.
   *
   * @param opt Pointer to the long option string (without leading "--").
   * @param idx Reference to the current argument index in `argv`.
   * @param argc Total number of command-line arguments.
   * @param argv Array of argument strings.
   *
   * @note noexcept guarantees this function does not throw exceptions.
   */
  __FORCE_INLINE void __add_parsed_option(size_t i, int &idx, int argc,
                                          const char **argv) noexcept {
    if (pcount >= __max_opt) {
      return;
    }

    const char *value = nullptr;
    if (options[i].type != OptionType::flag && idx + 1 < argc) {
      value = argv[++idx];
    }

    parsed[pcount++] = {static_cast<uint8_t>(i), value};
  }
};

// Builder with move semantics optimization
template <size_t __max_opt = MAX_OPTIONS> class __wc_argparser_builder {
public:
  constexpr __wc_argparser_builder() = default;

  template <typename _Tn>
  constexpr __wc_argparser_builder &flag(char short_name,
                                         std::string_view long_name) {
    parser.add_opt(__wc_option<bool>(short_name, long_name, OptionType::flag));
    return *this;
  }

  template <typename _Tn>
  constexpr __wc_argparser_builder &option(char short_name,
                                           std::string_view long_name) {
    parser.add_opt(__wc_option<_Tn>(short_name, long_name, OptionType::val));
    return *this;
  }

  constexpr __wc_argparser<__max_opt> build() { return std::move(parser); }

private:
  __wc_argparser<__max_opt> parser;
};

#endif // __wc_argparser_class
} // namespace detail

#endif // __wc_argv_parse

#if not defined(__wc_mapped_fs)
#define __wc_mapped_fs

// For file management (eg. reading)
// Map the file content to memory address
// with munmap() and mmap().
namespace fs {

using namespace detail;

// __wc_mapped_file mode.
// The constructor will depend on the
// specified mode.
enum class __wc_mapped_file_mode { BytesOnly, NeedMmap };

struct __wc_mapped_file {
  void *data_{};
  size_t size_{};

#ifdef _WIN32
  HANDLE hFile_ = INVALID_HANDLE_VALUE;
  HANDLE hMapFile_ = INVALID_HANDLE_VALUE;
#else
  int __fd = -1;
#endif

  std::string filename_;

  // After redesigning,
  // Should include the result in the file structure
  // (For better formatting)
  size_t __w_cnt{};
  size_t __l_cnt{};
  size_t __b_cnt{};
  size_t __c_cnt{};

  __wc_mapped_file_mode mode_;
  bool is_stdin_ = false;

  // Constructor for stdin
  explicit __wc_mapped_file()
      : filename_(""), is_stdin_(true), mode_(__wc_mapped_file_mode::NeedMmap) {
    // Read all stdin into memory
    std::vector<char> buffer;
    constexpr size_t chunk_size = 65536; // 64KB chunks
    char temp_buf[chunk_size];

    while (std::cin.read(temp_buf, chunk_size) || std::cin.gcount() > 0) {
      buffer.insert(buffer.end(), temp_buf, temp_buf + std::cin.gcount());
    }

    size_ = buffer.size();
    if (size_ > 0) {
      // Allocate memory and copy data
      data_ = malloc(size_);
      if (data_) {
        std::memcpy(data_, buffer.data(), size_);
      } else {
        size_ = 0;
      }
    }
  }

  explicit __wc_mapped_file(std::string_view __fn, __wc_mapped_file_mode __mode)
      : filename_(__fn), is_stdin_(false) {
    if (LIKELY(__mode == __wc_mapped_file_mode::NeedMmap)) [[likely]] {
      mode_ = __mode; // SET THE MODE!
#ifdef _WIN32
      hFile_ =
          CreateFileA(filename_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

      if (hFile_ == INVALID_HANDLE_VALUE) {
        display_error(TEXT("CreateFileA"));
        return;
      }

      LARGE_INTEGER fileSize;

      if (!GetFileSizeEx(hFile_, &fileSize)) {
        display_error(TEXT("GetFileSizeEx"));
        CloseHandle(hFile_);
        hFile_ = INVALID_HANDLE_VALUE;
        return;
      }

      size_ = static_cast<size_t>(fileSize.QuadPart);

      if (size_ == 0) {
        CloseHandle(hFile_);
        hFile_ = INVALID_HANDLE_VALUE;
        return;
      }

      hMapFile_ =
          CreateFileMappingA(hFile_, nullptr, PAGE_READONLY, 0, 0, nullptr);

      if (hMapFile_ == nullptr) {
        display_error(TEXT("CreateFileMappingA"));
        CloseHandle(hFile_);
        hFile_ = INVALID_HANDLE_VALUE;
        return;
      }

      data_ = MapViewOfFile(hMapFile_, FILE_MAP_READ, 0, 0, 0);

      if (data_ == nullptr) {
        display_error(TEXT("MapViewOfFile"));
        CloseHandle(hMapFile_);
        CloseHandle(hFile_);
        hMapFile_ = INVALID_HANDLE_VALUE;
        hFile_ = INVALID_HANDLE_VALUE;
        return;
      }

#else
      // stat and fstat are linux specific system calls
      struct stat __sb;

      if (UNLIKELY((__fd = open(filename_.c_str(), O_RDONLY)) == -1))
          [[unlikely]] {

        std::cerr << "Cannot open " << filename_ << ":";
        std::perror("open()");
        std::exit(WC_EXIT_CODE_FAILURE);
      }

      if (UNLIKELY(fstat(__fd, &__sb) == -1)) [[unlikely]] {
        close(__fd);
        __fd = -1;
        std::exit(WC_EXIT_CODE_FAILURE);
      }

      if (UNLIKELY((size_ = __sb.st_size) == 0)) [[unlikely]] {
        close(__fd);
        __fd = -1;
        std::exit(WC_EXIT_CODE_FAILURE);
      }

      if (UNLIKELY((data_ = mmap(nullptr, size_, PROT_READ,
                                 MAP_PRIVATE | MAP_POPULATE, __fd, 0)) ==
                   MAP_FAILED)) [[unlikely]] {
        std::cerr << "ERROR: mmap failed! errno=" << errno << " ("
                  << strerror(errno) << ")" << std::endl;
        data_ = nullptr;
        close(__fd);
        __fd = -1;
        return;
      }

      madvise(data_, size_, MADV_SEQUENTIAL | MADV_WILLNEED);
#ifdef MADV_HUGEPAGE
      // Give directions to the kernel about the address
      // range beginning at data (void*)
      madvise(data_, size_, MADV_HUGEPAGE);
#endif // MADV_HUGEPAGE
#endif // _WIN32
    }
    // Do some optimization tricks with -c option
    // Use statx instead of fstat or stat.
    else {
#ifdef _WIN32
      hFile_ =
          CreateFileA(filename_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
      if (hFile_ == INVALID_HANDLE_VALUE) {
        display_error(TEXT("CreateFileA"));
        return;
      }

      LARGE_INTEGER fileSize;
      if (!GetFileSizeEx(hFile_, &fileSize)) {
        display_error(TEXT("GetFileSizeEx"));
        CloseHandle(hFile_);
        hFile_ = INVALID_HANDLE_VALUE;
        return;
      }

      size_ = static_cast<size_t>(fileSize.QuadPart);
      if (size_ == 0) {
        CloseHandle(hFile_);
        hFile_ = INVALID_HANDLE_VALUE;
        return;
      }
#else
      struct statx __sb;
      if (UNLIKELY(statx(AT_FDCWD, filename_.c_str(), 0, STATX_SIZE, &__sb) ==
                   -1)) [[unlikely]] {
        std::cerr << "Statx error" << std::endl;
        return;
      }

      if (UNLIKELY((size_ = __sb.stx_size) == 0)) [[unlikely]] {
        std::cerr << "Size error" << std::endl;
        return;
      }

#endif // _WIN32
      mode_ = __mode;
    }
  }

  // No copy constructor (Object should be unique)
  __wc_mapped_file(const __wc_mapped_file &) = delete;
  __wc_mapped_file &operator=(const __wc_mapped_file &) = delete;

  // Move constructor
  __wc_mapped_file(__wc_mapped_file &&other) noexcept
      : data_(other.data_), size_(other.size_)
#if defined(__linux__)
        ,
        __fd(other.__fd)
#elif defined(_WIN32)
        ,
        hFile_(other.hFile_), hMapFile_(other.hMapFile_)
#endif
        ,
        filename_(std::move(other.filename_)), __w_cnt(other.__w_cnt),
        __l_cnt(other.__l_cnt), __b_cnt(other.__b_cnt), __c_cnt(other.__c_cnt),
        mode_(other.mode_), is_stdin_(other.is_stdin_) {
    other.data_ = nullptr;
    other.size_ = 0;
#if defined(__linux__)
    other.__fd = -1;
#elif defined(_WIN32)
    other.hFile_ = INVALID_HANDLE_VALUE;
    other.hMapFile_ = INVALID_HANDLE_VALUE;
#endif
  }

  /**
   * @brief Move-assigns a mapped file, transferring ownership of resources.
   *
   * Releases any existing resources held by `*this` before taking ownership
   * of `other`'s memory mapping, file handles, counters, and metadata.
   * After the move, `other` is left in a null/empty state.
   *
   * @param other Temporary `__wc_mapped_file` to move from.
   * @return Reference to `*this`.
   *
   * @note noexcept guarantees no exceptions during move.
   */
  __wc_mapped_file &operator=(__wc_mapped_file &&other) noexcept {
    if (this != &other) {
      // Clean up existing resources
      if (valid()) {
        if (is_stdin_ && data_ != nullptr) {
          // Free malloc'd memory for stdin
          free(data_);
        }
#ifdef __linux__
        else if (!is_stdin_ && data_ != nullptr) {
          munmap(data_, size_);
        }

        if (__fd != -1) {
          close(__fd);
        }
#elif defined(_WIN32)
        else if (!is_stdin_ && data_ != nullptr) {
          UnmapViewOfFile(data_);
        }

        if (hMapFile_ != INVALID_HANDLE_VALUE) {
          CloseHandle(hMapFile_);
        }

        if (hFile_ != INVALID_HANDLE_VALUE) {
          CloseHandle(hFile_);
        }
#endif
      }
      // Move from other
      data_ = other.data_;
      size_ = other.size_;
#ifdef __linux__
      __fd = other.__fd;
#elif defined(_WIN32)
      hFile_ = other.hFile_;
      hMapFile_ = other.hMapFile_;
#endif
      filename_ = std::move(other.filename_);
      __w_cnt = other.__w_cnt;
      __l_cnt = other.__l_cnt;
      __b_cnt = other.__b_cnt;
      __c_cnt = other.__c_cnt;
      mode_ = other.mode_;
      is_stdin_ = other.is_stdin_;
      // Nullify other
      other.data_ = nullptr;
      other.size_ = 0;
#ifdef __linux__
      other.__fd = -1;
#elif defined(_WIN32)
      other.hFile_ = INVALID_HANDLE_VALUE;
      other.hMapFile_ = INVALID_HANDLE_VALUE;
#endif
    }

    return *this;
  }

  /** Frees allocated memory for stdin, unmaps mapped files, and closes
   * OS file descriptors/handles depending on platform.
   */
  virtual ~__wc_mapped_file() {
    if (valid()) {
      if (is_stdin_ && data_ != nullptr) {
        // Free malloc'd memory for stdin
        free(data_);
      }
#ifdef _WIN32
      else {
        if (data_ != nullptr) {
          UnmapViewOfFile(data_);
        }

        if (hMapFile_ != INVALID_HANDLE_VALUE) {
          CloseHandle(hMapFile_);
        }

        if (hFile_ != INVALID_HANDLE_VALUE) {
          CloseHandle(hFile_);
        }
      }
#else
      else {
        if (data_ != nullptr) {
          munmap(data_, size_);
        }

        if (__fd != -1) {
          close(__fd);
        }
      }
#endif // _WIN32
    }
  }

  /**
   * @brief Returns the file contents as a read-only span.
   * @return std::span<const char> Span over the mapped data, or empty if
   * BytesOnly mode or data is null.
   */
  [[nodiscard]] __FORCE_INLINE std::span<const char> as_span() const noexcept {
    if (mode_ == __wc_mapped_file_mode::BytesOnly || data_ == nullptr)
      return {};
    return {static_cast<const char *>(data_), size_};
  }

  /**
   * @brief Checks if the mapped file object is valid.
   * @return true if mapped data exists or size is non-zero, false otherwise.
   */
  [[nodiscard]] __FORCE_INLINE bool valid() const noexcept {
    return (mode_ == __wc_mapped_file_mode::NeedMmap) ? data_ != nullptr
                                                      : size_ > 0;
  }

  /**
   * @brief Returns the size of the mapped data.
   * @return size_t Size in bytes.
   */
  [[nodiscard]] __FORCE_INLINE size_t size() const noexcept { return size_; }

  /**
   * @brief Returns the filename associated with the mapped file.
   * @return std::string Copy of the filename.
   */
  [[nodiscard]] __FORCE_INLINE std::string filename() const noexcept {
    return filename_;
  }

  __FORCE_INLINE void setWordCnt(size_t w_cnt) noexcept { __w_cnt = w_cnt; }
  __FORCE_INLINE void setLineCnt(size_t l_cnt) noexcept { __l_cnt = l_cnt; }
  __FORCE_INLINE void setCharCnt(size_t c_cnt) noexcept { __c_cnt = c_cnt; }
  __FORCE_INLINE void setBytesCnt(size_t b_cnt) noexcept { __b_cnt = b_cnt; }

  [[nodiscard]] size_t getWordCnt() const noexcept { return __w_cnt; }
  [[nodiscard]] size_t getLineCnt() const noexcept { return __l_cnt; }
  [[nodiscard]] size_t getCharCnt() const noexcept { return __c_cnt; }
  [[nodiscard]] size_t getBytesCnt() const noexcept { return __b_cnt; }
};
} // namespace fs

#endif // __wc_mapped_fs

#if not defined(__wc_internal_class_already_decl)
#define __wc_internal_class_already_decl

namespace wc_class {
using namespace detail;
using namespace std::literals;

// CPU macro for betteer performance
// Mainly for x86-64
inline constexpr size_t CACHE_LINE_SIZE = 64;
inline constexpr size_t L1_CHUNK = 32 * 1024;
inline constexpr size_t L2_CHUNK = 256 * 1024;
inline constexpr size_t L3_CHUNK = 8 * 1024 * 1024;

// Parallel processing threshold
inline constexpr size_t PARALLEL_THRESHOLD = 512 * 1024;

// The size of a SIMD register
#ifdef __AVX512F__
inline constexpr size_t SIMD_WIDTH{64};

#elif defined(__AVX2__)
inline constexpr size_t SIMD_WIDTH{32};

#elif defined(__SSE2__)
inline constexpr size_t SIMD_WIDTH{16};

#else
inline constexpr size_t SIMD_WIDTH{8};
#endif //__AVX512F__

template <class BitChar, class Translation> class __wc_internal_class {

public:
  /**
   * @brief Returns a singleton instance of `__wc_internal_class`.
   *
   * Provides thread-safe initialization of the singleton using one of
   * several strategies depending on compilation flags:
   * - `std::call_once`
   * - `std::atomic` with double-checked locking
   * - Meyers’ singleton (static local variable)
   *
   * @tparam BitChar Character type used in the internal class.
   * @tparam Translation Translation/mapping type used internally.
   *
   * @return Pointer to the singleton instance.
   *
   * @note The implementation ensures only one instance is created, even
   *       in multi-threaded contexts. The returned pointer must not be deleted.
   */
  static __wc_internal_class<BitChar, Translation> *Instance() {
#ifdef __wc_lib_use_std_call_once
#define __wc_lib_use_res_flag
    std::once_flag flag = wc_flag;
    std::call_once(flag, []() {
#if defined(__wc_lib_has_default_value)
      instance = new __wc_internal_class();
#else
                instance = new __wc_internal_class("", std::identity {});
#endif // __wc_lib_has_default_value
    });

    return instance;
#elif defined(__wc_lib_use_std_atomic)
    // FIX: Use nullptr initialization and remove the problematic optional reset
    static std::atomic<__wc_internal_class<BitChar, Translation> *> instance{
        nullptr};

    // Double-checked locking with proper memory ordering
    // to avoid race condition with the object instance when
    // using multithreaded wc.
    auto *mem = instance.load(std::memory_order_acquire);

    if (mem == nullptr) {
      std::lock_guard<std::shared_mutex> __guard(lock);
      mem = instance.load(std::memory_order_relaxed);
      if (mem == nullptr) {
        mem = new __wc_internal_class<BitChar, Translation>();
        instance.store(mem, std::memory_order_release);
      }
    }

    return mem;

    // The simplest and SAFEST implementation, not used here
#elif defined(___wc_lib_use_meyer)
    static __wc_internal_class<BitChar, Translation> instance;
    return &instance;
#endif // __wc_lib_use_std_call_once
  }

  // Initializing private members argc and argv.
  // Normally, argv is a vector of flags and filename.
  void init(int ___argc, const char **___argv) {
    this->argc = ___argc;

    auto views =
        std::ranges::subrange(___argv, ___argv + ___argc) |
        std::views::transform([](const char *s) { return std::string(s); })

#if __cplusplus >= 202302L
        // std::ranges::to was added in C++23, version below C++23 will make you
        // compilator complain about it. It's better to add a safe alternatives.
        | std::ranges::to<std::vector>();
    this->argv = views;
#else
        // Do not remove this semicolon!
        // It is an important one
        ;
    std::vector<std::string> args(views.begin(), views.end());
    this->argv = args;
#endif // __cplusplus
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((always_inline, hot))
#elif defined(_MSC_VER)
  __forceinline
#endif // __GNUC__

#if defined(_MSVC)
#pragma region __WC_CHAR_IMPL
#endif // _MSVC

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((always_inline, hot))
#elif defined(_MSVC)
      __force_inline
#endif // ___GNUC__

  /**
   * @brief Returns the size in bytes of a mapped file.
   *
   * Uses the pre-mapped file size directly, making it extremely fast.
   * Suitable for counting bytes (e.g., `-c` option). Returns 0 if the
   * file is invalid or the mapped file array is empty.
   *
   * @param translation Optional translation/mapping functor (default:
   * identity).
   * @param f_idx Index of the mapped file to query (default: 0).
   * @return size_t File size in bytes, or 0 if invalid.
   */
  size_t __wc_char_c(Translation translation = std::identity{},
                     size_t f_idx = 0) {
    if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
      return 0;
    }
    return mapped_file[f_idx].size();
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((always_inline, hot))
#elif defined(_MSVC)
  __force_inline
#endif // ___GNUC__

#if defined(_MSVC)
#pragma endregion __WC_CHAR_IMPL
#endif // _MSVC

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((always_inline, hot))
#elif defined(_MSVC)
      __force_inline
#endif // ___GNUC__

#if defined(_MSVC)
#pragma region __WC_LINE_IMPL
#endif // _MSVC

#ifdef __AVX512F__
  [[gnu::target("avx512f")]]
  // Using AVX512
  // Slightly better (Use inline assembly)

  /**
   * @brief Counts lines in a mapped file using AVX512 SIMD instructions.
   *
   * Processes 64 bytes at a time, comparing each byte to `'\n'` using
   * `_mm512_cmpeq_epi8_mask` and counting set bits. Falls back to scalar
   * counting for the remaining bytes. Extremely fast for large files.
   *
   * @param translation Optional translation/mapping functor (default:
   * identity).
   * @param f_idx Index of the mapped file to process (default: 0).
   * @return size_t Total number of newline characters in the file.
   *
   * @note Requires CPU with AVX512F support. Returns 0 if the mapped file
   *       is invalid or empty.
   */
  __FORCE_INLINE size_t __wc_line(Translation translation = std::identity{},
                                  size_t f_idx = 0) noexcept {
    if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
      return 0;
    }
    size_t __l_count{};
    auto __data = mapped_file[f_idx].as_span();
    const char *ptr = __data.data();
    const char *end = ptr + __data.size();
    const __m512i newline = _mm512_set1_epi8('\n');
    // Process 64 bytes at a time
    // If a newline is found on the n-th bits,
    // it is switched to one, else every non newline bits is
    // set to 0. That's what the function _mm512_cmpeq_epi8_mask() does.

    while (LIKELY(ptr + 64 <= end)) {
      __builtin_prefetch(ptr + 128, 0, 0);
      __m512i __chk =
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptr));
      __mmask64 __m = _mm512_cmpeq_epi8_mask(__chk, newline);

      // Count the number of set bits (population count)
      // for data type unsigned long long
      __l_count += __builtin_popcountll(__m);
      ptr += 64;
    }

    // Handle remainder (Because
    // every input is not forcefully with 0 mod 64.
    while (LIKELY(ptr < end)) {
      __l_count += (*ptr++ == '\n');
    }
    return __l_count;
  }

#elif defined(__AVX2__)
      /**
       * @brief Counts lines in a mapped file using AVX2 SIMD instructions.
       *
       * Processes 32 bytes at a time, comparing each byte to `'\n'` using
       * `_mm256_cmpeq_epi8` and `_mm256_movemask_epi8`, then counts set bits
       * with `std::popcount`. Remaining bytes are counted with a scalar loop.
       *
       * @param translation Optional translation/mapping functor (default:
       * identity).
       * @param f_idx Index of the mapped file to process (default: 0).
       * @return size_t Total number of newline characters in the file, 0 if
       *         the mapped file is empty or invalid.
       *
       * @note Requires CPU with AVX2 support.
       */
      [[gnu::target("avx2")]]
      __FORCE_INLINE size_t __wc_line(Translation translation = std::identity{},
                                      size_t f_idx = 0) noexcept {
    if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
      std::cerr << "Mapped file is empty" << std::endl;
      return 0;
    }
    size_t __l_count{};
    auto __data = mapped_file[f_idx].as_span();
    const char *ptr = __data.data();
    const char *end = ptr + __data.size();
    const __m256i newline = _mm256_set1_epi8('\n');
    size_t simd_iterations = 0;
    size_t simd_newlines = 0;

    // Same logic as with __AVX512F__
    // Just different instruction set
    // More high level and understandable
    while (LIKELY(ptr + 32 <= end)) {
      // __builtin_prefetch ( ptr + 128, 0, 3 );  // TEMPORARILY DISABLED
      __m256i __chk =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
      __m256i __cmp = _mm256_cmpeq_epi8(__chk, newline);
      int __m = _mm256_movemask_epi8(__cmp);
      int count = std::popcount(static_cast<uint32_t>(__m));

      __l_count += count;
      simd_newlines += count;
      simd_iterations++;
      ptr += 32;
    }

    // Handle remainder
    size_t rem = 0;
    while (LIKELY(ptr < end)) {
      rem += (*ptr++ == '\n');
    }

    __l_count += rem;
    return __l_count;
  }

#elif defined(__SSE2__)
  //  SSE is the newest instruction set for
  //  x86-64 CPU.

  /**
   * @brief Counts lines in a mapped file using SSE2 SIMD instructions.
   *
   * Processes 16 bytes at a time, comparing each byte to `'\n'` using
   * `_mm_cmpeq_epi8` and `_mm_movemask_epi8`, then counts set bits with
   * `std::popcount`. Remaining bytes are counted with a scalar loop.
   *
   * @param translation Optional translation/mapping functor (default:
   * identity).
   * @param f_idx Index of the mapped file to process (default: 0).
   * @return size_t Total number of newline characters in the file, or 0 if
   *         the mapped file is empty or invalid.
   *
   * @note Requires CPU with SSE2 support.
   */
  [[gnu::target("sse2")]]
  __FORCE_INLINE size_t __wc_line(Translation translation = std::identity{},
                                  size_t f_idx = 0) noexcept {
    if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
      return 0;
    }

    size_t __l_count{};
    auto __data = mapped_file[f_idx].as_span();
    const char *ptr = __data.data();
    const char *end = ptr + __data.size();
    const __m128i newline = _mm_set1_epi8('\n');

    while (LIKELY(ptr + 16 <= end)) {
      __builtin_prefetch(ptr + 64, 0, 3);
      __m128i __chk = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr));
      __m128i __cmp = _mm_cmpeq_epi8(__chk, newline);
      int mask = _mm_movemask_epi8(__cmp);
      __l_count += std::popcount(static_cast<uint32_t>(mask));
      ptr += 16;
    }

    while (LIKELY(ptr < end)) {
      __l_count += (*ptr++ == '\n');
    }

    return __l_count;
  }

#else // Fallback to a scalar implementation

  /**
   * @brief Counts lines in a mapped file using loop unrolling.
   *
   * Processes 32 bytes at a time with manual unrolling to count newline
   * characters (`'\n'`) efficiently, then handles any remaining bytes
   * with a scalar loop. This is a fast, portable alternative to SIMD.
   *
   * @param translation Optional translation/mapping functor (default:
   * identity).
   * @param f_idx Index of the mapped file to process (default: 0).
   * @return size_t Total number of newline characters in the file.
   *
   * @note Optimized for large files; works on all architectures.
   */
  __FORCE_INLINE size_t __wc_line(Translation translation = std::identity{},
                                  size_t f_idx = 0) noexcept {
    size_t __l_count{};
    auto __data = mapped_file[f_idx].as_span();
    const char *ptr = __data.data();
    const char *end = ptr + __data.size();

    // Process 32 bytes at a time (Ah the mf)
    while (LIKELY(ptr + 32 <= end)) {
      __l_count += (ptr[0] == '\n') + (ptr[1] == '\n') + (ptr[2] == '\n') +
                   (ptr[3] == '\n') + (ptr[4] == '\n') + (ptr[5] == '\n') +
                   (ptr[6] == '\n') + (ptr[7] == '\n') + (ptr[8] == '\n') +
                   (ptr[9] == '\n') + (ptr[10] == '\n') + (ptr[11] == '\n') +
                   (ptr[12] == '\n') + (ptr[13] == '\n') + (ptr[14] == '\n') +
                   (ptr[15] == '\n') + (ptr[16] == '\n') + (ptr[17] == '\n') +
                   (ptr[18] == '\n') + (ptr[19] == '\n') + (ptr[20] == '\n') +
                   (ptr[21] == '\n') + (ptr[22] == '\n') + (ptr[23] == '\n') +
                   (ptr[24] == '\n') + (ptr[25] == '\n') + (ptr[26] == '\n') +
                   (ptr[27] == '\n') + (ptr[28] == '\n') + (ptr[29] == '\n') +
                   (ptr[30] == '\n') + (ptr[31] == '\n');
      ptr += 32;
    }

    // Handle remainder
    while (LIKELY(ptr < end)) {
      __l_count += (*ptr++ == '\n');
    }

    return __l_count;
  }

#endif // __AVX512F__

#if defined(_MSVC)
#pragma endregion __WC_LINE_IMPL
#endif // _MSVC

#if defined(_MSVC)
#pragma region __WC_WORD_IMPL
#endif // _MSVC

  // Dummy function, will break on very large file
  // theorically.

  /**
   * @brief Counts words in a mapped file using a linear scan.
   *
   * Iterates over the file data as a string view, detecting words separated
   * by whitespace characters (`' '`, `'\r'`, `'\n'`, `'\t'`).
   * Simple, portable, but not SIMD-optimized.
   *
   * @param translation Optional translation/mapping functor (default:
   * identity).
   * @param f_idx Index of the mapped file to process (default: 0).
   * @return size_t Total number of words in the file.
   *
   * @note Linear implementation; may be slower for very large files.
   */
  size_t __wc_word(Translation translation = std::identity{},
                   size_t f_idx = 0) {
    size_t __w_count{}, pos = 0;
    auto __data = mapped_file[f_idx].as_span();
    auto __str = std::string_view(__data.data());

    while (LIKELY(true)) [[likely]] {
      if ((pos = __str.find_first_not_of(" \r\n\t", pos + 1)) == __str.npos) {
        break;
      }

      __w_count++;

      if ((pos = __str.find_first_of(" \r\n\t", pos + 1)) == __str.npos) {
        break;
      }
    }
    return __w_count;
  }

#if defined(_MSVC)
#pragma region __WC_CHAR_M_IMPL
#endif

#ifdef __AVX512F__
  /**
   * @brief Counts UTF-8 characters in a mapped file using AVX512F SIMD.
   *
   * Processes 64 bytes at a time, detecting UTF-8 character starts by
   * identifying non-continuation bytes (bytes not matching 10xxxxxx).
   * Remaining bytes are counted with a scalar loop.
   *
   * @param translation Optional translation/mapping functor (default:
   * identity).
   * @param f_idx Index of the mapped file to process (default: 0).
   * @return size_t Total number of UTF-8 characters in the file, 0 if
   *         the mapped file is empty or invalid.
   *
   * @note Requires CPU with AVX512F support.
   */
  [[gnu::target("avx512f")]]
  __FORCE_INLINE size_t __wc_char_m(Translation translation = std::identity{},
                                    size_t f_idx = 0) noexcept {
    if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
      return 0;
    }

    auto __data = mapped_file[f_idx].as_span();
    const char *ptr = __data.data();
    const char *end = ptr + __data.size();
    size_t char_count = 0;

    // UTF-8 continuation bytes: 10xxxxxx (top 2 bits = 10)
    const __m512i continuation_mask = _mm512_set1_epi8(0xC0);    // 11000000
    const __m512i continuation_pattern = _mm512_set1_epi8(0x80); // 10000000

    while (LIKELY(ptr + 64 <= end)) {
      __builtin_prefetch(ptr + 128, 0, 0);
      __m512i chunk =
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptr));

      // Mask to get top 2 bits of each byte
      __m512i masked = _mm512_and_si512(chunk, continuation_mask);

      // Compare with continuation pattern (10xxxxxx)
      __mmask64 is_continuation =
          _mm512_cmpeq_epi8_mask(masked, continuation_pattern);

      // Count non-continuation bytes (these are character starts)
      char_count += 64 - __builtin_popcountll(is_continuation);
      ptr += 64;
    }

    // Handle remainder
    while (LIKELY(ptr < end)) {
      char_count += ((*ptr & 0xC0) != 0x80);
      ptr++;
    }

    return char_count;
  }

#elif defined(__AVX2__)
  /**
   * @brief Counts UTF-8 characters in a mapped file using AVX2 SIMD.
   *
   * Processes 32 bytes at a time, detecting UTF-8 character starts by
   * identifying non-continuation bytes (bytes not in 0x80–0xBF range).
   * Remaining bytes are counted using a scalar loop.
   *
   * @param translation Optional translation/mapping functor (default:
   * identity).
   * @param f_idx Index of the mapped file to process (default: 0).
   * @return size_t Total number of UTF-8 characters, 0 if file is
   * empty/invalid.
   *
   * @note Requires CPU with AVX2 support.
   */
  [[gnu::target("avx2")]]
  __FORCE_INLINE size_t __wc_char_m(Translation translation = std::identity{},
                                    size_t f_idx = 0) noexcept {
    if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
      return 0;
    }

    auto __data = mapped_file[f_idx].as_span();
    const char *ptr = __data.data();
    const char *end = ptr + __data.size();
    size_t char_count = 0;

    // UTF-8 continuation bytes start with 10xxxxxx (0x80-0xBF)
    // We count all bytes that are NOT continuation bytes
    const __m256i continuation_mask = _mm256_set1_epi8(0xC0);    // 11000000
    const __m256i continuation_pattern = _mm256_set1_epi8(0x80); // 10000000

    while (LIKELY(ptr + 32 <= end)) {
      __m256i chunk =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));

      // Get top 2 bits: (byte & 11000000)
      __m256i masked = _mm256_and_si256(chunk, continuation_mask);

      // Compare with 10000000 (continuation bytes)
      __m256i is_continuation = _mm256_cmpeq_epi8(masked, continuation_pattern);

      // Get mask of continuation bytes
      int continuation_bits = _mm256_movemask_epi8(is_continuation);

      // Count non-continuation bytes (characters)
      // 32 total bytes - number of continuation bytes
      char_count +=
          32 - __builtin_popcount(static_cast<uint32_t>(continuation_bits));
      ptr += 32;
    }
    // Handle remainder
    while (LIKELY(ptr < end)) {
      // A byte is a character start if top 2 bits are NOT 10
      char_count += ((*ptr & 0xC0) != 0x80);
      ptr++;
    }

    return char_count;
  }

#elif defined(__SSE2__)
  /**
   * @brief Counts UTF-8 characters in a mapped file using SSE2 SIMD.
   *
   * Processes 16 bytes at a time, detecting UTF-8 character starts by
   * identifying non-continuation bytes (bytes not in 0x80–0xBF range).
   * Remaining bytes are counted using a scalar loop.
   *
   * @param translation Optional translation/mapping functor (default:
   * identity).
   * @param f_idx Index of the mapped file to process (default: 0).
   * @return size_t Total number of UTF-8 characters, 0 if file is empty or
   * invalid.
   *
   * @note Requires CPU with SSE2 support.
   */
  [[gnu::target("sse2")]]
  __FORCE_INLINE size_t __wc_char_m(Translation translation = std::identity{},
                                    size_t f_idx = 0) noexcept {
    if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
      return 0;
    }

    auto __data = mapped_file[f_idx].as_span();
    const char *ptr = __data.data();
    const char *end = ptr + __data.size();
    size_t char_count = 0;
    const __m128i continuation_mask = _mm_set1_epi8(0xC0);
    const __m128i continuation_pattern = _mm_set1_epi8(0x80);

    while (LIKELY(ptr + 16 <= end)) {
      __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr));
      __m128i masked = _mm_and_si128(chunk, continuation_mask);
      __m128i is_continuation = _mm_cmpeq_epi8(masked, continuation_pattern);
      int continuation_bits = _mm_movemask_epi8(is_continuation);
      char_count +=
          16 - __builtin_popcount(static_cast<uint32_t>(continuation_bits));
      ptr += 16;
    }

    while (LIKELY(ptr < end)) {
      char_count += ((*ptr & 0xC0) != 0x80);
      ptr++;
    }

    return char_count;
  }

#else // Scalar fallback
  /**
   * @brief Counts UTF-8 characters in a mapped file using unrolled loops.
   *
   * Processes 32 bytes at a time, treating any byte that is NOT a UTF-8
   * continuation byte (0x80–0xBF) as a character start.
   * Remainder bytes are handled with a scalar loop.
   *
   * @param translation Optional translation/mapping functor (default:
   * identity).
   * @param f_idx Index of the mapped file to process (default: 0).
   * @return size_t Total number of UTF-8 characters, 0 if file is
   * empty/invalid.
   *
   * @note This is a high-performance branch-unrolled implementation, portable
   *       across CPUs without SIMD support.
   */
  __FORCE_INLINE
  size_t __wc_char_m(Translation translation = std::identity{},
                     size_t f_idx = 0) noexcept {
    if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
      return 0;
    }

    auto __data = mapped_file[f_idx].as_span();
    const char *ptr = __data.data();
    const char *end = ptr + __data.size();
    size_t char_count = 0;

    // Process 32 bytes at a time
    while (LIKELY(ptr + 32 <= end)) {
      char_count += ((ptr[0] & 0xC0) != 0x80) + ((ptr[1] & 0xC0) != 0x80) +
                    ((ptr[2] & 0xC0) != 0x80) + ((ptr[3] & 0xC0) != 0x80) +
                    ((ptr[4] & 0xC0) != 0x80) + ((ptr[5] & 0xC0) != 0x80) +
                    ((ptr[6] & 0xC0) != 0x80) + ((ptr[7] & 0xC0) != 0x80) +
                    ((ptr[8] & 0xC0) != 0x80) + ((ptr[9] & 0xC0) != 0x80) +
                    ((ptr[10] & 0xC0) != 0x80) + ((ptr[11] & 0xC0) != 0x80) +
                    ((ptr[12] & 0xC0) != 0x80) + ((ptr[13] & 0xC0) != 0x80) +
                    ((ptr[14] & 0xC0) != 0x80) + ((ptr[15] & 0xC0) != 0x80) +
                    ((ptr[16] & 0xC0) != 0x80) + ((ptr[17] & 0xC0) != 0x80) +
                    ((ptr[18] & 0xC0) != 0x80) + ((ptr[19] & 0xC0) != 0x80) +
                    ((ptr[20] & 0xC0) != 0x80) + ((ptr[21] & 0xC0) != 0x80) +
                    ((ptr[22] & 0xC0) != 0x80) + ((ptr[23] & 0xC0) != 0x80) +
                    ((ptr[24] & 0xC0) != 0x80) + ((ptr[25] & 0xC0) != 0x80) +
                    ((ptr[26] & 0xC0) != 0x80) + ((ptr[27] & 0xC0) != 0x80) +
                    ((ptr[28] & 0xC0) != 0x80) + ((ptr[29] & 0xC0) != 0x80) +
                    ((ptr[30] & 0xC0) != 0x80) + ((ptr[31] & 0xC0) != 0x80);
      ptr += 32;
    }

    while (LIKELY(ptr < end)) {
      char_count += ((*ptr++ & 0xC0) != 0x80);
    }

    return char_count;
  }

#endif // __AVX512F__

#if defined(_MSVC)
#pragma endregion __WC_CHAR_M_IMPL
#endif

  /**
   * @brief Returns the total number of words counted.
   * @return size_t Total words.
   */
  [[nodiscard]] __FORCE_INLINE size_t getTotalWord() const noexcept {
    return total_word;
  }

  /**
   * @brief Returns the total number of lines counted.
   * @return size_t Total lines.
   */
  [[nodiscard]] __FORCE_INLINE size_t getTotalLine() const noexcept {
    return total_line;
  }

  /**
   * @brief Returns the total number of characters counted.
   * @return size_t Total characters.
   */
  [[nodiscard]] __FORCE_INLINE size_t getTotalChar() const noexcept {
    return total_char;
  }

  /**
   * @brief Returns the total number of bytes counted.
   * @return size_t Total bytes.
   */
  [[nodiscard]] __FORCE_INLINE size_t getTotalBytes() const noexcept {
    return total_bytes;
  }

  /**
   * @brief Prints the counts (lines, words, characters, bytes) for each mapped
   * file and the total.
   *
   * The output respects column widths for alignment. Only enabled counters
   * (count_line, count_word, count_char, count_bytes) are printed.
   */
  __FORCE_INLINE void printTotal() const noexcept {
    for (const auto &file : mapped_file) {
      if (count_line) {
        std::cout << std::setw(__max_line_width) << file.getLineCnt() << ' ';
      }
      if (count_word) {
        std::cout << std::setw(__max_word_width) << file.getWordCnt() << ' ';
      }
      if (count_char) {
        std::cout << std::setw(__max_char_width) << file.getCharCnt() << ' ';
      }
      if (count_bytes) {
        std::cout << std::setw(__max_bytes_width) << file.getBytesCnt() << ' ';
      }
      std::cout << file.filename() << std::endl;
    }
    if (count_line) {
      std::cout << std::setw(__max_line_width) << total_line << ' ';
    }
    if (count_word) {
      std::cout << std::setw(__max_word_width) << total_word << ' ';
    }
    if (count_char) {
      std::cout << std::setw(__max_char_width) << total_char << ' ';
    }
    if (count_bytes) {
      std::cout << std::setw(__max_bytes_width) << total_bytes << ' ';
    }
    std::cout << "total" << std::endl;
  }

#if defined(_MSVC)
#pragma endregion __WC_WORD_IMPL
#endif // _MSVC

  /**
   * @brief Computes line, word, character, and byte counts for all mapped
   * files.
   *
   * Iterates over each mapped file and updates per-file counts as well as
   * totals. Column widths (__max_*_width) are updated for aligned printing.
   *
   * @param __local_transform Optional character translation function (default:
   * identity).
   */
  void wc(Translation __local_transform = std::identity{}) {
    size_t var{};
    __parse_argv();

    for (int i = 0; i < mapped_file.size(); ++i) {
      if (count_line) {
        var = __wc_line(__local_transform, i);
        mapped_file[i].setLineCnt(var);
        __max_line_width = std::max(__max_line_width, detail::__int_width(var));
        total_line += var;
      }
      if (count_word) {
        var = __wc_word(__local_transform, i);
        mapped_file[i].setWordCnt(var);
        __max_word_width = std::max(__max_word_width, detail::__int_width(var));
        total_word += var;
      }
      if (count_bytes) {
        var = __wc_char_c(__local_transform, i);
        mapped_file[i].setBytesCnt(var);
        __max_bytes_width =
            std::max(__max_bytes_width, detail::__int_width(var));
        total_bytes += var;
      }
      if (count_char) {
        var = __wc_char_m(__local_transform, i);
        mapped_file[i].setCharCnt(var);
        __max_char_width = std::max(__max_char_width, detail::__int_width(var));
        total_char += var;
      }
    }
  }

  /**
   * @brief Performs a hybrid parallel word/line/byte/character count across
   * multiple files.
   *
   * This function adapts its strategy depending on the number of files and
   * available threads:
   *
   * - For a single file, it performs sequential counting (no parallelization).
   * - For multiple files, it uses a thread pool to distribute work across
   * threads, processing files in chunks to reduce contention.
   *
   * Each thread maintains a local accumulator (`ThreadLocalAccumulator`) for
   * per-thread totals and maximum column widths. After all threads complete,
   * the local accumulators are reduced to compute global totals and widths.
   *
   * The function respects the counting flags (`count_line`, `count_word`,
   * `count_bytes`, `count_char`) and updates per-file counts in `mapped_file`.
   *
   * @param __local_transform Optional character translation function applied
   * during counting (default: identity).
   */
  void wc_parallel_hybrid(Translation __local_transform = std::identity{}) {
    // process this->argv and activate each
    // corresponding flags.
    __parse_argv();

    auto *pool = tp::__wc_thread_pool::Instance();
    const size_t num_files = mapped_file.size();
    const size_t num_threads = pool->thread_count();

    // Handle single file case, no parallelization needed
    // (Can be a performance skyrocket)
    if (num_files == 1) {
      size_t var{};
      if (count_line) {
        var = __wc_line(__local_transform, 0);
        mapped_file[0].setLineCnt(var);
        total_line = var;
        __max_line_width = detail::__int_width(var);
      }
      if (count_word) {
        var = __wc_word(__local_transform, 0);
        mapped_file[0].setWordCnt(var);
        total_word = var;
        __max_word_width = detail::__int_width(var);
      }
      if (count_bytes) {
        var = __wc_char_c(__local_transform, 0);
        mapped_file[0].setBytesCnt(var);
        total_bytes = var;
        __max_bytes_width = detail::__int_width(var);
      }
      if (count_char) {
        var = __wc_char_m(__local_transform, 0);
        mapped_file[0].setCharCnt(var);
        total_char = var;
        __max_char_width = detail::__int_width(var);
      }

      return;
    }

    // Per-thread accumulators to minimize atomic contention.
    // Without thread_local, this implementation can be a candidate
    // for possible data race but it was not proved yet.
    // Multiple workers may write to same accumulator
    // which implies data race / false sharing.
    //
    // Even though total_XXXX are already declared as private members of
    // __wc_internal_class, this struct keeps track of result from workers.
    // At the end, its content will be affected to the corresponding value.
    struct alignas(64) ThreadLocalAccumulator {
      size_t total_line = 0;  // total number of line
      size_t total_word = 0;  // total number of word
      size_t total_bytes = 0; // total number of bytes
      size_t total_char = 0;  // total number of char

      // To have better formatting, we have to
      // keep track of the max width for each result.
      // then pass that with setw in std::cout streams.
      size_t max_line_width = 0;
      size_t max_word_width = 0;
      size_t max_bytes_width = 0;
      size_t max_char_width = 0;
    };

    std::vector<ThreadLocalAccumulator> accumulators(num_threads);
    std::vector<std::future<void>> futures;
    futures.reserve(num_files);

    // Determine optimal chunk size for work distribution
    const size_t min_chunk_size = 1;
    const size_t max_chunk_size = 10;
    const size_t chunk_size =
        std::clamp(num_files / num_threads, min_chunk_size, max_chunk_size);

    // Process files in chunks of chunk_size
    for (size_t chunk_start = 0; chunk_start < num_files;
         chunk_start += chunk_size) {
      size_t chunk_end = std::min(chunk_start + chunk_size, num_files);
      size_t thread_idx = (chunk_start / chunk_size) % num_threads;
      futures.push_back(pool->submit(
          [&, chunk_start, chunk_end, thread_idx, __local_transform]() {
            auto &acc = accumulators[thread_idx];
            for (size_t i = chunk_start; i < chunk_end; ++i) {
              size_t var{};
              if (count_line) {
                var = __wc_line(__local_transform, i);
                mapped_file[i].setLineCnt(var);
                acc.total_line += var;
                acc.max_line_width =
                    std::max(acc.max_line_width, detail::__int_width(var));
              }
              if (count_word) {
                var = __wc_word(__local_transform, i);
                mapped_file[i].setWordCnt(var);
                acc.total_word += var;
                acc.max_word_width =
                    std::max(acc.max_word_width, detail::__int_width(var));
              }
              if (count_bytes) {
                var = __wc_char_c(__local_transform, i);
                mapped_file[i].setBytesCnt(var);
                acc.total_bytes += var;
                acc.max_bytes_width =
                    std::max(acc.max_bytes_width, detail::__int_width(var));
              }
              if (count_char) {
                var = __wc_char_m(__local_transform, i);
                mapped_file[i].setCharCnt(var);
                acc.total_char += var;
                acc.max_char_width =
                    std::max(acc.max_char_width, detail::__int_width(var));
              }
            }
          }));
    }
    // Wait for all tasks
    for (auto &future : futures) {
      future.get();
    }

    // When all tasks is finished, which means
    // accumulator has already the final value, we reduce
    // the runtime to a single-threaded one.
    total_line = 0;
    total_word = 0;
    total_bytes = 0;
    total_char = 0;
    __max_line_width = 0;
    __max_word_width = 0;
    __max_bytes_width = 0;

    for (const auto &acc : accumulators) {
      total_line += acc.total_line;
      total_word += acc.total_word;
      total_bytes += acc.total_bytes;
      total_char += acc.total_char;

      __max_line_width = std::max(__max_line_width, acc.max_line_width);
      __max_word_width = std::max(__max_word_width, acc.max_word_width);
      __max_bytes_width = std::max(__max_bytes_width, acc.max_bytes_width);
      __max_char_width = std::max(__max_char_width, acc.max_char_width);
    }
  }

private:
  static std::shared_mutex lock;
  static std::once_flag wc_flag;

  int argc;
  std::vector<std::string> argv;

  __wc_internal_class() {}

  __wc_internal_class(BitChar __bc, Translation translation) {}

  bool count_line;
  bool count_bytes;
  bool count_char;
  bool count_word;

  size_t total_line{};
  size_t total_bytes{};
  size_t total_char{};
  size_t total_word{};

  size_t __max_line_width{};
  size_t __max_word_width{};
  size_t __max_char_width{};
  size_t __max_bytes_width{};

  std::vector<fs::__wc_mapped_file> mapped_file;

  // argv is already a member of the class
  inline void __parse_argv() {
    // Check for --version and --help first
    for (const auto &arg : argv) {
      if (arg == "--version") {
        std::cout << "fast-wc (GNU coreutils) 1.0.0\n"
                  << "Copyright (C) 2026 Free Software Foundation, Inc.\n"
                  << "License GPLv3+: GNU GPL version 3 or later "
                     "<https://gnu.org/licenses/gpl.html>.\n"
                  << "This is free software: you are free to change and "
                     "redistribute it.\n"
                  << "There is NO WARRANTY, to the extent permitted by law.\n"
                  << "Written by Sergio Randriamihoatra.\n";

        std::exit(WC_EXIT_CODE_SUCCESS);
      }

      if (arg == "--help") {
        std::cout
            << "Usage: " << argv[0] << " [OPTION]... [FILE]...\n"
            << "  or:  " << argv[0] << " [OPTION]... --files0-from=F\n"
            << "Print newline, word, and byte counts for each FILE, and a "
               "total line if\n"
            << "more than one FILE is specified.  A word is a nonempty "
               "sequence of non white\n"
            << "space delimited by white space characters or by start or end "
               "of input.\n"
            << "With no FILE, or when FILE is -, read standard input.\n"
            << "\n"
            << "The options below may be used to select which counts are "
               "printed, always in\n"
            << "the following order: newline, word, character, byte, maximum "
               "line length.\n"
            << "  -c, --bytes            print the byte counts\n"
            << "  -m, --chars            print the character counts\n"
            << "  -l, --lines            print the newline counts\n"
            << "      --files0-from=F    read input from the files specified "
               "by\n"
            << "                           NUL-terminated names in file F;\n"
            << "                           If F is - then read names from "
               "standard input\n"
            << "  -L, --max-line-length  print the maximum display width\n"
            << "  -w, --words            print the word counts\n"
            << "      --total=WHEN       when to print a line with total "
               "counts;\n"
            << "                           WHEN can be: auto, always, only, "
               "never\n"
            << "      --help        display this help and exit\n"
            << "      --version     output version information and exit\n"
            << "\n"
            << "Report bugs to: sergiorandriamihoatra@gmail.com\n";

        std::exit(WC_EXIT_CODE_SUCCESS);
      }
    }

    auto pb = __wc_argparser_builder<>()
                  .flag<bool>('l', "lines"sv)
                  .flag<bool>('c', "bytes"sv)
                  .flag<bool>('m', "chars"sv)
                  .flag<bool>('w', "words"sv);
    auto parser = pb.build();
    // Need to convert back to const char **
    std::vector<const char *> ___argv;

    for (const auto &arg : argv) {
      ___argv.push_back(arg.c_str());
    }
    parser.parse(this->argc, ___argv.data());
    // Set flags from the parser object
    count_line = parser.has('l');
    count_bytes = parser.has('c');
    count_char = parser.has('m');
    count_word = parser.has('w');

    fs::__wc_mapped_file_mode __m = fs::__wc_mapped_file_mode::NeedMmap;
    if (UNLIKELY(count_bytes && !count_word && !count_char && !count_line))
        [[unlikely]] {
      __m = fs::__wc_mapped_file_mode::BytesOnly;
    }

    if (UNLIKELY(!count_word && !count_char && !count_line && !count_bytes))
        [[unlikely]] {
      count_bytes = count_line = count_word = true;
    }

    mapped_file.clear();
    bool has_files = false;
    for (const std::string &s : argv) {
      if (s != argv[0]) {
        // stdin
        if (s == "-") {
          mapped_file.push_back(fs::__wc_mapped_file());
          has_files = true;
        }
        // Skip flags
        else if (s[0] != '-') {
          mapped_file.push_back(fs::__wc_mapped_file(s, __m));
          has_files = true;
        }
      }
    }
    // If no files provided, read from stdin
    if (!has_files) {
      mapped_file.push_back(fs::__wc_mapped_file());
    }
  }
};

// Initializing and declaring static member
#ifndef _WIN32
template <class BitChar, class Translation>
std::shared_mutex __wc_internal_class<BitChar, Translation>::lock;

template <class BitChar, class Translation>
std::once_flag __wc_internal_class<BitChar, Translation>::wc_flag;
#endif // _WIN32
} // namespace wc_class

#ifdef _WIN32
template <class BitChar, class Translation>
std::shared_mutex wc_class::__wc_internal_class<BitChar, Translation>::lock;

template <class BitChar, class Translation>
std::once_flag wc_class::__wc_internal_class<BitChar, Translation>::wc_flag;
#endif // _WIN32

#endif // __wc_internal_class

template <class BitChar, class Translation>
inline constexpr wc_class::__wc_internal_class<BitChar, Translation>
    wc_class_instance =
        wc_class::__wc_internal_class<BitChar, Translation>::Instance();

int main(int argc, const char **argv) {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  auto __wcObject =
      wc_class::__wc_internal_class<std::char_traits<char>,
                                    std::function<std::char_traits<char>(
                                        std::char_traits<char>)>>::Instance();

  __wcObject->init(argc, argv);

#ifdef BENCHMARK
  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
#endif // BENCHMARK

  __wcObject->wc_parallel_hybrid();
  __wcObject->printTotal();

#ifdef BENCHMARK
  auto end = std::chrono::high_resolution_clock::now();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  std::cout << ns.count() << " ns\n";
#endif // BENCHMARK

  return 0;
}
