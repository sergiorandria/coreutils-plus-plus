// This implementation is in progress ...

#ifndef _GNU_SOURCE 
#define _GNU_SOURCE 
#endif 

#include <iostream> 
#include <mutex> 
#include <string>
#include <vector>
#include <thread>
#include <climits>
#include <cstring>
#include <memory>
#include <future>
#include <type_traits>
#include <string_view>
#include <functional>

#if __cplusplus <= 201703L
#error "Need to be compiled with C++20" 
#endif // __cplusplus

namespace detail 
{ 
int hashFvn1(std::string_view __v)
{ 
    return INT_MAX;
}

} // namespace detail
 
namespace fs 
{



} // namespace fs 

namespace tp 
{ 

#ifdef LIMIT_HARDWARE_USAGE 
#ifndef BUILD_MAX_CORE 
#define BUILD_MAX_CORE 64 // Limit max cpu to 64 for 
                          // processor with more than that
#endif // BUILD_MAX_CORE 

#endif // LIMIT_HARDWARE_USAGE 

// For small buffer optimization 
#ifndef TASK_STACK_SIZE 
#define TASK_STACK_SIZE 1024
#endif // TASK_STACK_SIZE  

     
#ifndef N_CORE
// The explanation of this choice is inside fast-wc.cpp tp namespace. 
// The main difference is that we can now compile grep.cpp with 
// N_CORE defined at compile-time. 
inline unsigned int N_CORE = std::thread::hardware_concurrency(); 
#endif // N_CORE

template <size_t stackSize = TASK_STACK_SIZE>
class task_worker { 
// This class follows a multithreaded design pattern called 
// thread pool pattern. A task can only be performed three operations: 
// destroy, moved, invoke 
    public: 
        task_worker() = default; 

        // Create worker at compile time 
        template <typename F> constexpr task_worker(F&& f) 
        { 
            using DecayF = std::decay_t<F>;
            
            if constexpr (is_small_v<DecayF>()) 
                new (stack) DecayF(std::forward<F> (f));
            else 
                data = new char[stkSize+1];

            invoke_fn = [](void *p) 
            { 
                (*static_cast<DecayF*> (p))();
            };

            destroy_fn = [](void *p) 
            { 
                static_cast<DecayF*> (p)->~DecayF();
            };

            move_fn = [](void *src, void *dst) 
            { 
                new (dst) DecayF(*static_cast<DecayF*>(src));
                static_cast<DecayF*> (src)->~DecayF(); 
            };
        }

        task_worker(const task_worker&) = delete; 
        task_worker &operator=(const task_worker&) = delete;

    private:
        size_t stkSize = stackSize;

        std::function<void(void*)> *invoke_fn   = nullptr; 
        std::function<void(void*)> *destroy_fn  = nullptr; 
        std::function<void(void*)> *move_fn     = nullptr; 
        
        template <typename T>
        constexpr bool is_small_v() const 
        { 
            static_assert(sizeof(T) <= stkSize, 
                    "Task metadata too large for the thread stack"); 
            
            return stkSize <= TASK_STACK_SIZE; 
        }

        union 
        {
            alignas(std::max_align_t) char stack[stackSize];    // Inlined buffer for SBO
            alignas(std::max_align_t) char *data;               // Dynamically allocated buffer 
        };
}; 

template <bool EnablePriorityScheduling = true>
class ThreadPool 
{ 
    public: 
        using Priority = std::int8_t;
        
        ThreadPool() 
        {

        }
   
        ThreadPool &operator=(ThreadPool &mmAlloc)
        {
            return *this;
        }

    private: 
        std::vector<std::thread> threadPool;
};

} // namespace tp

namespace core 
{ 

class __grep_internal_class
{ 
    public: 
        static __grep_internal_class* Instance() 
        {

#ifdef OPTIMIZE_SINGLETON_BUILD
            // This was used for research purpose time and space 
            // complexity (will be deleted in the first stable version)
            static __grep_internal_class instance; 
            return &instance;
#else 
            if (instance == nullptr) 
            { 
                std::lock_guard<std::mutex> lock(gMutex); 
                if (instance == nullptr) 
                { 
                    instance = new __grep_internal_class(); 
                }
            } 

            return instance;
#endif // OPTIMIZE_SINGLETON_BUILD 
        } 

        void process() const 
        { 
        



        }

        __grep_internal_class(const __grep_internal_class& gObj) = delete; 
        __grep_internal_class(const __grep_internal_class&& gObj) = delete; 

    private: 
        __grep_internal_class() { }

        static __grep_internal_class *instance; 
        static std::mutex gMutex;
}; 

__grep_internal_class* __grep_internal_class::instance = nullptr; 
std::mutex __grep_internal_class::gMutex; 

} // namespace core 

int  
main(int argc, const char *argv[]) 
{ 
    std::ios_base::sync_with_stdio(false);  


    return 0; 
} 
