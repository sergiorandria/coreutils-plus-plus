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
#include <type_traits>
#include <string_view>
#include <functional>

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

    private: 
        std::function<void(void*)> *invoke_fn   = nullptr; 
        std::function<void(void*)> *destroy_fn  = nullptr; 
        std::function<void(void*)> *move_fn     = nullptr; 
        
        union 
        {
            alignas(std::max_align_t) char stack[stackSize];    // Inlined buffer for SBO
            alignas(std::max_align_t) char *data;               // Dynamically allocated buffer 
        }
}; 

class PoolBuilder
{ 
    class Pool;
    public: 
        PoolBuilder() { }

        Pool* buildPool() 
        { 
            return nullptr;
        }
};

class Pool 
{ 
    public: 
        Pool(int num_threads = N_CORE) 
        {

        }
   
        Pool &operator=(Pool &mmAlloc)
        {
            memcpy(&builder, &mmAlloc.builder, sizeof(PoolBuilder));
            return *this;
        }
    private: 
        std::vector<std::thread> threadPool;
        PoolBuilder builder;
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
