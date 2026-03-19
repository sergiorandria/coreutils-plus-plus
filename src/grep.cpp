#ifndef _GNU_SOURCE 
#define _GNU_SOURCE 
#endif 

#include <iostream> 
#include <mutex> 
#include <string>
#include <thread>
#include <type_traits>
#include <string_view>

namespace detail 
{ 
int hashFvn1(string_view __v)
{ 



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
     
#ifndef N_CORE
// The explanation of this choice is inside fast-wc.cpp tp namespace. 
// The main difference is that we can now compile grep.cpp with 
// N_CORE defined at compile-time. 
inline unsigned int N_CORE = std::thread::hardware_concurrency(); 



struct thread
{ 
    int id;
    
};

class ::Pool; 

class PoolBuilder
{ 
    public: 
        PoolBuilder() { }

        Pool* buildPool() { } 
};

class Pool 
{ 
    public: 
        Pool() 
        { 
            threadPool = buildPool(); 
        }
    
    private: 
        std::vector<thread> threadPool;
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
main(int argc, const *argv[]) 
{ 
    ios_base::sync_with_stdio(false);  


    return 0; 
} 
