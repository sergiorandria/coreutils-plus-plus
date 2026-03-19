#ifndef __BENCHMARK_H 
#define __BENCHMARK_H 

#define BENCHMARK

#include <mutex>
#include <functional>

// Partial implementation of a Benchmark class
namespace Benchmark {
class Benchmark { 
    public: 
        static Benchmark* Instance(); 

        void setLineImpl(auto& f); 
        void setWordImpl(auto& f); 
        void setCharImpl(auto& f); 
        void setBytesImpl(auto& f); 

        auto &getLineImpl() const; 
        auto &getWordImpl() const; 
        auto &getCharImpl() const; 
        auto &getBytesImpl() const; 

        constexpr ~Benchmark(); 

    private: 
        constexpr Benchmark(); 

        static Benchmark *instance;
        static std::mutex benchmarkLock;  

        std::function<size_t(std::char_traits<char> (std::char_traits<char>) )> __line_impl; 
        std::function<size_t(std::char_traits<char> (std::char_traits<char>) )> __word_impl; 
        std::function<size_t(std::char_traits<char> (std::char_traits<char>) )> __char_impl; 
        std::function<size_t(std::char_traits<char> (std::char_traits<char>) )> __bytes_impl; 
};
}
#endif 