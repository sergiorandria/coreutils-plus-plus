#include "Benchmark.hpp"

constexpr Benchmark::Benchmark::Benchmark() { }

constexpr Benchmark::Benchmark::~Benchmark() { }

Benchmark::Benchmark* Benchmark::Benchmark::Instance() { 
    if (instance == nullptr) {
        std::unique_lock<std::mutex> lock(benchmarkLock);
        if (instance == nullptr) {
            instance = new Benchmark();
        }    
    }

    return instance; 
}

void Benchmark::Benchmark::setLineImpl(auto& f) { 
    __line_impl = f; 
}
        
void Benchmark::Benchmark::setWordImpl(auto& f) {
    __word_impl = f; 
}  

void Benchmark::Benchmark::setCharImpl(auto& f) {
    __word_impl = f; 
}  

void Benchmark::Benchmark::setBytesImpl(auto& f) {
    __word_impl = f; 
}  

auto &Benchmark::Benchmark::getLineImpl() const { return __line_impl; } 
auto &Benchmark::Benchmark::getWordImpl() const { return __word_impl; } 
auto &Benchmark::Benchmark::getCharImpl() const { return __char_impl; } 
auto &Benchmark::Benchmark::getBytesImpl() const { return __bytes_impl; } 

