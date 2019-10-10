#ifndef SCOPEDTIMER_H_INCLUDED
#define SCOPEDTIMER_H_INCLUDED

#include <chrono>
#include <iostream>

class ScopedTimer final
{
public:
    
    explicit ScopedTimer(const std::string &targetName) :
        startTime(std::chrono::high_resolution_clock::now())
    {
        std::cout << targetName << std::endl;
    }
    
    ~ScopedTimer()
    {
        const auto endTime = std::chrono::high_resolution_clock::now();
        const auto milliSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - this->startTime).count();
        std::cout << "Done (" << milliSeconds << " ms)" << std::endl;
    }
    
private:
    
    std::chrono::high_resolution_clock::time_point startTime;
    
    TINYRNN_DISALLOW_COPY_AND_ASSIGN(ScopedTimer);
};


#endif  // SCOPEDTIMER_H_INCLUDED
