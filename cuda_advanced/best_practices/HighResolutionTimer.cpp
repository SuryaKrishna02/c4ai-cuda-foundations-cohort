#define WIN32_LEAN_AND_MEAN
#include <windows.h>
 
#include "HighResolutionTimer.h"

HighResolutionTimerImpl::HighResolutionTimerImpl()
: elapsedTime(0)
{
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t0);
}
 
void HighResolutionTimerImpl::Tick()
{
    QueryPerformanceCounter(&t1);
    // Compute the value in microseconds (1 second = 1,000,000 microseconds)
    elapsedTime = ( t1.QuadPart - t0.QuadPart ) * ( 1000000.0 / frequency.QuadPart );
 
    t0 = t1;
}
 
double HighResolutionTimerImpl::GetElapsedTimeInMicroSeconds()
{
    return elapsedTime;
}
 
class HighResolutionTimerImpl
{
public:
    HighResolutionTimerImpl();
 
    void Tick();
 
    double GetElapsedTimeInMicroSeconds();
 
private:
    LARGE_INTEGER t0, t1;
    LARGE_INTEGER frequency;
    double elapsedTime;
};