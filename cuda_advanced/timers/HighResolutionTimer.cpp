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

HighResolutionTimer::HighResolutionTimer(void)
{
    pImpl = new HighResolutionTimerImpl();
}
 
HighResolutionTimer::~HighResolutionTimer(void)
{
    delete pImpl;
}
 
void HighResolutionTimer::Tick()
{
    pImpl->Tick();
}
 
double HighResolutionTimer::ElapsedSeconds() const
{
    return pImpl->GetElapsedTimeInMicroSeconds() * 0.000001;
}
 
double HighResolutionTimer::ElapsedMilliSeconds() const
{
    return pImpl->GetElapsedTimeInMicroSeconds() * 0.001;
}
 
double HighResolutionTimer::ElapsedMicroSeconds() const
{
    return pImpl->GetElapsedTimeInMicroSeconds();
}