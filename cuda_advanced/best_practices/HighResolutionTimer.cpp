#define WIN32_LEAN_AND_MEAN
#include <windows.h>
 
#include "HighResolutionTimer.h"
 
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