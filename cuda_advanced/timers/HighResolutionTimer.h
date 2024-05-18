#pragma once

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
 
class HighResolutionTimer
{
public:
    HighResolutionTimer(void);
    ~HighResolutionTimer(void);
 
    // "Tick" the timer to compute the amount of time since the last it was ticked (or since the timer was created).
    void Tick();
 
    double ElapsedSeconds() const;
    double ElapsedMilliSeconds() const;
    double ElapsedMicroSeconds() const;
 
private:
    HighResolutionTimerImpl* pImpl;
};
