#include "MelSpec.h"
int soft_pad_reflect(const int index, const int size, const uint16_t pad_size)
{
    //e.g padsize = 2; size = 7
    //      10 20 30 40 50 60 70       (before padding)
    //30 20 10 20 30 40 50 60 70 60 50 (after padding)
    // input | output --- test case
    // 0     | 2
    // 1     | 1
    // 2     | 0
    // ...
    // 8     | 6
    // 9     | 5
    // 10    | 4
    //reflect without duplicate extremities

    if (index<pad_size)  //index of padded virtual array
        return pad_size-index; //return index of value on real array (not padded)
    else if (index > (size+pad_size-1))
        return (size-1) - (index-(size+pad_size-1));
    else
        return index-pad_size;
}

void multiply_window(FPRECISION* output, FPRECISION* input, FPRECISION* window, int n)
{
    for (int i=0; i<n; ++i){
        output[i]=input[i]*window[i];
    }
}

void hann_init(FPRECISION* window_hann, int nFFT)
{
    const FPRECISION nPi = 3.141592653589793;
    for(int i=0; i<nFFT; ++i)
    {
        window_hann[i] = (1-cos(2*nPi*i/nFFT))/2;
    }
}


FPRECISION hz_to_mel(FPRECISION fHz)
{
    constexpr FPRECISION fFMin = 0.0;
    constexpr FPRECISION fSp = 200.0 / 3;
    constexpr FPRECISION fLogStep = 0.0687517774209491228; //log(6.4) / 27.0;  //step size for log region
    constexpr FPRECISION fMinLogHz = 1000.0;  // beginning of log region (Hz)
    constexpr FPRECISION fMinLogMel = (fMinLogHz - fFMin) / fSp;  // same (Mels)
    
    if (fHz < fMinLogHz)
    {
        return (fHz - fFMin) / fSp;
    }
    else
    {
        return fMinLogMel + log(fHz / fMinLogHz) / fLogStep;
    }
}

FPRECISION mel_to_hz(FPRECISION fMel)
{
    constexpr FPRECISION fFMin = 0.0;
    constexpr FPRECISION fSp = 200.0 / 3;
    constexpr FPRECISION fLogStep = 0.0687517774209491228; //log(6.4) / 27.0;  //step size for log region
    constexpr FPRECISION fMinLogHz = 1000.0;  // beginning of log region (Hz)
    constexpr FPRECISION fMinLogMel = (fMinLogHz - fFMin) / fSp;  // same (Mels)
    
    if (fMel < fMinLogMel)
    {
        return fFMin + fSp*fMel;
    }
    else
    {
        return fMinLogHz * exp(fLogStep * (fMel - fMinLogMel));
    }
}
