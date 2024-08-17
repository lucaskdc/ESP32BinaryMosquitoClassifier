#ifndef MELSPEC_H
#define MELSPEC_H
#include "fft.h"

typedef float mel_weight_t;

typedef struct {
    fft_config_t *pFFTConfig;
    FPRECISION   *pWindow;
    mel_weight_t *pMelWeights;
    FPRECISION   *pTmpFFTPower;
}
melspec_config_t;

int soft_pad_reflect(const int index, const int size, const uint16_t pad_size);
void hann_init(FPRECISION* window_hann, int nFFT);
FPRECISION hz_to_mel(FPRECISION fHz);
FPRECISION mel_to_hz(FPRECISION fMel);

#ifdef __cplusplus
template<uint16_t NFFT, uint16_t NMELS>
void MelWeightsInitTemplate(FPRECISION fSR, mel_weight_t weights[NMELS][1+NFFT/2]);

template<uint16_t NFFT, uint16_t NMELS>
int32_t MelSpecInitFromMemoryTemplate(void *pMem, fft_type_t type, fft_direction_t direction, FPRECISION fSR);

template<uint16_t NFFT, int SAMPLES, uint16_t NMELS, uint16_t HOPLENGHT>
void MelSpecTemplate(melspec_config_t *config, FPRECISION fOutput[NMELS][SAMPLES/HOPLENGHT+1], FPRECISION fInput[SAMPLES]);

//**************************************
// \brief Initialize Mel Weights from RFFT
// \description Replicates python librosa library algorithm weights. Requires preallocated weights memory.
// \note Uses 96 bytes of stack size.
// 
template<uint16_t NFFT, uint16_t NMELS>
void MelWeightsInitTemplate(FPRECISION fSR, mel_weight_t weights[NMELS][1+NFFT/2])
{
    FPRECISION min_mel = hz_to_mel(0);
    FPRECISION max_mel = hz_to_mel(fSR/2);
    
    for (int i=0; i<NMELS; ++i)
    {
        FPRECISION mels_f_i  = mel_to_hz(min_mel + (max_mel-min_mel)*i    /(NMELS+2-1)); //mels_f[i]
        FPRECISION mels_f_i1 = mel_to_hz(min_mel + (max_mel-min_mel)*(i+1)/(NMELS+2-1)); //mels_f[i+1]
        FPRECISION mels_f_i2 = mel_to_hz(min_mel + (max_mel-min_mel)*(i+2)/(NMELS+2-1)); //mels_f[i+2]
        
        FPRECISION fdiff_i  = mels_f_i1 - mels_f_i;     //fdiff[i]   = mels_f[i+1] - mels_f[i]
        FPRECISION fdiff_i1 = mels_f_i2 - mels_f_i1;    //fdiff[i+1] = mels_f[i+2] - mels_f[i+1]
        //# lower and upper slopes for all bins
        for (int k=0; k<1+NFFT/2; ++k)
        {   
            FPRECISION fftfreq_k = static_cast<FPRECISION>(k)/NFFT*fSR;

            FPRECISION ramps_ik  = mels_f_i  - fftfreq_k;
            FPRECISION ramps_i2k = mels_f_i2 - fftfreq_k;

            FPRECISION lower = -ramps_ik / fdiff_i;
            FPRECISION upper = ramps_i2k / fdiff_i1;

            //# .. then intersect them with each other and zero
            if (0 > lower || 0 > upper)
            {
                weights[i][k] = 0;
            }
            else
            {
                if (lower < upper)
                {
                    weights[i][k] = lower;
                }
                else
                {
                    weights[i][k] = upper;
                }
                //Slaney style mel normalization
                weights[i][k] *= 2.0 / (mels_f_i2-mels_f_i);
            }
        }
    }
}

template<uint16_t NFFT, int SAMPLES, uint16_t NMELS, uint16_t HOPLENGHT>
void MelSpecTemplate(melspec_config_t *config, FPRECISION fOutput[NMELS][SAMPLES/HOPLENGHT+1], FPRECISION fInput[SAMPLES]){
    const int nFrames = SAMPLES/HOPLENGHT+1;

    FPRECISION *arPower = config->pTmpFFTPower; //(1+NFFT/2) size
    fft_config_t *fft = config->pFFTConfig;
        
        for (int nFrame=0, nFrameOffset=0; nFrame<nFrames; ++nFrame, nFrameOffset+=HOPLENGHT)
        {
            for (int i=0; i<NFFT; ++i)
            {
                const int padded_sample_ind  = soft_pad_reflect(i+nFrameOffset, SAMPLES, NFFT/2);
                fft->input[i]  = fInput[padded_sample_ind]  * config->pWindow[i];
                fft->output[i] = 0;
            }
            fft_execute(fft);

            arPower[0] = fft->output[0] * fft->output[0];
            for (int i=1; i<NFFT/2; ++i)
            {
                arPower[i] = fft->output[i*2]   * fft->output[i*2] +
                             fft->output[i*2+1] * fft->output[i*2+1];
            }
            arPower[NFFT/2] = fft->output[1] * fft->output[1];
        
            for (int band=0; band<NMELS; ++band)
            {
                fOutput[band][nFrame] = 0;

                for (int i=0; i<1+NFFT/2; ++i)
                {
                    fOutput[band][nFrame] += reinterpret_cast<mel_weight_t (*)[1+NFFT/2]>(config->pMelWeights)[band][i] * arPower[i];
                }
            }
                
        }
}

template<uint16_t NFFT, uint16_t NMELS>
int32_t MelSpecInitFromMemoryTemplate(void *pMem, fft_type_t type, fft_direction_t direction, FPRECISION fSR)
{   
    char *pNextMem = reinterpret_cast<char*>(pMem);
    
    melspec_config_t* pConfigMel    = reinterpret_cast<melspec_config_t *>(pNextMem);
    pNextMem += sizeof(melspec_config_t);

    FPRECISION *input               = reinterpret_cast<FPRECISION *>(pNextMem);
    pNextMem += NFFT*sizeof(FPRECISION);

    FPRECISION *output              = reinterpret_cast<FPRECISION *>(pNextMem);;
    pNextMem += NFFT*sizeof(FPRECISION);

    FPRECISION *twiddle_factors     = reinterpret_cast<FPRECISION *>(pNextMem);;
    pNextMem += 2*NFFT*sizeof(FPRECISION);
    
    pConfigMel->pFFTConfig          = reinterpret_cast<fft_config_t*>(pNextMem);
    pNextMem += sizeof(fft_config_t);
    
    pConfigMel->pWindow             = reinterpret_cast<FPRECISION *>(pNextMem);;
    pNextMem += NFFT*sizeof(FPRECISION);

    pConfigMel->pMelWeights         = reinterpret_cast<mel_weight_t *>(pNextMem);;
    pNextMem += (NMELS)*(1+NFFT/2)*sizeof(mel_weight_t);

    pConfigMel->pTmpFFTPower        = reinterpret_cast<FPRECISION *>(pNextMem);;
    pNextMem += (1+NFFT/2)*sizeof(FPRECISION);

    /*int8_t err = */fft_init_no_malloc(NFFT, type, direction, pConfigMel->pFFTConfig, input, output, twiddle_factors);

    hann_init(pConfigMel->pWindow, NFFT);
    MelWeightsInitTemplate<NFFT, NMELS>(fSR, reinterpret_cast<mel_weight_t (*)[1+NFFT/2]>(pConfigMel->pMelWeights));
    return reinterpret_cast<int>(pNextMem)-reinterpret_cast<int>(pMem);
}

#endif

#endif