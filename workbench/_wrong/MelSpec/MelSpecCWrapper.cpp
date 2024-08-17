#include "MelSpecCWrapper.h"

int32_t MelSpecInitFromMemory(void *pMem, fft_type_t type, fft_direction_t direction, FPRECISION fSR){
    return MelSpecInitFromMemoryTemplate<1024, 60>(pMem, type, direction, fSR);
}

void MelSpec(melspec_config_t *config, FPRECISION *fOutput, FPRECISION *fInput){
    MelSpecTemplate<1024, 9984, 60, 256>(config, reinterpret_cast<FPRECISION (*)[40]>(fOutput), fInput);
}
