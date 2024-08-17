#include "MelSpec.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t MelSpecInitFromMemory(void *pMem, fft_type_t type, fft_direction_t direction, FPRECISION fSR);
void MelSpec(melspec_config_t *config, FPRECISION *fOutput, FPRECISION *fInput);

#ifdef __cplusplus
}
#endif