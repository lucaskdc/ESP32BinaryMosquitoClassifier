
#include <stddef.h>

#ifdef __cplusplus
extern "C"{
#endif

float* custom_malloc(size_t size);
void custom_free(void*);

void log_ram(void);
void log_psram(void);

void write_float_to_bin(const char* name, float* buffer, int size);

void TaskDelayWTD(void);

#ifdef __cplusplus
}
#endif