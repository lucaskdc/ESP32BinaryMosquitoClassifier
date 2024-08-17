#include "k2c_tensor_include.h"

#ifdef __cplusplus
extern "C"{
#endif

//void model_binary_v1_constrained(k2c_tensor* conv2d_input_input, k2c_tensor* dense_1_output);
void model_binary_v1_constrained(k2c_tensor* conv2d_input_input, k2c_tensor* dense_1_output, void (*delayWTD)());

#ifdef __cplusplus
}
#endif