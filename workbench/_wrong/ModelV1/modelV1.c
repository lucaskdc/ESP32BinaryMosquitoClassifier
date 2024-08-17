#include "k2c_include.h"
#include "k2c_tensor_include.h"
#include "Esp32Utils.h"

#include "esp_log.h"

#include <string.h>

float* CUSTOM_k2c_read_binary_float32_array_file(const char* filename, const size_t array_size){
    float* ptr = (float*)custom_malloc(array_size * sizeof(float));
    k2c_read_binary_float32_array_file_inplace(filename, array_size, ptr);
    return ptr;
}

const char TAG_MODEL[] = "MODELV1"; 

void model_binary_v1_constrained(k2c_tensor* conv2d_input_input, k2c_tensor* dense_1_output, void (*delayWTD)()) { 

float* conv2d_output_array;
float* conv2d_kernel_array;
float* conv2d_bias_array;
float* max_pooling2d_output_array;
float* conv2d_1_output_array;
float* conv2d_1_kernel_array;
float* conv2d_1_bias_array;
float* max_pooling2d_1_output_array;
float* conv2d_2_output_array;
float* conv2d_2_kernel_array;
float* conv2d_2_bias_array;
float* flatten_output_array;
float* dense_output_array;
float* dense_kernel_array;
float* dense_bias_array;
float* dense_1_kernel_array;
float* dense_1_bias_array;

ESP_LOGI(TAG_MODEL, "BEGIN");

//INPUT: 	conv2d_input_input
//OUTPUT:	conv2d_output
//2400 + 288 + 32 + 70258 = 285.07 kB
if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 01: conv2d");
size_t conv2d_stride[2] = {1,1}; 	//Seria possivel desalocar uma variável criada estaticamente?
size_t conv2d_dilation[2] = {1,1};
conv2d_output_array = custom_malloc(70528 * sizeof(float));
conv2d_kernel_array = CUSTOM_k2c_read_binary_float32_array_file("/sdcard/BinaryV1conv2d_kernel_array.bin_weight",288); 	 // 1kB
conv2d_bias_array = CUSTOM_k2c_read_binary_float32_array_file("/sdcard/BinaryV1conv2d_bias_array.bin_weight",32); 		 // 128 B
k2c_tensor conv2d_output = {conv2d_output_array,3,70528,{58,38,32, 1, 1}}; 
k2c_tensor conv2d_kernel = {conv2d_kernel_array,4,288,{ 3, 3, 1,32, 1}}; 
k2c_tensor conv2d_bias = {conv2d_bias_array,1,32,{32, 1, 1, 1, 1}}; 
k2c_conv2d(&conv2d_output,conv2d_input_input,&conv2d_kernel, 
	&conv2d_bias,conv2d_stride,conv2d_dilation,k2c_relu);
//Serial.print("input pointer:") 
//free(conv2d_input_input->array);
free(conv2d_kernel.array);
free(conv2d_bias.array);
//write_float_to_bin("/sdcard/test_cases_result/0-1.bin_predicted", conv2d_output.array, conv2d_output.numel*sizeof(float));

//if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 02: max_pooling2d");
//INPUT: 	conv2d_output
//OUTPUT:	max_pooling2d_output
//70528 + 17632 = 88160 = 344.375 kB
size_t max_pooling2d_stride[2] = {2,2}; 
size_t max_pooling2d_pool_size[2] = {2,2};
max_pooling2d_output_array = custom_malloc(17632 * sizeof(float));
k2c_tensor max_pooling2d_output = {max_pooling2d_output_array,3,17632,{29,19,32, 1, 1}}; 
k2c_maxpool2d(&max_pooling2d_output,&conv2d_output,max_pooling2d_pool_size, 
	max_pooling2d_stride);
free(conv2d_output.array);
//write_float_to_bin("/sdcard/test_cases_result/0-2.bin_predicted", max_pooling2d_output.array, max_pooling2d_output.numel*sizeof(float));

if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 03: conv2d");
//INPUT: 	max_pooling2d_output
//OUTPUT:	conv2d_1_output
//17632 + 18432 + 64 + 29376 = 254.82 kB
size_t conv2d_1_stride[2] = {1,1}; 
size_t conv2d_1_dilation[2] = {1,1};
conv2d_1_output_array = custom_malloc(29376 * sizeof(float));
conv2d_1_kernel_array = CUSTOM_k2c_read_binary_float32_array_file("/sdcard/BinaryV1conv2d_1_kernel_array.bin_weight",18432); 			// 75 kB
conv2d_1_bias_array = CUSTOM_k2c_read_binary_float32_array_file("/sdcard/BinaryV1conv2d_1_bias_array.bin_weight",64); 					// 128 kB
k2c_tensor conv2d_1_output = {conv2d_1_output_array,3,29376,{27,17,64, 1, 1}}; 
k2c_tensor conv2d_1_kernel = {conv2d_1_kernel_array,4,18432,{ 3, 3,32,64, 1}}; 
k2c_tensor conv2d_1_bias = {conv2d_1_bias_array,1,64,{64, 1, 1, 1, 1}}; 
k2c_conv2d(&conv2d_1_output,&max_pooling2d_output,&conv2d_1_kernel, 
	&conv2d_1_bias,conv2d_1_stride,conv2d_1_dilation,k2c_relu);
free(conv2d_1_kernel.array);
free(conv2d_1_bias.array);
free(max_pooling2d_output.array);
//write_float_to_bin("/sdcard/test_cases_result/0-3.bin_predicted", conv2d_1_output.array, conv2d_1_output.numel*sizeof(float));

//if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 04: max_pooling2d");
//INPUT: 	conv2d_1_output
//OUTPUT:	max_pooling2d_1_output
//29376 + 6656 = 36032 = 140.75 kB
size_t max_pooling2d_1_stride[2] = {2,2}; 
size_t max_pooling2d_1_pool_size[2] = {2,2};
max_pooling2d_1_output_array = custom_malloc(6656 * sizeof(float)); 
k2c_tensor max_pooling2d_1_output = {max_pooling2d_1_output_array,3,6656,{13, 8,64, 1, 1}}; 
k2c_maxpool2d(&max_pooling2d_1_output,&conv2d_1_output,max_pooling2d_1_pool_size, 
	max_pooling2d_1_stride);
free(conv2d_1_output.array);
//write_float_to_bin("/sdcard/test_cases_result/0-4.bin_predicted", max_pooling2d_1_output.array, max_pooling2d_1_output.numel*sizeof(float));

if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 05: conv2d");
//INPUT: 	max_pooling2d_1_output
//OUTPUT:	conv2d_2_output
//6656 + 36864 + 64 + 4224 = 47808 = 186.75 kB
size_t conv2d_2_stride[2] = {1,1}; 
size_t conv2d_2_dilation[2] = {1,1};
conv2d_2_output_array = custom_malloc(4224 * sizeof(float));
conv2d_2_kernel_array = CUSTOM_k2c_read_binary_float32_array_file("/sdcard/BinaryV1conv2d_2_kernel_array.bin_weight",36864); 			// 144 kB
conv2d_2_bias_array = CUSTOM_k2c_read_binary_float32_array_file("/sdcard/BinaryV1conv2d_2_bias_array.bin_weight",64); 					// 256 B
k2c_tensor conv2d_2_output = {conv2d_2_output_array,3,4224,{11, 6,64, 1, 1}}; 
k2c_tensor conv2d_2_kernel = {conv2d_2_kernel_array,4,36864,{ 3, 3,64,64, 1}}; 
k2c_tensor conv2d_2_bias = {conv2d_2_bias_array,1,64,{64, 1, 1, 1, 1}}; 
k2c_conv2d(&conv2d_2_output,&max_pooling2d_1_output,&conv2d_2_kernel, 
	&conv2d_2_bias,conv2d_2_stride,conv2d_2_dilation,k2c_relu);
free(max_pooling2d_1_output.array);
free(conv2d_2_kernel.array);
free(conv2d_2_bias.array);
//write_float_to_bin("/sdcard/test_cases_result/0-5.bin_predicted", conv2d_2_output.array, conv2d_2_output.numel*sizeof(float));

//if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 06: flatten");
//================================
// dense_layer
//INPUT: 	conv2d_2_output
//OUTPUT:	flatten_output (shares same array of conv2d_2_output)
//4224 + 4224(0) = 8448(4224) = 33 kB (17 kB)
flatten_output_array = conv2d_2_output_array;
k2c_tensor flatten_output = {flatten_output_array,1,4224,{4224,   1,   1,   1,   1}}; 
//k2c_flatten_inplace(&flatten_output); //flatten_output tensor alreay has the correct nelem and shape on initialization
//write_float_to_bin("/sdcard/test_cases_result/0-6.bin_predicted", flatten_output.array, flatten_output.numel*sizeof(float));

//if (delayWTD) delayWTD(); moved to inside reading loop
ESP_LOGI(TAG_MODEL,"Layer 07: dense");
//INPUT: 	flatten_output (shares same array of conv2d_2_output)
//OUTPUT:	dense_output
//4224 + 1081344 + 256 + 256 = 4242.5 kB !!!!
//4424 + nrows_dense_kernel*256 + 256 +256
//4424 + 1076*256 + 256 + 256 = 280392 float = 1121568 B
dense_output_array = custom_malloc(256 * sizeof(float));
const size_t nrows_dense_kernel = 1076; //Performance adjustment parameter
dense_kernel_array = custom_malloc(nrows_dense_kernel*256*sizeof(float)); 
dense_bias_array = CUSTOM_k2c_read_binary_float32_array_file("/sdcard/BinaryV1dense_bias_array.bin_weight",256); 						// 1 kB		
k2c_tensor dense_output = {dense_output_array,1,256,{256,  1,  1,  1,  1}}; 
k2c_tensor dense_kernel = {dense_kernel_array,2,1081344,{4224, 256,   1,   1,   1}}; 
k2c_tensor dense_bias = {dense_bias_array,1,256,{256,  1,  1,  1,  1}}; 
//float dense_fwork[1085568] = {0}; 
/*
k2c_dense_1d(&dense_output,&flatten_output,&dense_kernel, 
	&dense_bias,k2c_relu);
*/


//////////////////////////////////////////////
//	Low RAM Optimized Dense 1D layer for big kernel size
const size_t outcols = dense_kernel.shape[1];
const size_t innerdim = dense_kernel.shape[0];
const size_t outsize = 1*outcols;

size_t ind_inner;
memset(dense_output.array, 0, outsize*sizeof(float)); //reset output array to add up iteractions
for(ind_inner=0; ind_inner<4224; ind_inner += nrows_dense_kernel){ //for each nrows_dense_kernel a time runs the iteraction
	if (delayWTD) delayWTD();
	size_t inner_to = ind_inner + nrows_dense_kernel;
	if (inner_to > innerdim){
		inner_to = innerdim; //assures no access violation happens
	}
	k2c_read_binary_float32_array_file_offset_limit("/sdcard/BinaryV1dense_kernel_array.bin_weight",outcols*nrows_dense_kernel,outcols*ind_inner, dense_kernel.array); //load kernel slice from file
	k2c_matmul_low_memory_B_iter(dense_output.array,flatten_output.array,dense_kernel.array, //run multiplication iteraction
						1,outcols,innerdim,
						0,1,
						ind_inner, inner_to);
	
}

//Add bias after multiplication
for(size_t index=0; index < outsize; index++){
	dense_output.array[index] += dense_bias.array[index]; //only works if output and bias shape is equal
}

free(flatten_output.array); //free layer's input pointer
free(dense_kernel.array);   //free kernel pointer
free(dense_bias.array);     //free bias

k2c_relu(dense_output.array,outsize); //apply layer activation function after dense transformation

//write_float_to_bin("/sdcard/test_cases_result/0-7.bin_predicted", dense_output.array, dense_output.numel*sizeof(float));

if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 08: dense");
//INPUT: 	dense_output
//OUTPUT:	dense_1_output
//512 + 2 + 2 = 516 floats = 2.02 kB
//dense_1_output_array = custom_malloc(2 * sizeof(float));	//SAIDA
dense_1_kernel_array = CUSTOM_k2c_read_binary_float32_array_file("/sdcard/BinaryV1dense_1_kernel_array.bin_weight",512); 				// 2 kB		
dense_1_bias_array = CUSTOM_k2c_read_binary_float32_array_file("/sdcard/BinaryV1dense_1_bias_array.bin_weight",2); 						// 8 B		
k2c_tensor dense_1_kernel = {dense_1_kernel_array,2,512,{256,  2,  1,  1,  1}}; 
k2c_tensor dense_1_bias = {dense_1_bias_array,1,2,{2,1,1,1,1}}; 
//float dense_1_fwork[768] = {0}; 
k2c_dense_1d(dense_1_output,&dense_output,&dense_1_kernel, 
	&dense_1_bias,k2c_sigmoid);
free(dense_1_kernel.array);
free(dense_1_bias.array);
free(dense_output.array);
//write_float_to_bin("/sdcard/test_cases_result/0-8.bin_predicted", dense_1_output->array, dense_1_output->numel*sizeof(float));

ESP_LOGI(TAG_MODEL, "END");

}
