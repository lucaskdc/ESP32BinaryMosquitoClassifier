#include "k2c_include.h"
#include "k2c_tensor_include.h"
#include "Esp32Utils.h"

#include "esp_log.h"

#include <string.h>

#include "esp32/himem.h"

//#define CONFIG_USE_HIMEM 1
//#define CONFIG_DIRECT_MAP_8MB 0
//#define CONFIG_IDF_TARGET_ESP32 1
//#define CONFIG_IDF_TARGET_ESP32S3 0

#define DENSE_DENOMINATOR (132/CONFIG_SPIRAM_BANKSWITCH_RESERVE) //must be divisor of 4224 (in case of using HIMEM, also must be a divisor of 132)
#define WRITE_DATA 0

#if (4224 % DENSE_DENOMINATOR)
#error "Dense denominator must be divisible by 4224"
#endif

#if (132 % DENSE_DENOMINATOR) && (!CONFIG_USE_HIMEM)
#error "If using HIMEM, dense denominator must be a divisor of 132"
#endif

#if ((132/DENSE_DENOMINATOR) > CONFIG_SPIRAM_BANKSWITCH_RESERVE)
#error "DENSE_DENOMINATOR DOESNT MATCH RESERVED SPIRAM TO BANKSWITCH"
#endif

#if CONFIG_IDF_TARGET_ESP32 && CONFIG_DIRECT_MAP_8MB
#error "ESP32 cannot map over 4MB external ram, change CONFIG_DIRECT_MAP_8MB"
#endif

#if CONFIG_IDF_RAGET_ESP32S3 && CONFIG_USE_HIMEM
#error "ESP32S3 DOESNT SUPPORT HIMEM and there is no need for it. Reset definition USE_HIMEM"
#endif


#if WRITE_DATA
	char nametmp[64];
	int iFile = 0;
	#define WRITE_OUTPUT(tensor) { 	sprintf(nametmp, "/sdcard/%02d.fbin", iFile++); \
										write_float_to_bin(nametmp, tensor.array, tensor.numel*sizeof(float));}
#else
	#define WRITE_OUTPUT(tensor) {}
#endif

float* CUSTOM_k2c_read_binary_float32_array_file(const char* filename, const size_t array_size){
    float* ptr = (float*)custom_malloc(array_size * sizeof(float));
    k2c_read_binary_float32_array_file_inplace(filename, array_size, ptr);
    return ptr;
}

const char TAG_MODEL[] = "MODELV1";

int bLoaded = 0;

float* conv2d_output_array; // var   70528*4 B = 282112B ~~ 9*32K = 294912 B
float* conv2d_kernel_array; // const 1152 B ~~ 1*32K
float* conv2d_bias_array;	// const  128 B ~~ 1*32K 

float* max_pooling2d_output_array; // var 17632*4 B = 70528 B ~~ 3*32K = 98304 B

float* conv2d_1_output_array;	//var 29376*4 B = 117504 B ~~ 4*32K = 
float* conv2d_1_kernel_array;   //const 73728 B ~~ 3*32K
float* conv2d_1_bias_array;		//const   256 B ~~ 1*32K

float* max_pooling2d_1_output_array; // var 6656*4 B = 26624 B ~~ 1*32K

float* conv2d_2_output_array; //var  4224*4 B = 16896 B ~~ 1*32K
float* conv2d_2_kernel_array; //const 147456 B ~~ 5*32K
float* conv2d_2_bias_array;   //const    256 B ~~ 1*32K

float* flatten_output_array;  //var (equal, flat) ~~ 0

float* dense_output_array;	  //var 256*4B = 1024B ~~ 1*32K
//float* dense_kernel_array;    //const 4325376 B = 132*32K
float* dense_bias_array;      //const    1024 B ~~ 1*32K
//float* dense_kernel_array_first_sixth;
float* dense_kernel_array_himem_mapped;
esp_himem_handle_t himem_handle_dense_kernel;
esp_himem_rangehandle_t himem_range_handle_dense_kernel;

float* dense_1_kernel_array;  //const    2048 B ~~ 1*32K
float* dense_1_bias_array;    //const       8 B ~~ 1*32K

#define CONV2D_K_FN "/sdcard/BinaryV1conv2d_kernel_array.bin_weight"
#define CONV2D_B_FN "/sdcard/BinaryV1conv2d_bias_array.bin_weight"

#define CONV2D_1_K_FN "/sdcard/BinaryV1conv2d_1_kernel_array.bin_weight"
#define CONV2D_1_B_FN "/sdcard/BinaryV1conv2d_1_bias_array.bin_weight"

#define CONV2D_2_K_FN "/sdcard/BinaryV1conv2d_2_kernel_array.bin_weight"
#define CONV2D_2_B_FN "/sdcard/BinaryV1conv2d_2_bias_array.bin_weight"

#define DENSE_K_FN "/sdcard/BinaryV1dense_kernel_array.bin_weight"
#define DENSE_B_FN "/sdcard/BinaryV1dense_bias_array.bin_weight"

#define DENSE_1_K_FN "/sdcard/BinaryV1dense_1_kernel_array.bin_weight"
#define DENSE_1_B_FN "/sdcard/BinaryV1dense_1_bias_array.bin_weight"

#define CONV2D_K_SIZE (288)
#define CONV2D_B_SIZE (32)

#define CONV2D_1_K_SIZE (18432)
#define CONV2D_1_B_SIZE (64)

#define CONV2D_2_K_SIZE (36864)
#define CONV2D_2_B_SIZE (64)

#define DENSE_K_SIZE (1081344)
#define DENSE_B_SIZE (256)

#define DENSE_1_K_SIZE (512)
#define DENSE_1_B_SIZE (2)

void fill_dummy(float* pArr, size_t size)
{
	for(size_t i=0; i<size; i++)
	{
		pArr[i] = 1e-6*(float)(rand() % 2000001 - 1000000);
	}
}

void model_binary_v1_load_himem(void (*delayWTD)(), int loadDummy) {
	ESP_LOGI(TAG_MODEL, "INIT");
	ESP_LOGI(TAG_MODEL, "iterations himem %d", DENSE_DENOMINATOR);
	ESP_LOGI(TAG_MODEL, "blkz mapped: %d",132/DENSE_DENOMINATOR);
	ESP_LOGI(TAG_MODEL, "blkz reserved: %d", CONFIG_SPIRAM_BANKSWITCH_RESERVE);


	ESP_LOGI(TAG_MODEL, "BEGIN LOAD");
	conv2d_output_array = custom_malloc(70528*sizeof(float)); // var   70528*4 B = 282112B ~~ 9*32K = 294912 B
	
	conv2d_kernel_array = custom_malloc(CONV2D_K_SIZE*sizeof(float)); // const 1152 B ~~ 1*32K
	conv2d_bias_array   = custom_malloc(CONV2D_B_SIZE*sizeof(float));	// const  128 B ~~ 1*32K
	if (!loadDummy) { 
		k2c_read_binary_float32_array_file_inplace(CONV2D_K_FN, CONV2D_K_SIZE, conv2d_kernel_array);
		k2c_read_binary_float32_array_file_inplace(CONV2D_B_FN, CONV2D_B_SIZE, conv2d_bias_array);
	} else {
		fill_dummy(conv2d_kernel_array, CONV2D_K_SIZE);
		fill_dummy(conv2d_bias_array, CONV2D_B_SIZE);
	}

	max_pooling2d_output_array = custom_malloc(17632*sizeof(float)); // var 17632*4 B = 70528 B ~~ 3*32K = 98304 B


	conv2d_1_output_array = conv2d_output_array;	//var 29376*4 B = 117504 B ~~ 4*32K = 

	conv2d_1_kernel_array = custom_malloc(CONV2D_1_K_SIZE*sizeof(float));   //const 73728 B ~~ 3*32K
	conv2d_1_bias_array   = custom_malloc(CONV2D_1_B_SIZE*sizeof(float));		//const   256 B ~~ 1*32K
	if(!loadDummy){
		k2c_read_binary_float32_array_file_inplace(CONV2D_1_K_FN, CONV2D_1_K_SIZE, conv2d_1_kernel_array);
		k2c_read_binary_float32_array_file_inplace(CONV2D_1_B_FN, CONV2D_1_B_SIZE, conv2d_1_bias_array);
	} else {
		fill_dummy(conv2d_1_kernel_array, CONV2D_1_K_SIZE);
		fill_dummy(conv2d_1_bias_array, CONV2D_1_B_SIZE);
	}

	max_pooling2d_1_output_array = max_pooling2d_output_array; // var 6656*4 B = 26624 B ~~ 1*32K


	conv2d_2_output_array = conv2d_1_output_array; //var  4224*4 B = 16896 B ~~ 1*32K

	conv2d_2_kernel_array = custom_malloc(CONV2D_2_K_SIZE*sizeof(float)); //const 147456 B ~~ 5*32K
	conv2d_2_bias_array   = custom_malloc(CONV2D_2_B_SIZE*sizeof(float));   //const    256 B ~~ 1*32K
	if(!loadDummy) {
		k2c_read_binary_float32_array_file_inplace(CONV2D_2_K_FN, CONV2D_2_K_SIZE, conv2d_2_kernel_array);
		k2c_read_binary_float32_array_file_inplace(CONV2D_2_B_FN, CONV2D_2_B_SIZE, conv2d_2_bias_array);
	} else {
		fill_dummy(conv2d_2_kernel_array, CONV2D_2_K_SIZE);
		fill_dummy(conv2d_2_bias_array, CONV2D_2_B_SIZE);
	}

	flatten_output_array  = conv2d_2_output_array;  //var (equal, flat) ~~ 0


	dense_output_array    = max_pooling2d_1_output_array;	  //var 256*4B = 1024B ~~ 1*32K
	dense_bias_array      = custom_malloc(DENSE_B_SIZE*sizeof(float)); //const    1024 B ~~ 1*32K

	ESP_LOGI(TAG_MODEL, "BEGIN HIMEM");
	ESP_LOGI(TAG_MODEL, "HIMEM: ALLOC");
	ESP_ERROR_CHECK(esp_himem_alloc(6*180224*sizeof(float), &himem_handle_dense_kernel));
	ESP_LOGI(TAG_MODEL, "HIMEM: MAP RANGE");
	ESP_ERROR_CHECK(esp_himem_alloc_map_range(256*4224*sizeof(float)/DENSE_DENOMINATOR, &himem_range_handle_dense_kernel));
	
	for(int i=0; i<DENSE_DENOMINATOR; ++i)
	{	
		const size_t nLenFile     = 256*4224/DENSE_DENOMINATOR; //180224*sizeof(float);
		const size_t nOffsetFile  = i*nLenFile;
		const size_t nLenHimem    = nLenFile*sizeof(float);
		const size_t nOffsetHimem = i*nLenHimem;
		ESP_ERROR_CHECK(esp_himem_map(himem_handle_dense_kernel, himem_range_handle_dense_kernel, 
			nOffsetHimem, 0, nLenHimem, 0, (void**)&dense_kernel_array_himem_mapped));

		if(!loadDummy) {
			k2c_read_binary_float32_array_file_offset_limit(DENSE_K_FN, nLenFile, nOffsetFile, dense_kernel_array_himem_mapped); //load kernel slice from file
		} else {
			fill_dummy(dense_kernel_array_himem_mapped, nLenFile);
		}
		
		ESP_ERROR_CHECK(esp_himem_unmap(himem_range_handle_dense_kernel, dense_kernel_array_himem_mapped, nLenHimem));
	}
	if(!loadDummy) {
		k2c_read_binary_float32_array_file_inplace(DENSE_B_FN, DENSE_B_SIZE, dense_bias_array);
	} else {
		fill_dummy(dense_bias_array, DENSE_B_SIZE);
	}

	dense_1_kernel_array  = custom_malloc(DENSE_1_K_SIZE*sizeof(float));  //const    2048 B ~~ 1*32K
	dense_1_bias_array    = custom_malloc(DENSE_1_B_SIZE*sizeof(float));  //const       8 B ~~ 1*32K
	if(!loadDummy) {
		k2c_read_binary_float32_array_file_inplace(DENSE_1_K_FN, DENSE_1_K_SIZE, dense_1_kernel_array);
		k2c_read_binary_float32_array_file_inplace(DENSE_1_B_FN, DENSE_1_B_SIZE, dense_1_bias_array);
	} else {
		fill_dummy(dense_1_kernel_array, DENSE_1_K_SIZE);
		fill_dummy(dense_1_bias_array, DENSE_1_B_SIZE);
	}
	bLoaded = 1;
}

//4MB HIMEM + 4MB ordinary PSRAM
//128 *32K  + 128*32K
void model_binary_v1_constrained(k2c_tensor* conv2d_input_input, k2c_tensor* dense_1_output, void (*delayWTD)()) { 

ESP_LOGI(TAG_MODEL, "BEGIN");

WRITE_OUTPUT((*conv2d_input_input));

//INPUT: 	conv2d_input_input
//OUTPUT:	conv2d_output
//2400 + 288 + 32 + 70258 = 285.07 kB
if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 01: conv2d");
size_t conv2d_stride[2] = {1,1}; 	//Seria possivel desalocar uma variável criada estaticamente?
size_t conv2d_dilation[2] = {1,1};
k2c_tensor conv2d_output = {conv2d_output_array,3,70528,{58,38,32, 1, 1}}; 
k2c_tensor conv2d_kernel = {conv2d_kernel_array,4,288,{ 3, 3, 1,32, 1}}; 
k2c_tensor conv2d_bias = {conv2d_bias_array,1,32,{32, 1, 1, 1, 1}}; 
k2c_conv2d(&conv2d_output,conv2d_input_input,&conv2d_kernel, 
	&conv2d_bias,conv2d_stride,conv2d_dilation,k2c_relu);
WRITE_OUTPUT(conv2d_output);

//if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 02: max_pooling2d");
//INPUT: 	conv2d_output
//OUTPUT:	max_pooling2d_output
//70528 + 17632 = 88160 = 344.375 kB
size_t max_pooling2d_stride[2] = {2,2}; 
size_t max_pooling2d_pool_size[2] = {2,2};
k2c_tensor max_pooling2d_output = {max_pooling2d_output_array,3,17632,{29,19,32, 1, 1}}; 
k2c_maxpool2d(&max_pooling2d_output,&conv2d_output,max_pooling2d_pool_size, 
	max_pooling2d_stride);
WRITE_OUTPUT(max_pooling2d_output);

if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 03: conv2d");
//INPUT: 	max_pooling2d_output
//OUTPUT:	conv2d_1_output
//17632 + 18432 + 64 + 29376 = 254.82 kB
size_t conv2d_1_stride[2] = {1,1}; 
size_t conv2d_1_dilation[2] = {1,1};
k2c_tensor conv2d_1_output = {conv2d_1_output_array,3,29376,{27,17,64, 1, 1}}; 
k2c_tensor conv2d_1_kernel = {conv2d_1_kernel_array,4,18432,{ 3, 3,32,64, 1}}; 
k2c_tensor conv2d_1_bias = {conv2d_1_bias_array,1,64,{64, 1, 1, 1, 1}}; 
k2c_conv2d(&conv2d_1_output,&max_pooling2d_output,&conv2d_1_kernel, 
	&conv2d_1_bias,conv2d_1_stride,conv2d_1_dilation,k2c_relu);
WRITE_OUTPUT(conv2d_1_output);

//if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 04: max_pooling2d");
//INPUT: 	conv2d_1_output
//OUTPUT:	max_pooling2d_1_output
//29376 + 6656 = 36032 = 140.75 kB
size_t max_pooling2d_1_stride[2] = {2,2}; 
size_t max_pooling2d_1_pool_size[2] = {2,2};
k2c_tensor max_pooling2d_1_output = {max_pooling2d_1_output_array,3,6656,{13, 8,64, 1, 1}}; 
k2c_maxpool2d(&max_pooling2d_1_output,&conv2d_1_output,max_pooling2d_1_pool_size, 
	max_pooling2d_1_stride);
WRITE_OUTPUT(max_pooling2d_1_output);

if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 05: conv2d");
//INPUT: 	max_pooling2d_1_output
//OUTPUT:	conv2d_2_output
//6656 + 36864 + 64 + 4224 = 47808 = 186.75 kB
size_t conv2d_2_stride[2] = {1,1}; 
size_t conv2d_2_dilation[2] = {1,1};
k2c_tensor conv2d_2_output = {conv2d_2_output_array,3,4224,{11, 6,64, 1, 1}}; 
k2c_tensor conv2d_2_kernel = {conv2d_2_kernel_array,4,36864,{ 3, 3,64,64, 1}}; 
k2c_tensor conv2d_2_bias = {conv2d_2_bias_array,1,64,{64, 1, 1, 1, 1}}; 
k2c_conv2d(&conv2d_2_output,&max_pooling2d_1_output,&conv2d_2_kernel, 
	&conv2d_2_bias,conv2d_2_stride,conv2d_2_dilation,k2c_relu);
WRITE_OUTPUT(conv2d_2_output);


//if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 06: flatten");
//================================
// dense_layer
//INPUT: 	conv2d_2_output
//OUTPUT:	flatten_output (shares same array of conv2d_2_output)
//4224 + 4224(0) = 8448(4224) = 33 kB (17 kB)
flatten_output_array = conv2d_2_output_array;
k2c_tensor flatten_output = {flatten_output_array,1,4224,{4224,   1,   1,   1,   1}}; 
WRITE_OUTPUT(flatten_output);


//if (delayWTD) delayWTD(); moved to inside reading loop
ESP_LOGI(TAG_MODEL,"Layer 07: dense");
//INPUT: 	flatten_output (shares same array of conv2d_2_output)
//OUTPUT:	dense_output
//4224 + 1081344 + 256 + 256 = 4242.5 kB !!!!
//4424 + nrows_dense_kernel*256 + 256 +256
//4424 + 1076*256 + 256 + 256 = 280392 float = 1121568 B
const size_t nrows_dense_kernel = 4224 / DENSE_DENOMINATOR;
const size_t nelems_nrows_dense_kernel = 256*nrows_dense_kernel; //180224
k2c_tensor dense_output = {dense_output_array,1,256,{256,  1,  1,  1,  1}}; 
k2c_tensor dense_kernel = {NULL,2,1081344,{4224, 256,   1,   1,   1}}; 
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
for(int i=0; i<DENSE_DENOMINATOR; i++){ //for each nrows_dense_kernel a time runs the iteraction
	ind_inner = i*nrows_dense_kernel;
	if (delayWTD) delayWTD();
	size_t inner_to = ind_inner + nrows_dense_kernel;
	if (inner_to > innerdim){
		inner_to = innerdim; //assures no access violation happens
	}

	const size_t nLenHimem    = nelems_nrows_dense_kernel*sizeof(float); //180224*sizeof(float);
	const size_t nOffsetHimem = i*nLenHimem;
	ESP_ERROR_CHECK(esp_himem_map(himem_handle_dense_kernel, himem_range_handle_dense_kernel, 
			nOffsetHimem, 0, nLenHimem, 0, (void**)&dense_kernel_array_himem_mapped));

	dense_kernel.array = dense_kernel_array_himem_mapped;
	k2c_matmul_low_memory_B_iter(dense_output.array,flatten_output.array,dense_kernel.array, //run multiplication iteraction
						1,outcols,innerdim,
						0,1,
						ind_inner, inner_to);

	ESP_ERROR_CHECK(esp_himem_unmap(himem_range_handle_dense_kernel, dense_kernel_array_himem_mapped, nLenHimem));
}

//Add bias after multiplication
for(size_t index=0; index < outsize; index++){
	dense_output.array[index] += dense_bias.array[index]; //only works if output and bias shape is equal
}

k2c_relu(dense_output.array,outsize); //apply layer activation function after dense transformation
WRITE_OUTPUT(dense_output);


if (delayWTD) delayWTD();
ESP_LOGI(TAG_MODEL,"Layer 08: dense");
//INPUT: 	dense_output
//OUTPUT:	dense_1_output
//512 + 2 + 2 = 516 floats = 2.02 kB
k2c_tensor dense_1_kernel = {dense_1_kernel_array,2,512,{256,  2,  1,  1,  1}}; 
k2c_tensor dense_1_bias = {dense_1_bias_array,1,2,{2,1,1,1,1}}; 
k2c_dense_1d(dense_1_output,&dense_output,&dense_1_kernel, 
	&dense_1_bias,k2c_sigmoid);
WRITE_OUTPUT((*dense_1_output));

ESP_LOGI(TAG_MODEL, "END");
}