#include "ClassifierManager.h"

#include "Esp32Utils.h"
#include "MelSpecCWrapper.h"
#include "SdCardManager.h"
#include "esp_log.h"
#include "k2c_tensor_include.h"
#include "math.h"
#include "modelV1.h"
#include "fft.h"

float *input_layer_array;
float output_layer_array[2];
melspec_config_t *melConfig;
k2c_tensor input_layer;
k2c_tensor output_layer;

FPRECISION *input_mel;
FPRECISION *output_mel;

const char TAG[] = "CLASSF-MNG";

void init_classifierV1() {
  input_layer_array = custom_malloc(2400 * sizeof(float));
  void *pMem = custom_malloc(168228);

// pConfigMel	16	16
// input	8192	8208  (FPRECISION as double)
// output	8192	16400 (FPRECISION as double)
// twiddle_factors	16384	32784 (FPRECISION as double)
// pFFTConfig	28	32812 (FPRECISION as double)
// pWindow	8192	41004 (FPRECISION as double)
// pMelWeights	123120	164124 (mel weights dtype=float32) (as in filters.py)
// pTmpFFTPower	4104	168228 ()
                                 
  size_t uMemUsed = MelSpecInitFromMemory(pMem, FFT_REAL, FFT_FORWARD, 8000);  // requires 168228 bytes;
  ESP_LOGI(TAG, "Init Mel spec mem usage: %d", uMemUsed);
  
  melConfig = pMem;

  input_layer.array = input_layer_array;
  input_layer.ndim = 3;
  input_layer.numel = 2400;
  input_layer.shape[0] = 60;
  input_layer.shape[1] = 40;
  input_layer.shape[2] = 1;
  input_layer.shape[3] = 1;
  input_layer.shape[4] = 1;

  output_layer.array = output_layer_array;
  output_layer.ndim = 1;
  output_layer.numel = 2;
  output_layer.shape[0] = 2;
  output_layer.shape[1] = 1;
  output_layer.shape[2] = 1;
  output_layer.shape[3] = 1;
  output_layer.shape[4] = 1;

  input_mel = (FPRECISION *)custom_malloc(9984 * sizeof(FPRECISION));
  output_mel = (FPRECISION *)custom_malloc(60 * 40 * sizeof(FPRECISION));

  //model_binary_v1_load_himem(&TaskDelayWTD);
  model_binary_v1_load_himem(&TaskDelayWTD, 0);
}

inline FPRECISION max(FPRECISION a, FPRECISION b) {
  return (a > b) ? a : b;
}

FPRECISION max_array(FPRECISION a[], size_t size) {
  FPRECISION max_value = a[0];
  for (size_t i = 1; i < size; i++) {
    max_value = max(max_value, a[i]);
  }
  return max_value;
}

FPRECISION power_to_db(FPRECISION value, FPRECISION ref_value) {
  // const double ref_value = 1.0;
  const FPRECISION amin = 1e-10;
  // const double top_db = 80;

  value = fabs(value);

  value = log10(max(amin, value));
  value -= log10(max(amin, ref_value));
  value *= 10;
  return value;
}

void power_to_db_array(FPRECISION output[60*40], FPRECISION input[60*40]) {
  // SAFE to call on same array output and input
  const FPRECISION top_db = 80;
  FPRECISION max_input = max_array(input, 60*40);
  //printf("MAX ARRAY %lf", max_input);

  // from librosa.power_to_db()
  // spectrum_max - top_db
  // ref_value is maximum of time series so max of spectrum is 0
  // therefore, min-value should be 0-top_db which

  for (int i = 0; i < 60*40; i++) {
    output[i] = power_to_db(input[i], max_input);
    output[i] = max(output[i], -top_db);
  }
}

void read_float_from_file(const char* name, float* buffer, int n, int size)
{
    ESP_LOGI("[LOAD]","Begin... %s",name);
    FILE *f = fopen(name, "rb");
    fseek(f, n*sizeof(float)*size, SEEK_SET);
    if (f == NULL) {
        ESP_LOGE("[LOAD]", "Failed to open %s for reading", name);
        return;
    }

    int elements_read = 0;

    //fread((char*)(buffer), 1, size, f);
    while (!(feof(f)) && (elements_read < size)){
        elements_read += fread((char*)buffer, sizeof(float), size-elements_read, f);     
    }
    
    fclose(f);
    ESP_LOGI("[LOAD]","End");
}

int run_classifierV1(float output[2]);

int run_classifierV1_FPRECISION16bits(uint8_t *input, float output[2]) {
  if (is_sdcard_mounted()) {
    ESP_LOGI(TAG, "Run ClassifierV1 - 16 bits");

    for (int i = 0; i < 9984; i++) {
      // WAV is little endian as ESP32
      const int16_t val = ((int16_t *)(input))[i];
      input_mel[i] = (FPRECISION)val / (1 << 15);
    }

    return run_classifierV1(output);
    
  } else {
    ESP_LOGI(TAG, "NOT running ClassifierV1: SD card not mounted");
    return -1;
  }
}

float buffer[9984];
int test_classifierV1(){
  float tmpOutput[2];
  const char *name = "/sdcard/outputs120_pn_2_2.fbin";
  ESP_LOGI("[WRITE]","Begin... %s",name);
  FILE *f = fopen(name, "wb");
  if (f == NULL) {
    ESP_LOGE("[WRITE]", "Failed to open %s for writing", name);
    return -1;
  }

  
  for(int i=0; i<120; i++){
    ESP_LOGI("Test", "Iteration %d", i+1);
    read_float_from_file("/sdcard/p120_9984.fbin", buffer, i, 9984);
    for(int j=0; j<9984; j++)
      input_mel[j] = buffer[j];
    run_classifierV1(tmpOutput);

    ESP_LOGI("Test", "Write p%d  [0]:%f,  [1]:%f",i+1,tmpOutput[0],tmpOutput[1]);
    fwrite((char*)(tmpOutput), 1, 8, f);

    read_float_from_file("/sdcard/n120_9984.fbin", buffer, i, 9984);
    for(int j=0; j<9984; j++){
      input_mel[j] = buffer[j];
      //if (!(j%100))
      //  ESP_LOGI("??", "j=%d   buf=%f  mel=%lf", j, buffer[j], input_mel[j]);
    }
    run_classifierV1(tmpOutput);

    ESP_LOGI("Test", "Write n%d  [0]:%f,  [1]:%f",i+1,tmpOutput[0],tmpOutput[1]);
    fwrite((char*)(tmpOutput), 1, 8, f);
  }

  fclose(f);
  ESP_LOGI("[WRITE]","End");
  return 0;
}

int run_classifierV1(float output[2]) {

    uint32_t beginMelTime = 0;
    uint32_t endMelTime   = 0;
    uint32_t beginModelV1 = 0;
    uint32_t endModelV1   = 0;

    #if FPRECISION_DOUBLE
    #define EXT_BIN ".dbin"
    #else
    #define EXT_BIN ".fbin"
    #endif

#define WRITE_DATA 0

#if WRITE_DATA
    write_float_to_bin("/sdcard/MelInp" EXT_BIN, input_mel, 9984*sizeof(FPRECISION));
#endif

    ESP_LOGI(TAG, "Start Mel");
    beginMelTime = esp_log_timestamp();
    MelSpec(melConfig, output_mel, input_mel);

#if WRITE_DATA
    write_float_to_bin("/sdcard/TodBInp" EXT_BIN, output_mel, 60*40*sizeof(FPRECISION));
#endif

    ESP_LOGI(TAG, "Start dB");
    power_to_db_array(output_mel, output_mel);

#if WRITE_DATA
    write_float_to_bin("/sdcard/NormInp" EXT_BIN, output_mel, 60*40*sizeof(FPRECISION));
#endif

    ESP_LOGI(TAG, "Start Norm");
    // Normalization
    for (int i = 0; i < 60*40; i++) {
      output_mel[i] /= (FPRECISION)80.0;
      output_mel[i] += (FPRECISION)1.0;
    }
    
    endMelTime = esp_log_timestamp();

    for (int i = 0; i < 60 * 40; i++) {
      input_layer.array[i] = output_mel[i];
    }

    beginModelV1 = esp_log_timestamp();
    model_binary_v1_constrained(&input_layer, &output_layer, &TaskDelayWTD);
    output[0] = output_layer.array[0];
    output[1] = output_layer.array[1];

    endModelV1 = esp_log_timestamp();

    ESP_LOGW(TAG, "T Mel: %lu", endMelTime-beginMelTime);
    ESP_LOGW(TAG, "T Copy: %lu", beginModelV1-endMelTime);
    ESP_LOGW(TAG, "T Model: %lu", endModelV1-beginModelV1);
    ESP_LOGW(TAG, "TimeTotal: %lu", endModelV1-beginMelTime);

    return 0;
}
