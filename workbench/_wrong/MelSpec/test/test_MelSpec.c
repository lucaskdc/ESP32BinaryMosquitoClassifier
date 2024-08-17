//--------------------------------------------------
// Filename: test_MelSpec.h
// Project:  Mosquitoramento
// Author:   Lucas K Dal Castel
// Date:     2023-Feb-05
// Version:  0.1
//
// Description: This file holds a test case for MelSpec component
// it uses input_mel and expected_mel mocks generated as static variables that are at same directory.
//--------------------------------------------------

#include "unity.h"
#include "esp_heap_caps.h"
#include "MelSpecCWrapper.h"
#include "esp_log.h"

extern double input_mel[9984];
double* output_mel;
extern double expected_mel[60][40];
melspec_config_t* melConfig;

double MaxDiff(double a[60][40], double b[60][40]){
  double max_val = 0;
  for (int j=0; j<40; ++j){
    for (int i=0; i<60; ++i){
      double diff = fabs(a[i][j] - b[i][j]);

      if (diff > max_val)
          max_val = diff;
    }
  }

  return max_val;
}

TEST_CASE("Test MelSpec 1..9984", "[MELSPEC]"){
  printf("START ALLOCATING MEL SPEC\n");
  void *pMem = heap_caps_malloc(168288, MALLOC_CAP_SPIRAM);//custom_malloc(168228);

  printf("START INIT MELSPEC\n");
  MelSpecInitFromMemory(pMem, FFT_REAL, FFT_FORWARD, 8000);  // requires 168228 bytes;
  melConfig = pMem;

  printf("START ALLOCATING OUTPUT PSRAM\n");
  output_mel = (double *)heap_caps_malloc(60 * 40 * sizeof(double), MALLOC_CAP_SPIRAM);

  printf("RUN MELSPEC\n");
  MelSpec(melConfig, output_mel, input_mel);
  
  printf("GET DIFF\n");
  double max_diff = MaxDiff((double (*)[40])output_mel, expected_mel);

  printf("abs max_diff = %lf\n", max_diff);

  printf("JUST ASSERT < 1e-5");
  TEST_ASSERT( max_diff < 1e-5);

  free(pMem);
  free(output_mel);
}
