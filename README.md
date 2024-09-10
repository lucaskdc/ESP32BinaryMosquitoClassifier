In summary the flow is this:

1. fixed_binary/binary.py -- creates and somewhat trains the neural network
   it outputs the model_binary.json and model_binary.h5 that are keras files for the model architecture and its weights

2. Verificacao.ipynb is a jupyter lab notebook used to convert the files above in binary files containing the weights for each layer (used to generate the *.bin_weight files).
  Also it takes 120 random samples of input audio for negative and positive classification (file n120_9984.fbin with 120 audio segments of 9984 samples each of mosquitos that are not aedes aegypti, and p120_9984.fbin that are of aedes aegypti)

3. In order to use the HIMEM feature of esp-idf, I need to fix it in the esp-idf 5.0 with the files "esp32\esp_psram-espidf5.0-patched.c" replacing the corresponding esp_psram.c inside the esp-idf installation folder.
This file is built during usual build process of any project using the esp_psram component from esp-idf. No need to manually rebuild this.

4. Then with the correct flags built with the code in the esp32 we may burn the firmware in the esp32 with write data enabled,
5. so it may generate the "outputs120_pn_2_2---FLOATFPRECISION.fbin" and "outputs120_pn_2_2---DOUBLEFPRECISION.fbin" with the output layer for each segment provided. This is used to check if the implemented model is matematically consistent.
6. Disabling writing the outputs, we may figure out the time used to compute the inputs.

Currently the process is not automatic and neither fully documentend. I intented to do it as soon is it possible, considering the time available and community demand.
   
