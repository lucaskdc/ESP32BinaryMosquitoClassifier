/*
 * SPDX-FileCopyrightText: 2010-2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: CC0-1.0
 */

#include <stdio.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_log.h"
#include <stdint.h>

#include "ClassifierManager.h"
#include "SdCardManager.h"

void app_main(void)
{
    printf("Hello world!\n");

    /* Print chip information */
    esp_chip_info_t chip_info;
    uint32_t flash_size;
    esp_chip_info(&chip_info);
    printf("This is %s chip with %d CPU core(s), WiFi%s%s, ",
           CONFIG_IDF_TARGET,
           chip_info.cores,
           (chip_info.features & CHIP_FEATURE_BT) ? "/BT" : "",
           (chip_info.features & CHIP_FEATURE_BLE) ? "/BLE" : "");

    printf("silicon revision %d, ", chip_info.revision);
    if(esp_flash_get_size(NULL, &flash_size) != ESP_OK) {
        printf("Get flash size failed");
        return;
    }

    printf("%luMB %s flash\n", flash_size / (1024 * 1024),
           (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded" : "external");

    printf("Minimum free heap size: %ld bytes\n", esp_get_minimum_free_heap_size());

    static uint8_t input[9984*2] = {0};
    static float output[2];
    for (int i=0; i<sizeof(input)/sizeof(*input); i+=2)
    {
        *(int16_t*)(&input[i])=(i/2)%80-40;
    }

    mount_sdcard();
    test_sdcard();
    #if CONFIG_COMPILER_OPTIMIZATION_SIZE
    ESP_LOGI("MAIN"," optimize: size:"); 
    #endif

    #if CONFIG_COMPILER_OPTIMIZATION_PERF
    ESP_LOGI("MAIN", "optimize: perf O2");
    #endif

    #if CONFIG_SPIRAM_CACHE_WORKAROUND
    ESP_LOGI("MAIN", "YES SPIRAM CACHE WORKAROUND");
    #else
    ESP_LOGI("MAIN", "NO SPIRAM CACHE WORKAROUND");
    #endif

    //ESP_LOGI("MAIN", "FREQ: %d", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);

    

    ESP_LOGI("MAIN", "INITING...");
    init_classifierV1();
    ESP_LOGI("MAIN", "INIT OK");

    ESP_LOGI("MAIN", "START CLASSIFIER 1 ");
    run_classifierV1_16bits(input, output);
    ESP_LOGI("MAIN", "FINISH_CLASSIFIER 1 ");
    printf("result = %f , %f\n", output[0], output[1]);

    ESP_LOGI("MAIN", "START CLASSIFIER 2 ");
    run_classifierV1_16bits(input, output);
    ESP_LOGI("MAIN", "FINISH_CLASSIFIER 2 ");
    printf("result = %f , %f\n", output[0], output[1]);

    ESP_LOGI("MAIN", "START CLASSIFIER 3 ");
    run_classifierV1_16bits(input, output);
    ESP_LOGI("MAIN", "FINISH_CLASSIFIER 3 ");
    printf("result = %f , %f\n", output[0], output[1]);
    
    //test_classifierV1();

    unmount_sdcard();


    for (int i = 10; i >= 0; i--) {
        printf("Restarting in %d seconds...\n", i);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    printf("Restarting now.\n");
    fflush(stdout);
    esp_restart();

}
