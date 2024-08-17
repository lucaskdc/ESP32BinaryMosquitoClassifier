#include "Esp32Utils.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

float* custom_malloc(size_t size){
    float *ptr = (float*)heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
    log_psram();
    return ptr;
}

void custom_free(void* ptr){
    heap_caps_free(ptr);
}

void log_ram(){
    size_t freeHeap = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t fullHeap = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    ESP_LOGI("[ESP RAM]", "Heap free: %d/%d \tUSED:%d", freeHeap,fullHeap,fullHeap-freeHeap);

    freeHeap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    fullHeap = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI("[ESP SPI RAM]", "Heap free: %d/%d \tUSED:%d", freeHeap,fullHeap,fullHeap-freeHeap);
}

void log_psram()
{
    size_t nFreeHeap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t nLargestBlockFree = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    ESP_LOGI("[ESP RAM]","Free PSRAM: %d : Largest block: %d",nFreeHeap, nLargestBlockFree);
}

void write_float_to_bin(const char* name, float* buffer, int size){
    ESP_LOGI("[WRITE]","Begin... %s",name);
    FILE *f = fopen(name, "wb");
    if (f == NULL) {
        ESP_LOGE("[WRITE]", "Failed to open %s for writing", name);
        return;
    }
    fwrite((char*)(buffer), 1, size, f);
    fclose(f);
    ESP_LOGI("[WRITE]","End");
}

void TaskDelayWTD(void)
{
    vTaskDelay(5);
}