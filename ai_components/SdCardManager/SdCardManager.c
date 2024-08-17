#include "SdCardManager.h"
#include "sdmmc_cmd.h"
#include "driver/spi_common.h" //SPI_DMA_CH_AUTO
#include "driver/sdspi_host.h"
#include "esp_vfs_fat.h"
#include <string.h>

#define TAG "SDSPI"

sdmmc_host_t host = SDSPI_HOST_DEFAULT();
sdmmc_card_t* card;
char g_SdCardMounted = 0;

void mount_sdcard(void)
{
    //host.max_freq_khz = 18000; //it's not necessary on the new prototype, i guess the sdcard didn't had compliant pull-up resistors before.

    esp_err_t ret;
    // Options for mounting the filesystem.
    // If format_if_mount_failed is set to true, SD card will be partitioned and
    // formatted in case when mounting fails.
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        //.max_files = 5,
        .max_files = 16,
        //.allocation_unit_size = 16 * 1024
        .allocation_unit_size   = 64 * 1024
    };
    ESP_LOGI(TAG, "Initializing SD card");

    spi_bus_config_t bus_cfg = {
        .mosi_io_num = SDSPI_MOSI,
        .miso_io_num = SDSPI_MISO,
        .sclk_io_num = SDSPI_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000,
    };
    ret = spi_bus_initialize((spi_host_device_t)host.slot, &bus_cfg, SPI_DMA_CHAN);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize bus.");
        return;
    }

    // This initializes the slot without card detect (CD) and write protect (WP) signals.
    // Modify slot_config.gpio_cd and slot_config.gpio_wp if your board has these signals.
    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = SDSPI_CS;
    slot_config.host_id = (spi_host_device_t)host.slot;

    ret = esp_vfs_fat_sdspi_mount(SD_MOUNT_POINT, &host, &slot_config, &mount_config, &card);

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount filesystem.");
        } else {
            ESP_LOGE(TAG, "Failed to initialize the card (%s). "
                "Make sure SD card lines have pull-up resistors in place.", esp_err_to_name(ret));
        }
        return;
    }

    // Card has been initialized, print its properties
    sdmmc_card_print_info(stdout, card);

    //It would return before reaching here if it failed.
    g_SdCardMounted = 1;
}

void unmount_sdcard(void)
{
    // All done, unmount partition and disable SPI peripheral
    esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
    ESP_LOGI(TAG, "Card unmounted");
    // Deinitialize the bus after all devices are removed
    spi_bus_free((spi_host_device_t)host.slot);
    g_SdCardMounted = 0;
}

void test_sdcard(void)
{
    ESP_LOGI("sdcard test: ", "Starting");
    FILE *f = fopen("/sdcard/FOO.TXT", "w");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to open FOO.TXT for writing");
        return;
    }

    fprintf(f, "escrita ok\n");
    fclose(f);

    f = fopen("/sdcard/FOO.TXT", "r");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to open FOO.TXT for reading");
        return;
    }

    // Read a line from file
    char line[64];
    fgets(line, sizeof(line), f);
    fclose(f);

    // Strip newline
    char *pos = strchr(line, '\n');
    if (pos) {
        *pos = '\0';
    }
    ESP_LOGI(TAG, "Read from file: '%s'", line);
}