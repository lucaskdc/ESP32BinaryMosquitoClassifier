#ifndef SDCARDMANAGER_H
#define SDCARDMANAGER_H

#ifdef __cplusplus
extern "C"{
#endif

#define SD_MOUNT_POINT "/sdcard"
#define SDSPI_CS CONFIG_SDSPI_CS
#define SDSPI_CLK CONFIG_SDSPI_CLK
#define SDSPI_MISO CONFIG_SDSPI_MISO
#define SDSPI_MOSI CONFIG_SDSPI_MOSI

#define SPI_DMA_CHAN SPI_DMA_CH_AUTO

void mount_sdcard(void);
void unmount_sdcard(void);
void test_sdcard(void);
int is_sdcard_mounted(void);

#ifdef __cplusplus
}
#endif

#endif