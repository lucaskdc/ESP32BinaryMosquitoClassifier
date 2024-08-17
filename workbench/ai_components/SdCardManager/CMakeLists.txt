idf_component_register(SRCS "SdCardManager.c"
                        INCLUDE_DIRS "."
                        PRIV_REQUIRES fatfs vfs sdmmc)