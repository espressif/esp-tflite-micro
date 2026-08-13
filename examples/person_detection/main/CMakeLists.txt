
#
# Main component of TF Micro project 'person_detection'.
#

set(pd_priv_reqs console static_images spi_flash driver)
# gpio driver lives in its own component since IDF v5.3 (required on v6+)
if("${IDF_VERSION_MAJOR}.${IDF_VERSION_MINOR}" VERSION_GREATER_EQUAL "5.3")
    list(APPEND pd_priv_reqs esp_driver_gpio)
endif()

idf_component_register(
    SRCS
        "detection_responder.cc"
        "image_provider.cc"
        "main.cc"
        "main_functions.cc"
        "model_settings.cc"
        "person_detect_model_data.cc"
        "esp_video_if.c"
        "app_camera_esp.c"
        "esp_cli.c"

    PRIV_REQUIRES ${pd_priv_reqs}
    INCLUDE_DIRS "")
