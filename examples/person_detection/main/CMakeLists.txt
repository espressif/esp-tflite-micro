
#
# Main component of TF Micro project 'person_detection'.
#

idf_component_register(
    SRCS
        "detection_responder.cc"
        "image_provider.cc"
        "main.cc"
        "main_functions.cc"
        "model_settings.cc"
        "person_detect_model_data.cc"
        "app_camera_esp.c"
        "esp_cli.c"
        "pcb_qianzi_model_data.cc"
    PRIV_REQUIRES console tflite-lib esp32-camera static_images spi_flash
    INCLUDE_DIRS "")
