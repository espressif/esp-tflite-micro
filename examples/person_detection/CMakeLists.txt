cmake_minimum_required(VERSION 3.5)

set(EXTRA_COMPONENT_DIRS static_images)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

function(add_bsp SDKCONFIG BSP TARGET)
    string(REGEX MATCH "CONFIG_${BSP}=y" REGEX_RESULT ${SDKCONFIG})
    if (REGEX_RESULT)
        set(ENV{${BSP}} "${TARGET}")
    endif()
endfunction()

# 1. Define all variables used in main/idf_component.yml
set(ENV{TFLITE_USE_BSP_S3_EYE} "false")
set(ENV{TFLITE_USE_BSP_KORVO_2} "false")
set(ENV{TFLITE_USE_BSP_KALUGA} "false")

# 2. Set correct var to 'target'
# This is a workaround idf-component-manager limitation, where only
# target and idf_version can be in the if-clause
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/sdkconfig)
    file(READ ${CMAKE_CURRENT_LIST_DIR}/sdkconfig SDKCONFIG_RULE)

    add_bsp("${SDKCONFIG_RULE}" "TFLITE_USE_BSP_S3_EYE" "esp32s3")
    add_bsp("${SDKCONFIG_RULE}" "TFLITE_USE_BSP_KORVO_2" "esp32s3")
    add_bsp("${SDKCONFIG_RULE}" "TFLITE_USE_BSP_KALUGA" "esp32s2")
endif()

project(person_detection)
