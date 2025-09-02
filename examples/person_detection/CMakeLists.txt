cmake_minimum_required(VERSION 3.5)

set(EXTRA_COMPONENT_DIRS static_images)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# Helper function to set required environment variables as per configuration set by user
function(add_bsp SDKCONFIG BSP)
    string(REGEX MATCH "CONFIG_${BSP}=y" REGEX_RESULT ${SDKCONFIG})
    if (REGEX_RESULT)
        set(ENV{${BSP}} 1) # If the config option is set then set corresponding environment variable to 1
    endif()
endfunction()

# 1. Define all variables used in main/idf_component.yml and set them to zero
set(ENV{TFLITE_USE_BSP_S3_EYE} 0)
set(ENV{TFLITE_USE_BSP_KORVO_2} 0)
set(ENV{TFLITE_USE_BSP_KALUGA} 0)

# 2. Set correct var to 1
# This is a workaround idf-component-manager limitation, where only
# target, idf_version and environment variables can be in the if-clause
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/sdkconfig)
    file(READ ${CMAKE_CURRENT_LIST_DIR}/sdkconfig SDKCONFIG_RULE)

    add_bsp("${SDKCONFIG_RULE}" "TFLITE_USE_BSP_S3_EYE") # Check for CONFIG_TFLITE_USE_BSP_S3_EYE option in sdkconfig
    add_bsp("${SDKCONFIG_RULE}" "TFLITE_USE_BSP_KORVO_2") # Check for CONFIG_TFLITE_USE_BSP_KORVO_2 option in sdkconfig
    add_bsp("${SDKCONFIG_RULE}" "TFLITE_USE_BSP_KALUGA") # Check for CONFIG_TFLITE_USE_BSP_KALUGA option in sdkconfig
endif()

project(person_detection)
