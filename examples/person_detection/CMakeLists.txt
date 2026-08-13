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

project(person_detection)
