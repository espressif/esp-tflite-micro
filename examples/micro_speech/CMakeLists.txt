cmake_minimum_required(VERSION 3.5)
set(EXTRA_COMPONENT_DIRS
        $ENV{IDF_PATH}/examples/common_components/protocol_examples_common
        ../../components/tflite-lib)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(micro_speech)
