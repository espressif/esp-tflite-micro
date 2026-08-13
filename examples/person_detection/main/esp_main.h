// Copyright 2020-2021 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sdkconfig.h"

// Use CLI_ONLY_INFERENCE mode when there is no camera path available:
// - on ESP32-P4 without the P4 function EV board BSP (camera comes via
//   esp_video there, selected by CONFIG_TFLITE_USE_BSP_P4_EV_FUNC)
// - on other targets whenever the esp32-camera component is not part
//   of the build
// Can also be defined manually to skip camera usage.
#if CONFIG_IDF_TARGET_ESP32P4
#if !CONFIG_TFLITE_USE_BSP_P4_EV_FUNC
#define CLI_ONLY_INFERENCE 1
#endif
#elif !__has_include("esp_camera.h")
#define CLI_ONLY_INFERENCE 1
#endif

// Enable this to get cpu stats
#define COLLECT_CPU_STATS 1

#if !defined(CLI_ONLY_INFERENCE)
// Enable display support if BSP is enabled in menuconfig
#if (CONFIG_TFLITE_USE_BSP)
#define DISPLAY_SUPPORT 1
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif
extern void run_inference(void *ptr);
#ifdef __cplusplus
}
#endif
