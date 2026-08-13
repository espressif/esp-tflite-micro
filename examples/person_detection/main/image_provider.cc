/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "esp_video_if.h"
#include "detection_responder.h"
#include "string.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <cstddef>
#include <stdint.h>

#if (CONFIG_TFLITE_USE_BSP)
#include "bsp/esp-bsp.h"
#endif

#include "esp_heap_caps.h"
#include "esp_log.h"

#include "app_camera_esp.h"
/* esp32-camera is not part of the build on targets/configs without camera */
#if __has_include("esp_camera.h")
#include "esp_camera.h"
#endif
#include "model_settings.h"
#include "image_provider.h"
#include "esp_main.h"

#if DISPLAY_SUPPORT
#include "draw/sw/lv_draw_sw_utils.h"
#endif

static const char* TAG = "app_camera";
static uint16_t* display_buf;

// Get the camera module ready
TfLiteStatus InitCamera() {
#if CLI_ONLY_INFERENCE
  ESP_LOGI(TAG, "CLI_ONLY_INFERENCE enabled, skipping camera init");
  return kTfLiteOk;
#endif
// if display support is present, initialise display buf
#if DISPLAY_SUPPORT
  if (display_buf == NULL) {
    // Size of display_buf:
#ifdef CONFIG_TFLITE_USE_BSP_P4_EV_FUNC
    display_buf = (uint16_t *) heap_caps_malloc(CAM_WIDTH * CAM_HEIGHT * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#else
    // Frame 96x96 from camera is extrapolated to 192x192. RGB565 pixel format -> 2 bytes per pixel
    display_buf = (uint16_t *) heap_caps_malloc(96 * 2 * 96 * 2 * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#endif
  }
  if (display_buf == NULL) {
    ESP_LOGE(TAG, "Couldn't allocate display buffer");
    return kTfLiteError;
  }
#endif // DISPLAY_SUPPORT

#if ESP_CAMERA_SUPPORTED | CONFIG_TFLITE_USE_BSP_P4_EV_FUNC
  int ret = app_camera_init();
  if (ret != 0) {
    MicroPrintf("Camera init failed\n");
    return kTfLiteError;
  }
  MicroPrintf("Camera Initialized\n");
#else
  ESP_LOGE(TAG, "Camera not supported for this device");
  return kTfLiteError;
#endif
  return kTfLiteOk;
}

void scale_rgb565_to_gray(const int old_width, const int old_height, const int new_width, const int new_height, uint16_t *buf, int8_t *output_buf) {

  float ht_ratio = (float) old_height / new_height;
  float wh_ratio = (float) old_width  / new_width;

  float ht = 0;
  float wh = 0;

  for(int i = 0; i < new_height; i++) {
    wh = 0;
    for (int j = 0; j < new_width; j++) {
      wh += wh_ratio;
      uint16_t inference_pixel = buf[((int) ht * old_width) + (int) wh];

      // for inference
      uint8_t hb = inference_pixel & 0xFF;
      uint8_t lb = inference_pixel >> 8;
      uint8_t r = (lb & 0x1F) << 3;
      uint8_t g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
      uint8_t b = (hb & 0xF8);

      /**
       * Gamma corected rgb to greyscale formula: Y = 0.299R + 0.587G + 0.114B
       * for effiency we use some tricks on this + quantize to [-128, 127]
       */
      uint8_t grey_pixel = ((305 * r + 600 * g + 119 * b) >> 10) - 128;

      // quantize to int8
      output_buf[(i * new_width) + j] = grey_pixel ^ 0x80;
    }

    ht += ht_ratio;
  }
}

void image_provider_get_display_buf(uint16_t **buf, uint32_t *w, uint32_t *h) {
  *buf = display_buf;
#if CONFIG_TFLITE_USE_BSP_P4_EV_FUNC
  *w = CAM_WIDTH;
  *h = CAM_HEIGHT;
#else
  *w = 192;
  *h = 192;
#endif
}

// Get an image from the camera module
TfLiteStatus GetImage(int image_width, int image_height, int channels, int8_t* image_data) {
#if CONFIG_TFLITE_USE_BSP_P4_EV_FUNC
  video_fb_t *frame = esp_video_if_get_frame();
  if (frame == NULL) {
    ESP_LOGE(TAG, "Camera capture failed");
    return kTfLiteError;
  }

  assert(frame->height == CAM_HEIGHT);
  assert(frame->width == CAM_WIDTH);
  memcpy(display_buf, frame->buf, frame->len);

  scale_rgb565_to_gray(frame->width, frame->height, image_width, image_height, (uint16_t *) frame->buf, image_data);
  esp_video_if_release_frame(frame);
  return kTfLiteOk;

#elif ESP_CAMERA_SUPPORTED
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    ESP_LOGE(TAG, "Camera capture failed");
    return kTfLiteError;
  }

#if DISPLAY_SUPPORT
  // In case if display support is enabled, we initialise camera in rgb mode
  // Hence, we need to convert this data to grayscale to send it to tf model
  // For display we extra-polate the data to 192X192

  // point to the last quarter of buffer
  uint16_t* cam_buf = display_buf + (96 * 96 * 3);
  memcpy((uint8_t*)cam_buf, fb->buf, fb->len);
  esp_camera_fb_return(fb);

  scale_rgb565_to_gray(kNumCols, kNumRows, image_width, image_height, cam_buf, image_data);

  // for display
  lv_draw_sw_rgb565_swap(cam_buf, 96 * 96);
  for (int i = 0; i < kNumRows; i++) {
    for (int j = 0; j < kNumCols; j++) {
      uint16_t pixel = cam_buf[i * kNumCols + j];
      display_buf[2 * i * kNumCols * 2 + 2 * j] = pixel;
      display_buf[2 * i * kNumCols * 2 + 2 * j + 1] = pixel;
      display_buf[(2 * i + 1) * kNumCols * 2 + 2 * j] = pixel;
      display_buf[(2 * i + 1) * kNumCols * 2 + 2 * j + 1] = pixel;
    }
  }
#else // DISPLAY_SUPPORT
  MicroPrintf("Image Captured\n");
  // We have initialised camera to grayscale
  // Just quantize to int8_t
  for (int i = 0; i < image_width * image_height; i++) {
    image_data[i] = ((uint8_t *) fb->buf)[i] ^ 0x80;
  }

  esp_camera_fb_return(fb);
#endif // DISPLAY_SUPPORT
  /* here the esp camera can give you grayscale image directly */
  return kTfLiteOk;
#else
  return kTfLiteError;
#endif
}
