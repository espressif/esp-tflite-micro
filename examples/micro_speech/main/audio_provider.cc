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

#include "audio_provider.h"

#include <cstdlib>
#include <cstring>

// FreeRTOS.h must be included before some of the following dependencies.
// Solves b/150260343.
// clang-format off
#include "freertos/FreeRTOS.h"
// clang-format on

#include "driver/i2s.h"
#include "esp_log.h"
#include "esp_spi_flash.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/task.h"
#include "ringbuf.h"
#include "micro_model_settings.h"

using namespace std;

// for c2 and c3, I2S support was added from IDF v4.4 onwards
#define NO_I2S_SUPPORT CONFIG_IDF_TARGET_ESP32C2 || \
                          (CONFIG_IDF_TARGET_ESP32C3 \
                          && (ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(4, 4, 0)))

static const char* TAG = "TF_LITE_AUDIO_PROVIDER";
/* ringbuffer to hold the incoming audio data */
ringbuf_t* g_audio_capture_buffer;
volatile int32_t g_latest_audio_timestamp = 0;
/* model requires 20ms new data from g_audio_capture_buffer and 10ms old data
 * each time , storing old data in the history buffer , {
 * history_samples_to_keep = 10 * 16 } */
constexpr int32_t history_samples_to_keep =
    ((kFeatureDurationMs - kFeatureStrideMs) *
     (kAudioSampleFrequency / 1000));
/* new samples to get each time from ringbuffer, { new_samples_to_get =  20 * 16
 * } */
constexpr int32_t new_samples_to_get =
    (kFeatureStrideMs * (kAudioSampleFrequency / 1000));

const int32_t kAudioCaptureBufferSize = 40000;
const int32_t i2s_bytes_to_read = 3200;

namespace {
int16_t g_audio_output_buffer[kMaxAudioSampleSize * 32];
bool g_is_audio_initialized = false;
int16_t g_history_buffer[history_samples_to_keep];

#if !NO_I2S_SUPPORT
uint8_t g_i2s_read_buffer[i2s_bytes_to_read] = {};
#if CONFIG_IDF_TARGET_ESP32
i2s_port_t i2s_port = I2S_NUM_1; // for esp32-eye
#else
i2s_port_t i2s_port = I2S_NUM_0; // for esp32-s3-eye
#endif
#endif
}  // namespace

#if NO_I2S_SUPPORT
  // nothing to be done here
#else
static void i2s_init(void) {
  // Start listening for audio: MONO @ 16KHz
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = 16000,
      .bits_per_sample = (i2s_bits_per_sample_t) 16,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_I2S,
      .intr_alloc_flags = 0,
      .dma_buf_count = 3,
      .dma_buf_len = 300,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = -1,
  };
#if CONFIG_IDF_TARGET_ESP32S3
  i2s_pin_config_t pin_config = {
      .bck_io_num = 41,    // IIS_SCLK
      .ws_io_num = 42,     // IIS_LCLK
      .data_out_num = -1,  // IIS_DSIN
      .data_in_num = 2,   // IIS_DOUT
  };
  i2s_config.bits_per_sample = (i2s_bits_per_sample_t) 32;
#else
  i2s_pin_config_t pin_config = {
      .bck_io_num = 26,    // IIS_SCLK
      .ws_io_num = 32,     // IIS_LCLK
      .data_out_num = -1,  // IIS_DSIN
      .data_in_num = 33,   // IIS_DOUT
  };
#endif

  esp_err_t ret = 0;
  ret = i2s_driver_install(i2s_port, &i2s_config, 0, NULL);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Error in i2s_driver_install");
  }
  ret = i2s_set_pin(i2s_port, &pin_config);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Error in i2s_set_pin");
  }

  ret = i2s_zero_dma_buffer(i2s_port);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Error in initializing dma buffer with 0");
  }
}
#endif

static void CaptureSamples(void* arg) {
#if NO_I2S_SUPPORT
  ESP_LOGE(TAG, "i2s support not available on C3 chip for IDF < 4.4.0");
#else
  size_t bytes_read = i2s_bytes_to_read;
  i2s_init();
  while (1) {
    /* read 100ms data at once from i2s */
    i2s_read(i2s_port, (void*)g_i2s_read_buffer, i2s_bytes_to_read,
             &bytes_read, pdMS_TO_TICKS(100));

    if (bytes_read <= 0) {
      ESP_LOGE(TAG, "Error in I2S read : %d", bytes_read);
    } else {
      if (bytes_read < i2s_bytes_to_read) {
        ESP_LOGW(TAG, "Partial I2S read");
      }
#if CONFIG_IDF_TARGET_ESP32S3
      // rescale the data
      for (int i = 0; i < bytes_read / 4; ++i) {
        ((int16_t *) g_i2s_read_buffer)[i] = ((int32_t *) g_i2s_read_buffer)[i] >> 14;
      }
      bytes_read = bytes_read / 2;
#endif
      /* write bytes read by i2s into ring buffer */
      int bytes_written = rb_write(g_audio_capture_buffer,
                                   (uint8_t*)g_i2s_read_buffer, bytes_read, pdMS_TO_TICKS(100));
      if (bytes_written != bytes_read) {
        ESP_LOGI(TAG, "Could only write %d bytes out of %d", bytes_written, bytes_read);
      }
      /* update the timestamp (in ms) to let the model know that new data has
       * arrived */
      g_latest_audio_timestamp = g_latest_audio_timestamp +
          ((1000 * (bytes_written / 2)) / kAudioSampleFrequency);
      if (bytes_written <= 0) {
        ESP_LOGE(TAG, "Could Not Write in Ring Buffer: %d ", bytes_written);
      } else if (bytes_written < bytes_read) {
        ESP_LOGW(TAG, "Partial Write");
      }
    }
  }
#endif
  vTaskDelete(NULL);
}

TfLiteStatus InitAudioRecording() {
  g_audio_capture_buffer = rb_init("tf_ringbuffer", kAudioCaptureBufferSize);
  if (!g_audio_capture_buffer) {
    ESP_LOGE(TAG, "Error creating ring buffer");
    return kTfLiteError;
  }
  /* create CaptureSamples Task which will get the i2s_data from mic and fill it
   * in the ring buffer */
  xTaskCreate(CaptureSamples, "CaptureSamples", 1024 * 4, NULL, 10, NULL);
  while (!g_latest_audio_timestamp) {
    vTaskDelay(1); // one tick delay to avoid watchdog
  }
  ESP_LOGI(TAG, "Audio Recording started");
  return kTfLiteOk;
}

TfLiteStatus GetAudioSamples1(int* audio_samples_size, int16_t** audio_samples)
{
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording();
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_audio_initialized = true;
  }
  int bytes_read =
    rb_read(g_audio_capture_buffer, (uint8_t*)(g_audio_output_buffer), 16000, 1000);
  if (bytes_read < 0) {
    ESP_LOGI(TAG, "Couldn't read data in time");
    bytes_read = 0;
  }
  *audio_samples_size = bytes_read;
  *audio_samples = g_audio_output_buffer;
  return kTfLiteOk;
}

TfLiteStatus GetAudioSamples(int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording();
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_audio_initialized = true;
  }
  /* copy 160 samples (320 bytes) into output_buff from history */
  memcpy((void*)(g_audio_output_buffer), (void*)(g_history_buffer),
         history_samples_to_keep * sizeof(int16_t));

  /* copy 320 samples (640 bytes) from rb at ( int16_t*(g_audio_output_buffer) +
   * 160 ), first 160 samples (320 bytes) will be from history */
  int bytes_read =
      rb_read(g_audio_capture_buffer,
              ((uint8_t*)(g_audio_output_buffer + history_samples_to_keep)),
              new_samples_to_get * sizeof(int16_t), pdMS_TO_TICKS(200));
  if (bytes_read < 0) {
    ESP_LOGE(TAG, " Model Could not read data from Ring Buffer");
  } else if (bytes_read < new_samples_to_get * sizeof(int16_t)) {
    ESP_LOGD(TAG, "RB FILLED RIGHT NOW IS %d",
             rb_filled(g_audio_capture_buffer));
    ESP_LOGD(TAG, " Partial Read of Data by Model ");
    ESP_LOGV(TAG, " Could only read %d bytes when required %d bytes ",
             bytes_read, (int) (new_samples_to_get * sizeof(int16_t)));
  }

  /* copy 320 bytes from output_buff into history */
  memcpy((void*)(g_history_buffer),
         (void*)(g_audio_output_buffer + new_samples_to_get),
         history_samples_to_keep * sizeof(int16_t));

  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }
