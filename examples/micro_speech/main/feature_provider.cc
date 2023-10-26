/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <esp_log.h>

#include <cstring>
#include "feature_provider.h"

#include "audio_provider.h"
#include "micro_features_generator.h"
#include "micro_model_settings.h"
#include "tensorflow/lite/micro/micro_log.h"

// extern const uint8_t no_30ms_start[]          asm("_binary_no_30ms_wav_start");
// extern const uint8_t yes_30ms_start[]         asm("_binary_yes_30ms_wav_start");
extern const uint8_t yes_1000ms_start[]       asm("_binary_yes_1000ms_wav_start");
extern const uint8_t no_1000ms_start[]        asm("_binary_no_1000ms_wav_start");
extern const uint8_t noise_1000ms_start[]     asm("_binary_noise_1000ms_wav_start");
extern const uint8_t silence_1000ms_start[]   asm("_binary_silence_1000ms_wav_start");

Features g_features;
const char *TAG = "feature_provider";

FeatureProvider::FeatureProvider(int feature_size, int8_t* feature_data)
    : feature_size_(feature_size),
      feature_data_(feature_data),
      is_first_run_(true) {
  // Initialize the feature data to default values.
  for (int n = 0; n < feature_size_; ++n) {
    feature_data_[n] = 0;
  }
}

FeatureProvider::~FeatureProvider() {}

TfLiteStatus FeatureProvider::PopulateFeatureData(
    int32_t last_time_in_ms, int32_t time_in_ms, int* how_many_new_slices) {
  if (feature_size_ != kFeatureElementCount) {
    MicroPrintf("Requested feature_data_ size %d doesn't match %d",
                feature_size_, kFeatureElementCount);
    return kTfLiteError;
  }

  // Quantize the time into steps as long as each window stride, so we can
  // figure out which audio data we need to fetch.
  const int last_step = (last_time_in_ms / kFeatureStrideMs);
  const int current_step = (time_in_ms / kFeatureStrideMs);

  int slices_needed = current_step - last_step;
  // If this is the first call, make sure we don't use any cached information.
  if (is_first_run_) {
    TfLiteStatus init_status = InitializeMicroFeatures();
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    ESP_LOGI(TAG, "InitializeMicroFeatures successful");
    is_first_run_ = false;
    slices_needed = kFeatureCount;
  }
#if 1
  if (slices_needed > kFeatureCount) {
    slices_needed = kFeatureCount;
  }
  *how_many_new_slices = slices_needed;

  const int slices_to_keep = kFeatureCount - slices_needed;
  const int slices_to_drop = kFeatureCount - slices_to_keep;
  // If we can avoid recalculating some slices, just move the existing data
  // up in the spectrogram, to perform something like this:
  // last time = 80ms          current time = 120ms
  // +-----------+             +-----------+
  // | data@20ms |         --> | data@60ms |
  // +-----------+       --    +-----------+
  // | data@40ms |     --  --> | data@80ms |
  // +-----------+   --  --    +-----------+
  // | data@60ms | --  --      |  <empty>  |
  // +-----------+   --        +-----------+
  // | data@80ms | --          |  <empty>  |
  // +-----------+             +-----------+
  if (slices_to_keep > 0) {
    for (int dest_slice = 0; dest_slice < slices_to_keep; ++dest_slice) {
      int8_t* dest_slice_data =
          feature_data_ + (dest_slice * kFeatureSize);
      const int src_slice = dest_slice + slices_to_drop;
      const int8_t* src_slice_data =
          feature_data_ + (src_slice * kFeatureSize);
      for (int i = 0; i < kFeatureSize; ++i) {
        dest_slice_data[i] = src_slice_data[i];
      }
    }
  }
  // Any slices that need to be filled in with feature data have their
  // appropriate audio data pulled, and features calculated for that slice.
  if (slices_needed > 0) {
    for (int new_slice = slices_to_keep; new_slice < kFeatureCount;
         ++new_slice) {
      const int new_step = (current_step - kFeatureCount + 1) + new_slice;
      const int32_t slice_start_ms = (new_step * kFeatureStrideMs);
      int16_t* audio_samples = nullptr;
      int audio_samples_size = 0;
      // TODO(petewarden): Fix bug that leads to non-zero slice_start_ms
      GetAudioSamples((slice_start_ms > 0 ? slice_start_ms : 0),
                      kFeatureDurationMs, &audio_samples_size,
                      &audio_samples);
      if (audio_samples_size < kMaxAudioSampleSize) {
        MicroPrintf("Audio data size %d too small, want %d",
                    audio_samples_size, kMaxAudioSampleSize);
        return kTfLiteError;
      }
      int8_t* new_slice_data = feature_data_ + (new_slice * kFeatureSize);
      // size_t num_samples_read;
      // TfLiteStatus generate_status = GenerateMicroFeatures(
      //     audio_samples, audio_samples_size, kFeatureSize,
      //     new_slice_data, &num_samples_read);
      TfLiteStatus generate_status = GenerateFeatures(
            audio_samples, audio_samples_size, &g_features);
      if (generate_status != kTfLiteOk) {
        return generate_status;
      }

      // copy features
      for (int j = 0; j < kFeatureSize; ++j) {
        new_slice_data[j] = g_features[0][j];
      }
    }
  }
#elif 1
    *how_many_new_slices = kFeatureCount;
    int16_t* audio_samples = nullptr;
    int audio_samples_size = 0;
    // GetAudioSamples(0, kFeatureDurationMs, &audio_samples_size, &audio_samples);
    static int cnt = 0;
    audio_samples = (int16_t *) (silence_1000ms_start + 44);

    switch(cnt++ % 4) {
      case 0:
        audio_samples = (int16_t *) (yes_1000ms_start + 44);
        break;
      case 1:
        audio_samples = (int16_t *) (no_1000ms_start + 44);
        break;
      case 2:
        audio_samples = (int16_t *) (noise_1000ms_start + 44);
        break;
      case 3:
        audio_samples = (int16_t *) (silence_1000ms_start + 44);
        break;
      default:
        break;
    }
    audio_samples_size = 16000;

    GetAudioSamples1(&audio_samples_size, &audio_samples);

    TfLiteStatus generate_status = GenerateFeatures(
          audio_samples, audio_samples_size, &g_features);
    if (generate_status != kTfLiteOk) {
      return generate_status;
    }
    // copy features
    for (int i = 0; i < kFeatureCount; ++i) {
      for (int j = 0; j < kFeatureSize; ++j) {
        feature_data_[i * kFeatureSize + j] = g_features[i][j];
      }
    }
    vTaskDelay(pdMS_TO_TICKS(500));
#else
    *how_many_new_slices = kFeatureCount;
    int16_t* audio_samples = nullptr;
    int audio_samples_size = 16000;
    GetAudioSamples(0, kFeatureDurationMs, &audio_samples_size, &audio_samples);

    memset(g_features, 0, sizeof(g_features));

    TfLiteStatus generate_status = GenerateFeatures(
          audio_samples, audio_samples_size, &g_features);
    if (generate_status != kTfLiteOk) {
      return generate_status;
    }
    // copy features
    for (int i = 0; i < kFeatureCount; ++i) {
      for (int j = 0; j < kFeatureSize; ++j) {
        feature_data_[i * kFeatureSize + j] = g_features[i][j];
      }
    }
    vTaskDelay(pdMS_TO_TICKS(500));
#endif
  return kTfLiteOk;
}
