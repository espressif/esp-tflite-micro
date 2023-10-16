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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_FEATURES_GENERATOR_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_FEATURES_GENERATOR_H_

#include "tensorflow/lite/c/common.h"
#include "micro_model_settings.h"

using Features = int8_t[kFeatureCount][kFeatureSize];

// Sets up any resources needed for the feature generation pipeline.
TfLiteStatus InitializeMicroFeatures();

// Converts audio sample data into a more compact form that's appropriate for
// feeding into a neural network.
// TfLiteStatus GenerateMicroFeatures(const int16_t* input, int input_size,
//                                    int output_size, int8_t* output,
//                                    size_t* num_samples_read);

TfLiteStatus GenerateFeatures(const int16_t* audio_data,
                              const size_t audio_data_size,
                              Features* features_output);

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_FEATURES_GENERATOR_H_
