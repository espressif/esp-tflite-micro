/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>
#include <cstdint>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  int8_t* lut;
};

inline float GeluFloat(float x, bool approximate) {
  if (approximate) {
    // tanh approximation used by the DeiT export.
    const float inner =
        0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
  }
  return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f));
}

void* GeluInit(TfLiteContext* context, const char* buffer, size_t length) {
  (void)buffer;
  (void)length;
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus GeluPrepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = static_cast<OpData*>(node->user_data);
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt8);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt8);

  const auto* params =
      static_cast<const TfLiteGeluParams*>(node->builtin_data);
  const bool approximate = params != nullptr && params->approximate;

  data->lut = static_cast<int8_t*>(
      context->AllocatePersistentBuffer(context, 256));
  TF_LITE_ENSURE(context, data->lut != nullptr);
  const float input_scale = input->params.scale;
  const int input_zero_point = input->params.zero_point;
  const float output_scale = output->params.scale;
  const int output_zero_point = output->params.zero_point;
  for (int q = -128; q <= 127; ++q) {
    const float x = static_cast<float>(q - input_zero_point) * input_scale;
    const float y = GeluFloat(x, approximate);
    int32_t quantized = static_cast<int32_t>(
                            TfLiteRound(y / output_scale)) +
                        output_zero_point;
    if (quantized < -128) {
      quantized = -128;
    }
    if (quantized > 127) {
      quantized = 127;
    }
    data->lut[static_cast<uint8_t>(q)] = static_cast<int8_t>(quantized);
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus GeluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  const OpData* data = static_cast<const OpData*>(node->user_data);
  const int8_t* in = tflite::micro::GetTensorData<int8_t>(input);
  int8_t* out = tflite::micro::GetTensorData<int8_t>(output);
  const int n = tflite::micro::GetTensorShape(input).FlatSize();
  const int8_t* lut = data->lut;
  for (int i = 0; i < n; ++i) {
    out[i] = lut[static_cast<uint8_t>(in[i])];
  }
  (void)output;
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_GELU() {
  return tflite::micro::RegisterOp(GeluInit, GeluPrepare, GeluEval);
}

}  // namespace micro
}  // namespace tflite
