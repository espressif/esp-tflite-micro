/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// Espressif variant of the quantize kernel: the int8 -> int8 requantize case
// is a pure per-byte map with per-layer constants, so it goes through a
// 256-entry lookup table built with the untouched reference (bit-exact by
// construction), split across both cores. Everything else falls through to
// the shared reference eval.

#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/micro/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/quantize.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

#if ESP_NN
#if defined(__has_include)
#if __has_include(<esp_nn_multicore.h>)
#include <esp_nn_multicore.h>
#define ESP_NN_HAVE_MULTICORE 1
#endif
#endif
#endif

namespace tflite {
namespace micro {
namespace {

#if ESP_NN
struct QuantizeLutData {
  OpDataQuantizeReference base;  // must stay first
  int8_t lut[256];
  bool lut_valid;
};
#endif

void* InitQuantize(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
#if ESP_NN
  return context->AllocatePersistentBuffer(context, sizeof(QuantizeLutData));
#else
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataQuantizeReference));
#endif
}

#if ESP_NN
TfLiteStatus PrepareQuantizeLut(TfLiteContext* context, TfLiteNode* node) {
  TfLiteStatus status = PrepareQuantizeReference(context, node);
  if (status != kTfLiteOk) {
    return status;
  }
  QuantizeLutData* d = static_cast<QuantizeLutData*>(node->user_data);
  d->lut_valid = false;
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  if (input != nullptr && output != nullptr &&
      input->type == kTfLiteInt8 && output->type == kTfLiteInt8) {
    int8_t lut_in[256];
    for (int i = 0; i < 256; ++i) {
      lut_in[i] = (int8_t)(i - 128);
    }
    reference_ops::Requantize(lut_in, 256,
                              d->base.requantize_output_multiplier,
                              d->base.requantize_output_shift,
                              d->base.input_zero_point,
                              d->base.quantization_params.zero_point, d->lut);
    d->lut_valid = true;
  }
  if (input != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(input);
  }
  if (output != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(output);
  }
  return status;
}
#endif

#if ESP_NN
struct QuantLutJob {
  const int8_t* lut;
  const int8_t* in;
  int8_t* out;
  int32_t count;
  static void Run(void* p) {
    const QuantLutJob* j = static_cast<const QuantLutJob*>(p);
    const int8_t* lut128 = j->lut + 128;
    const int8_t* in = j->in;
    int8_t* out = j->out;
    int32_t n = j->count;
    for (; n >= 8; n -= 8, in += 8, out += 8) {
      out[0] = lut128[in[0]];
      out[1] = lut128[in[1]];
      out[2] = lut128[in[2]];
      out[3] = lut128[in[3]];
      out[4] = lut128[in[4]];
      out[5] = lut128[in[5]];
      out[6] = lut128[in[6]];
      out[7] = lut128[in[7]];
    }
    for (; n > 0; --n) {
      *out++ = lut128[*in++];
    }
  }
};
#endif

TfLiteStatus EvalQuantize(TfLiteContext* context, TfLiteNode* node) {
#if ESP_NN
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  if (input->type == kTfLiteInt8 && output->type == kTfLiteInt8) {
    TFLITE_DCHECK(node->user_data != nullptr);
    const QuantizeLutData* d =
        static_cast<const QuantizeLutData*>(node->user_data);
    if (d->lut_valid) {
      const int32_t count = (int32_t)ElementCount(*input->dims);
      const int8_t* in_data = tflite::micro::GetTensorData<int8_t>(input);
      int8_t* out_data = tflite::micro::GetTensorData<int8_t>(output);
#ifdef ESP_NN_HAVE_MULTICORE
      if (esp_nn_dual_core_active() && count >= 32768) {
        const int32_t half = count / 2;
        QuantLutJob wjob = {d->lut, in_data, out_data, half};
        if (esp_nn_dual_core_run(QuantLutJob::Run, &wjob)) {
          QuantLutJob rest = {d->lut, in_data + half, out_data + half,
                              count - half};
          QuantLutJob::Run(&rest);
          esp_nn_dual_core_wait();
          return kTfLiteOk;
        }
      }
#endif  // ESP_NN_HAVE_MULTICORE
      QuantLutJob job = {d->lut, in_data, out_data, count};
      QuantLutJob::Run(&job);
      return kTfLiteOk;
    }
  }
#endif
  return EvalQuantizeReference(context, node);
}

}  // namespace

TFLMRegistration Register_QUANTIZE() {
#if ESP_NN
  return tflite::micro::RegisterOp(InitQuantize, PrepareQuantizeLut,
                                   EvalQuantize);
#else
  return tflite::micro::RegisterOp(InitQuantize, PrepareQuantizeReference,
                                   EvalQuantize);
#endif
}

}  // namespace micro
}  // namespace tflite
