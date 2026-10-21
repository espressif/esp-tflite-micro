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

// Espressif variant of the hard-swish kernel. The per-layer constants make
// hard-swish a pure int8 -> int8 map: the kernel evaluates the untouched
// TFLM reference for all 256 inputs into a lookup table (bit-exact by
// construction) and streams the tensor through it, optionally split across
// both cores through the ESP-NN dual-core worker.

#include "tensorflow/lite/micro/kernels/internal/reference/hard_swish.h"

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/hard_swish.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
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
struct HardSwishLutData {
  HardSwishParams params;  // must stay first: shared Prepare writes through it
  int8_t lut[256];
  bool lut_valid;
};
#endif

void* HardSwishInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
#if ESP_NN
  return context->AllocatePersistentBuffer(context, sizeof(HardSwishLutData));
#else
  return context->AllocatePersistentBuffer(context, sizeof(HardSwishParams));
#endif
}

#if ESP_NN
TfLiteStatus HardSwishPrepareLut(TfLiteContext* context, TfLiteNode* node) {
  TfLiteStatus status = tflite::micro::HardSwishPrepare(context, node);
  if (status != kTfLiteOk) {
    return status;
  }
  HardSwishLutData* d = static_cast<HardSwishLutData*>(node->user_data);
  d->lut_valid = false;
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  if (input != nullptr && input->type == kTfLiteInt8) {
    int8_t lut_in[256];
    for (int i = 0; i < 256; ++i) {
      lut_in[i] = (int8_t)(i - 128);
    }
    const RuntimeShape lut_shape(1, 256);
    tflite::micro::reference_ops::HardSwish<int8_t>(d->params, lut_shape, lut_in,
                                             lut_shape, d->lut);
    d->lut_valid = true;
  }
  if (input != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(input);
  }
  return status;
}
#endif

#if ESP_NN
// Hard-swish is an elementwise int8 -> int8 map with per-layer constants, so
// there are only 256 possible results. Build the table by evaluating the
// untouched TFLM reference on all 256 inputs (bit-exact by construction);
// the hot loop is then one byte lookup per element.
struct HardSwishLutJob {
  const int8_t* lut;  // indexed by input byte + 128
  const int8_t* input;
  int8_t* output;
  int count;
};

void HardSwishLutWorker(void* p) {
  const HardSwishLutJob* j = static_cast<const HardSwishLutJob*>(p);
  const int8_t* lut128 = j->lut + 128;
  const int8_t* in = j->input;
  int8_t* out = j->output;
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
#endif

TfLiteStatus HardSwishEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kHardSwishInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kHardSwishOutputTensor);
#if !ESP_NN
  /* the ESP_NN int8 path reads params through HardSwishLutData instead */
  HardSwishParams* params = static_cast<HardSwishParams*>(node->user_data);
#endif

  switch (input->type) {
    case kTfLiteFloat32: {
      tflite::micro::reference_ops::HardSwish<float>(
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
    } break;
    case kTfLiteInt8: {
#if ESP_NN
      const HardSwishLutData* d =
          static_cast<const HardSwishLutData*>(node->user_data);
      const int flat_size = tflite::micro::GetTensorShape(input).FlatSize();
      const int8_t* in_data = tflite::micro::GetTensorData<int8_t>(input);
      int8_t* out_data = tflite::micro::GetTensorData<int8_t>(output);
      if (d->lut_valid) {
#ifdef ESP_NN_HAVE_MULTICORE
        if (esp_nn_dual_core_active() && flat_size >= 32768) {
          const int half = flat_size / 2;
          HardSwishLutJob job = {d->lut, in_data, out_data, half};
          if (esp_nn_dual_core_run(HardSwishLutWorker, &job)) {
            HardSwishLutJob rest = {d->lut, in_data + half, out_data + half,
                                    flat_size - half};
            HardSwishLutWorker(&rest);
            esp_nn_dual_core_wait();
            break;
          }
        }
#endif  // ESP_NN_HAVE_MULTICORE
        HardSwishLutJob job = {d->lut, in_data, out_data, flat_size};
        HardSwishLutWorker(&job);
        break;
      }
      const RuntimeShape flat_shape(1, flat_size);
      tflite::micro::reference_ops::HardSwish<int8_t>(d->params, flat_shape, in_data,
                                               flat_shape, out_data);
#else
      tflite::micro::reference_ops::HardSwish<int8_t>(
          *params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
#endif
    } break;
    default: {
      MicroPrintf("Unsupported type %s", TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_HARD_SWISH() {
#if ESP_NN
  return tflite::micro::RegisterOp(HardSwishInit, HardSwishPrepareLut,
                                   HardSwishEval);
#else
  return tflite::micro::RegisterOp(HardSwishInit, tflite::micro::HardSwishPrepare,
                                   HardSwishEval);
#endif
}

}  // namespace micro
}  // namespace tflite
