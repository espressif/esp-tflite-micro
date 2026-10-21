// Espressif variant: int8 logistic through a reference-built 256-entry
// lookup table, dual-core split. Same arithmetic as upstream (bit-exact).
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

#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/logistic.h"

#if ESP_NN
#if defined(__has_include)
#if __has_include(<esp_nn_multicore.h>)
#include <esp_nn_multicore.h>
#define ESP_NN_HAVE_MULTICORE 1
#endif
#endif
#endif

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/logistic.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

#if ESP_NN
struct OpDataLogisticLut {
  OpDataLogistic base;  // must stay first: shared Prepare writes through it
  int8_t lut[256];
  bool lut_valid;
};
#endif

void* LogisticInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
#if ESP_NN
  return context->AllocatePersistentBuffer(context, sizeof(OpDataLogisticLut));
#else
  return context->AllocatePersistentBuffer(context, sizeof(OpDataLogistic));
#endif
}

TfLiteStatus LogisticEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kLogisticInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kLogisticOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpDataLogistic* data = static_cast<OpDataLogistic*>(node->user_data);

  if (input->type == kTfLiteFloat32) {
    switch (output->type) {
      case kTfLiteFloat32: {
        reference_ops::Logistic(tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<float>(input),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
        return kTfLiteOk;
      }
      default:
        MicroPrintf("Input %s, output %s not supported.",
                    TfLiteTypeGetName(input->type),
                    TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else if (input->type == kTfLiteInt16) {
    switch (output->type) {
      case kTfLiteInt16: {
        reference_integer_ops::Logistic(
            data->input_multiplier, data->input_left_shift,
            NumElements(input->dims),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorData<int16_t>(output));
        return kTfLiteOk;
      }
      default:
        MicroPrintf("Input %s, output %s not supported.",
                    TfLiteTypeGetName(input->type),
                    TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else if (input->type == kTfLiteInt8) {
    switch (output->type) {
      case kTfLiteInt8: {
#if ESP_NN
        // Persistent 256-entry LUT built in Prepare from the untouched
        // reference (bit-exact); the hot loop is one byte lookup per element,
        // split across both cores only when the tensor is large enough for
        // the dispatch overhead to vanish.
        const OpDataLogisticLut* d =
            static_cast<const OpDataLogisticLut*>(node->user_data);
        const int32_t count = NumElements(input->dims);
        if (d->lut_valid) {
          const int8_t* lut = d->lut;
          const int8_t* in_data = tflite::micro::GetTensorData<int8_t>(input);
          int8_t* out_data = tflite::micro::GetTensorData<int8_t>(output);
          struct LutJob {
            const int8_t* lut;
            const int8_t* in;
            int8_t* out;
            int32_t count;
            static void Run(void* p) {
              const LutJob* j = static_cast<const LutJob*>(p);
              // Bias the table pointer so the index is the raw signed byte
              // (in + 128 disappears), and unroll: the lookups are
              // independent, letting loads pipeline.
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
#ifdef ESP_NN_HAVE_MULTICORE
          if (esp_nn_dual_core_active() && count >= 32768) {
            const int32_t half = count / 2;
            LutJob wjob = {lut, in_data, out_data, half};
            if (esp_nn_dual_core_run(LutJob::Run, &wjob)) {
              LutJob rest = {lut, in_data + half, out_data + half,
                             count - half};
              LutJob::Run(&rest);
              esp_nn_dual_core_wait();
              return kTfLiteOk;
            }
          }
#endif  // ESP_NN_HAVE_MULTICORE
          LutJob job = {lut, in_data, out_data, count};
          LutJob::Run(&job);
          return kTfLiteOk;
        }
#endif
        reference_integer_ops::Logistic(
            data->input_zero_point, data->input_range_radius,
            data->input_multiplier, data->input_left_shift,
            NumElements(input->dims),
            tflite::micro::GetTensorData<int8_t>(input),
            tflite::micro::GetTensorData<int8_t>(output));
        return kTfLiteOk;
      }
      default:
        MicroPrintf("Input %s, output %s not supported.",
                    TfLiteTypeGetName(input->type),
                    TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else {
    // TODO(b/141211002): Also support other data types once we have supported
    // temporary tensors in TFLM.
    MicroPrintf("Input %s, output %s not supported.",
                TfLiteTypeGetName(input->type),
                TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus LogisticPrepareLut(TfLiteContext* context, TfLiteNode* node) {
  TfLiteStatus status = LogisticPrepare(context, node);
#if ESP_NN
  if (status != kTfLiteOk) {
    return status;
  }
  OpDataLogisticLut* d = static_cast<OpDataLogisticLut*>(node->user_data);
  d->lut_valid = false;
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  if (input != nullptr && input->type == kTfLiteInt8) {
    int8_t lut_in[256];
    for (int i = 0; i < 256; ++i) {
      lut_in[i] = (int8_t)(i - 128);
    }
    reference_integer_ops::Logistic(d->base.input_zero_point,
                                    d->base.input_range_radius,
                                    d->base.input_multiplier,
                                    d->base.input_left_shift, 256, lut_in,
                                    d->lut);
    d->lut_valid = true;
  }
  if (input != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(input);
  }
#endif
  return status;
}

}  // namespace

TFLMRegistration Register_LOGISTIC() {
  return tflite::micro::RegisterOp(LogisticInit, LogisticPrepareLut,
                                   LogisticEval);
}
}  // namespace micro
}  // namespace tflite
