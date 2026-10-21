/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include <string.h>

#include "tensorflow/lite/micro/kernels/internal/reference/transpose.h"

#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/transpose.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {


#if ESP_NN
// Espressif fast path: generic strided 4D int8 transpose with the two
// innermost output dimensions specialised into tight loops (the reference
// walks every element through a recursive index computation). Pure data
// movement - values are identical by construction.
static bool TransposeInt8Fast4D(const TransposeParams& params,
                                const RuntimeShape& in_shape,
                                const int8_t* in,
                                const RuntimeShape& out_shape, int8_t* out) {
  if (in_shape.DimensionsCount() != 4 || params.perm_count != 4) {
    return false;
  }
  // Input strides (elements), then per-output-dim source strides.
  int32_t in_stride[4];
  in_stride[3] = 1;
  for (int d = 2; d >= 0; d--) {
    in_stride[d] = in_stride[d + 1] * in_shape.Dims(d + 1);
  }
  int32_t src_stride[4], od[4];
  for (int d = 0; d < 4; d++) {
    src_stride[d] = in_stride[params.perm[d]];
    od[d] = out_shape.Dims(d);
  }
  int8_t* dst = out;
  for (int32_t a = 0; a < od[0]; a++) {
    const int8_t* pa = in + (size_t)a * src_stride[0];
    for (int32_t b = 0; b < od[1]; b++) {
      const int8_t* pb = pa + (size_t)b * src_stride[1];
      if (src_stride[3] == 1) {
        // Innermost source contiguous: row memcpy.
        for (int32_t c = 0; c < od[2]; c++) {
          memcpy(dst, pb + (size_t)c * src_stride[2], od[3]);
          dst += od[3];
        }
      } else {
        for (int32_t c = 0; c < od[2]; c++) {
          const int8_t* pc = pb + (size_t)c * src_stride[2];
          int32_t e = 0;
          const int32_t s3 = src_stride[3];
          for (; e <= od[3] - 4; e += 4) {
            dst[e] = pc[(size_t)e * s3];
            dst[e + 1] = pc[(size_t)(e + 1) * s3];
            dst[e + 2] = pc[(size_t)(e + 2) * s3];
            dst[e + 3] = pc[(size_t)(e + 3) * s3];
          }
          for (; e < od[3]; e++) {
            dst[e] = pc[(size_t)e * s3];
          }
          dst += od[3];
        }
      }
    }
  }
  return true;
}
#endif  // ESP_NN


TfLiteStatus TransposeEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* perm_tensor =
      tflite::micro::GetEvalInput(context, node, kTransposePermTensor);
  const int32_t* perm_data = perm_tensor->data.i32;
  const int size = perm_tensor->dims->data[0];
  TransposeParams params;
  params.perm_count = size;
  for (int i = 0; i < size; ++i) {
    params.perm[i] = perm_data[i];
  }

  // Transpose kernel only does rearranging values not numeric evaluations
  // on each cell. It's safe to implement per size of scalar type and this
  // trick keeps the total code size in a reasonable range.
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kTransposeInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kTransposeOutputTensor);
  switch (input->type) {
    case kTfLiteFloat32:
      reference_ops::Transpose(params, tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<float>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
#if ESP_NN
      if (!TransposeInt8Fast4D(params, tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<int8_t>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<int8_t>(output)))
#endif  // ESP_NN
      {
        reference_ops::Transpose(params, tflite::micro::GetTensorShape(input),
                                 tflite::micro::GetTensorData<int8_t>(input),
                                 tflite::micro::GetTensorShape(output),
                                 tflite::micro::GetTensorData<int8_t>(output));
      }
      break;
    case kTfLiteInt16:
      reference_ops::Transpose(params, tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<int16_t>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<int16_t>(output));
      break;
    default:
      MicroPrintf(
          "Type %s is currently not supported by Transpose. "
          "Only float32, int8 and int16 are supported",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_TRANSPOSE() {
  return tflite::micro::RegisterOp(nullptr, TransposePrepare, TransposeEval);
}
}  // namespace micro
}  // namespace tflite
