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
#include "tensorflow/lite/micro/kernels/internal/reference/pad.h"

#include <string.h>

#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pad.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {


#if ESP_NN
// Espressif fast path: 4D pads that only touch the spatial dims reduce to
// per-row memset/memcpy. Values are identical to the reference element walk.
static bool PadSpatialFast(const tflite::micro::PadParams& p,
                           const tflite::micro::RuntimeShape& in_shape,
                           const int8_t* in, int8_t pad_value,
                           const tflite::micro::RuntimeShape& out_shape, int8_t* out) {
  if (in_shape.DimensionsCount() != 4 || p.left_padding_count != 4 ||
      p.right_padding_count != 4) {
    return false;
  }
  if (p.left_padding[0] != 0 || p.right_padding[0] != 0 ||
      p.left_padding[3] != 0 || p.right_padding[3] != 0) {
    return false;
  }
  const int32_t B = in_shape.Dims(0), H = in_shape.Dims(1),
                W = in_shape.Dims(2), C = in_shape.Dims(3);
  const int32_t OH = out_shape.Dims(1), OW = out_shape.Dims(2);
  const int32_t pt = p.left_padding[1], pl = p.left_padding[2];
  const int32_t pb = p.right_padding[1];
  const size_t in_row = (size_t)W * C;
  const size_t out_row = (size_t)OW * C;
  for (int32_t b = 0; b < B; b++) {
    int8_t* ob = out + (size_t)b * OH * out_row;
    const int8_t* ib = in + (size_t)b * H * in_row;
    memset(ob, pad_value, (size_t)pt * out_row);
    for (int32_t h = 0; h < H; h++) {
      int8_t* dst = ob + (size_t)(pt + h) * out_row;
      memset(dst, pad_value, (size_t)pl * C);
      memcpy(dst + (size_t)pl * C, ib + (size_t)h * in_row, in_row);
      memset(dst + (size_t)(pl + W) * C, pad_value,
             out_row - (size_t)(pl + W) * C);
    }
    memset(ob + (size_t)(pt + H) * out_row, pad_value, (size_t)pb * out_row);
  }
  return true;
}
#endif  // ESP_NN


TfLiteStatus PadEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, /*index=*/0);
  const TfLiteEvalTensor* constant_values =
      NumInputs(node) == 3
          ? tflite::micro::GetEvalInput(context, node, /*index=*/2)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, /*index=*/0);

  switch (input->type) {
    case kTfLiteFloat32: {
      float pad_value =
          constant_values == nullptr
              ? 0.f
              : *tflite::micro::GetTensorData<float>(constant_values);
      if (data->params.resizing_category == ResizingCategory::kImageStyle) {
        reference_ops::PadImageStyle(
            data->params, tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<float>(input), &pad_value,
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<float>(output));
      } else {
        reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<float>(input),
                           &pad_value, tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<float>(output));
      }
    } break;
    case kTfLiteInt8: {
      int8_t pad_value;
      if (constant_values == nullptr) {
        pad_value = static_cast<uint8_t>(data->output_zero_point);
      } else {
        pad_value = *tflite::micro::GetTensorData<int8_t>(constant_values);
      }
#if ESP_NN
      if (PadSpatialFast(data->params, tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<int8_t>(input),
                         pad_value, tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<int8_t>(output))) {
      } else if (data->params.resizing_category == ResizingCategory::kImageStyle) {
#else
      if (data->params.resizing_category == ResizingCategory::kImageStyle) {
#endif  // ESP_NN
        reference_ops::PadImageStyle(
            data->params, tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int8_t>(input), &pad_value,
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int8_t>(output));
      } else {
        reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<int8_t>(input),
                           &pad_value, tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<int8_t>(output));
      }
    } break;
    case kTfLiteInt16: {
      int16_t pad_value =
          constant_values == nullptr
              ? 0
              : *tflite::micro::GetTensorData<int16_t>(constant_values);
      reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<int16_t>(input),
                         &pad_value, tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<int16_t>(output));
    } break;
    case kTfLiteInt32: {
      int32_t pad_value =
          constant_values == nullptr
              ? 0
              : *tflite::micro::GetTensorData<int32_t>(constant_values);
      reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<int32_t>(input),
                         &pad_value, tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<int32_t>(output));
    } break;
    default:

      MicroPrintf("Type %s not currently supported by Pad.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_PAD() {
  return tflite::micro::RegisterOp(PadInit, PadPrepare, PadEval);
}

// Also register Pad as PadV2.
TFLMRegistration Register_PADV2() {
  return tflite::micro::RegisterOp(PadInit, PadPrepare, PadEval);
}

}  // namespace micro
}  // namespace tflite
