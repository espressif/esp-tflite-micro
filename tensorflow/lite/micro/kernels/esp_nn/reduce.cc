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

// Espressif variant of the reduce kernel. Adds a fast path for int8 SUM over
// a contiguous run of axes (e.g. the common NHWC spatial reduction
// [1, H, W, C] -> [1, C]); everything else falls back to the shared helpers.

#include <algorithm>

#include "tensorflow/lite/micro/kernels/internal/reference/reduce.h"

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/mean.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace micro {

namespace {

void* InitReduce(TfLiteContext* context, const char* buffer, size_t length) {
  void* op_data =
      context->AllocatePersistentBuffer(context, sizeof(OpDataReduce));
  return new (op_data) OpDataReduce();
}

TfLiteStatus PrepareMinMax(TfLiteContext* context, TfLiteNode* node) {
  return PrepareMinMaxHelper(context, node,
                             static_cast<OpDataReduce*>(node->user_data));
}

TfLiteStatus PrepareMeanOrSum(TfLiteContext* context, TfLiteNode* node) {
  return PrepareMeanOrSumHelper(context, node,
                                static_cast<OpDataReduce*>(node->user_data));
}

TfLiteStatus PrepareAll(TfLiteContext* context, TfLiteNode* node) {
  return PrepareAllHelper(context, node,
                          static_cast<OpDataReduce*>(node->user_data));
}

// Reduces (SUM or MEAN) int8 data over a contiguous [outer x reduce x inner]
// layout and requantizes with the same integer arithmetic as
// reference_ops::QuantizedMeanOrSum, so results stay bit-exact.
bool ReduceInt8ContiguousFast(const TfLiteEvalTensor* input,
                              const TfLiteEvalTensor* axis,
                              TfLiteEvalTensor* output, OpDataReduce* op_data,
                              bool compute_sum) {
  constexpr int kMaxRank = 8;
  const int rank = input->dims->size;
  if (rank <= 0 || rank > kMaxRank) {
    return false;
  }
  const int* dims = input->dims->data;
  const int num_axis = static_cast<int>(ElementCount(*axis->dims));
  const int* axis_data = tflite::micro::GetTensorData<int>(axis);

  bool is_reduced[kMaxRank] = {};
  for (int i = 0; i < num_axis; ++i) {
    int a = axis_data[i];
    if (a < 0) {
      a += rank;
    }
    if (a < 0 || a >= rank) {
      return false;
    }
    is_reduced[a] = true;
  }

  // The reduced dimensions (ignoring size-1 dimensions) must form one
  // contiguous block so the input decomposes as [outer][reduce][inner].
  int first = -1, last = -1;
  for (int i = 0; i < rank; ++i) {
    if (is_reduced[i] && dims[i] != 1) {
      if (first < 0) first = i;
      last = i;
    }
  }
  if (first < 0) {
    return false;
  }
  for (int i = first; i <= last; ++i) {
    if (!is_reduced[i] && dims[i] != 1) {
      return false;
    }
  }

  int outer = 1, reduce_count = 1, inner = 1;
  for (int i = 0; i < first; ++i) outer *= dims[i];
  for (int i = first; i <= last; ++i) reduce_count *= dims[i];
  for (int i = last + 1; i < rank; ++i) inner *= dims[i];

  const int8_t* in = tflite::micro::GetTensorData<int8_t>(input);
  int8_t* out = tflite::micro::GetTensorData<int8_t>(output);
  int32_t mult = op_data->multiplier;
  int shift = op_data->shift;
  if (!compute_sum) {
    // Fold 1/reduce_count into the multiplier, exactly as
    // reference_ops::QuantizedMeanOrSum does for the mean.
    int adj = 63 - __builtin_clzll((uint64_t)reduce_count);
    adj = adj < 32 ? adj : 32;
    adj = adj < (31 + shift) ? adj : (31 + shift);
    if (adj < 0) {
      return false; /* degenerate shift; leave it to the reference path */
    }
    mult = (int32_t)(((int64_t)mult << adj) / reduce_count);
    shift = shift - adj;
  }
  const int32_t acc_init = -op_data->input_zp * reduce_count;
  const int32_t out_zp = op_data->output_zp;

  constexpr int kChunk = 128;
  int32_t acc[kChunk];
  for (int o = 0; o < outer; ++o) {
    const int8_t* in_o = in + o * reduce_count * inner;
    int8_t* out_o = out + o * inner;
    for (int c0 = 0; c0 < inner; c0 += kChunk) {
      const int n = std::min(kChunk, inner - c0);
      for (int i = 0; i < n; ++i) {
        acc[i] = acc_init;
      }
      const int8_t* p = in_o + c0;
      for (int r = 0; r < reduce_count; ++r) {
        for (int i = 0; i < n; ++i) {
          acc[i] += p[i];
        }
        p += inner;
      }
      for (int i = 0; i < n; ++i) {
        int32_t v =
            MultiplyByQuantizedMultiplier(acc[i], mult, shift) + out_zp;
        v = std::min(std::max(v, static_cast<int32_t>(-128)),
                     static_cast<int32_t>(127));
        out_o[c0 + i] = static_cast<int8_t>(v);
      }
    }
  }
  return true;
}

TfLiteStatus EvalMean(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data = static_cast<OpDataReduce*>(node->user_data);
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  if (input->type == kTfLiteInt8 && output->type == kTfLiteInt8 &&
      ReduceInt8ContiguousFast(input, axis, output, op_data,
                               /*compute_sum=*/false)) {
    return kTfLiteOk;
  }
  return EvalMeanHelper(context, node, op_data);
}

TfLiteStatus EvalMax(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data = static_cast<OpDataReduce*>(node->user_data);
  return EvalMaxHelper(context, node, op_data);
}

TfLiteStatus EvalMin(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data = static_cast<OpDataReduce*>(node->user_data);
  return EvalMinHelper(context, node, op_data);
}

TfLiteStatus EvalSum(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data = static_cast<OpDataReduce*>(node->user_data);
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  if (input->type == kTfLiteInt8 && output->type == kTfLiteInt8 &&
      ReduceInt8ContiguousFast(input, axis, output, op_data,
                               /*compute_sum=*/true)) {
    return kTfLiteOk;
  }
  return EvalSumHelper(context, node, op_data);
}

TfLiteStatus EvalAll(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data = static_cast<OpDataReduce*>(node->user_data);
  return EvalAllHelper(context, node, op_data);
}

}  // namespace

TFLMRegistration Register_MEAN() {
  return tflite::micro::RegisterOp(InitReduce, PrepareMeanOrSum, EvalMean);
}

TFLMRegistration Register_REDUCE_MAX() {
  return tflite::micro::RegisterOp(InitReduce, PrepareMinMax, EvalMax);
}

TFLMRegistration Register_REDUCE_MIN() {
  return tflite::micro::RegisterOp(InitReduce, PrepareMinMax, EvalMin);
}

TFLMRegistration Register_SUM() {
  return tflite::micro::RegisterOp(InitReduce, PrepareMeanOrSum, EvalSum);
}

TFLMRegistration Register_REDUCE_ALL() {
  return tflite::micro::RegisterOp(InitReduce, PrepareAll, EvalAll);
}

}  // namespace micro
}  // namespace tflite
