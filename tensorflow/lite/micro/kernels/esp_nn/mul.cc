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

#include "tensorflow/lite/micro/kernels/mul.h"

#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/micro/kernels/internal/reference/mul.h"
#include "tensorflow/lite/micro/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"

#if ESP_NN
#include <esp_nn.h>
#if defined(__has_include)
#if __has_include(<esp_nn_multicore.h>)
#include <esp_nn_multicore.h>
#define ESP_NN_HAVE_MULTICORE 1
#endif
#endif
#endif

#include <esp_timer.h>

long long mul_total_time = 0;

namespace tflite {
namespace micro {
#if ESP_NN
void MulEvalQuantized(TfLiteContext* context, TfLiteNode* node,
                      const OpDataMul* data, const TfLiteEvalTensor* input1,
                      const TfLiteEvalTensor* input2,
                      TfLiteEvalTensor* output) {
  tflite::micro::ArithmeticParams op_params = {};
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.float_activation_max = data->output_activation_max_f32;
  op_params.input1_offset = -data->input1_zero_point;
  op_params.input2_offset = -data->input2_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;

  bool need_broadcast = reference_ops::ProcessBroadcastShapes(
      tflite::micro::GetTensorShape(input1),
      tflite::micro::GetTensorShape(input2), &op_params);

  if (need_broadcast) {
    // Channel-vector broadcast (e.g. squeeze-excite gating):
    // [N,H,W,C] * [1,..,1,C]. esp-nn's channel-broadcast kernel does the
    // per-element arithmetic of BroadcastMul4DSlow (bit-exact). The rows are
    // split across both cores when available.
    const RuntimeShape sh1 = tflite::micro::GetTensorShape(input1);
    const RuntimeShape sh2 = tflite::micro::GetTensorShape(input2);
    const RuntimeShape sho = tflite::micro::GetTensorShape(output);
    const int8_t* d1 = tflite::micro::GetTensorData<int8_t>(input1);
    const int8_t* d2 = tflite::micro::GetTensorData<int8_t>(input2);
    // The fast path multiplies element i by vec[i % C]: only right when the
    // smaller operand runs along the last (channel) axis, not merely has C
    // elements (e.g. [1,8,1,1] against [1,8,8,8] is a per-row scale).
    const int out_ch = sho.Dims(sho.DimensionsCount() - 1);
    const bool vec_is_2 = (sh2.FlatSize() == out_ch) &&
                          (sh2.DimensionsCount() > 0) &&
                          (sh2.Dims(sh2.DimensionsCount() - 1) == out_ch) &&
                          (sh1.FlatSize() == sho.FlatSize());
    const bool vec_is_1 = (sh1.FlatSize() == out_ch) &&
                          (sh1.DimensionsCount() > 0) &&
                          (sh1.Dims(sh1.DimensionsCount() - 1) == out_ch) &&
                          (sh2.FlatSize() == sho.FlatSize());
    if (vec_is_2 || vec_is_1) {
      const int8_t* big = vec_is_2 ? d1 : d2;
      const int8_t* vec = vec_is_2 ? d2 : d1;
      const int32_t big_off = vec_is_2 ? op_params.input1_offset : op_params.input2_offset;
      const int32_t vec_off = vec_is_2 ? op_params.input2_offset : op_params.input1_offset;
      const int ch = sho.Dims(sho.DimensionsCount() - 1);
      const int positions = sho.FlatSize() / ch;
      int8_t* out_data = tflite::micro::GetTensorData<int8_t>(output);
      struct BcastMulJob {
        const int8_t *big, *vec;
        int8_t* out;
        int positions, ch;
        int32_t big_off, vec_off, out_off, out_mult, out_shift, act_min, act_max;
        static void Run(void* p) {
          const BcastMulJob* j = static_cast<const BcastMulJob*>(p);
          esp_nn_mul_broadcast_channel_s8(j->big, j->vec, j->big_off, j->vec_off,
                                          j->out, j->out_off, j->out_mult,
                                          j->out_shift, j->act_min, j->act_max,
                                          j->positions, j->ch);
        }
      };
      BcastMulJob job = {big, vec, out_data, positions, ch,
                         big_off, vec_off, op_params.output_offset,
                         op_params.output_multiplier, op_params.output_shift,
                         op_params.quantized_activation_min,
                         op_params.quantized_activation_max};
#ifdef ESP_NN_HAVE_MULTICORE
      if (esp_nn_dual_core_active() && positions * ch >= 8192) {
        const int p0 = positions / 2;
        BcastMulJob wjob = job;
        wjob.positions = p0;
        if (esp_nn_dual_core_run(BcastMulJob::Run, &wjob)) {
          BcastMulJob rest = job;
          rest.big += p0 * ch;
          rest.out += p0 * ch;
          rest.positions = positions - p0;
          BcastMulJob::Run(&rest);
          esp_nn_dual_core_wait();
          return;
        }
      }
#endif  // ESP_NN_HAVE_MULTICORE
      BcastMulJob::Run(&job);
      return;
    }
    reference_integer_ops::BroadcastMul4DSlow(
        op_params, tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<int8_t>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<int8_t>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  } else {
    const int8_t *input1_data = tflite::micro::GetTensorData<int8_t>(input1);
    const int8_t *input2_data = tflite::micro::GetTensorData<int8_t>(input2);
    int8_t *out_data = tflite::micro::GetTensorData<int8_t>(output);

    esp_nn_mul_elementwise_s8(input1_data, input2_data, op_params.input1_offset,
                              op_params.input2_offset, out_data, op_params.output_offset,
                              op_params.output_multiplier, op_params.output_shift,
                              op_params.quantized_activation_min, op_params.quantized_activation_max,
                              MatchingElementsSize(tflite::micro::GetTensorShape(input1),
                                                    tflite::micro::GetTensorShape(input2),
                                                    tflite::micro::GetTensorShape(output)));
  }
}
#endif

TfLiteStatus MulEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLiteMulParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataMul* data = static_cast<const OpDataMul*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kMulInput1Tensor);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kMulInput2Tensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kMulOutputTensor);

  long long start_time = esp_timer_get_time();
  switch (input1->type) {
    case kTfLiteInt8:
#if ESP_NN
      MulEvalQuantized(context, node, data, input1, input2, output);
#else
      EvalMulQuantizedReference(context, node, data, input1, input2, output);
#endif
      break;
    case kTfLiteInt16:
    case kTfLiteInt32:
      EvalMulQuantizedReference(context, node, data, input1, input2, output);
      break;
    case kTfLiteFloat32:
      EvalMulFloatReference(context, node, params, data, input1, input2,
                            output);
      break;
    default:
      MicroPrintf("Type %s (%d) not supported.",
                  TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  mul_total_time += esp_timer_get_time() - start_time;
  return kTfLiteOk;
}

TFLMRegistration Register_MUL() {
  return tflite::micro::RegisterOp(MulInit, MulPrepare, MulEval);
}

}  // namespace micro
}  // namespace tflite
