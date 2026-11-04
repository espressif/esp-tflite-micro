/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/fully_connected.h"

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/micro/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#if ESP_NN
#include <esp_nn.h>
#include "sdkconfig.h"
#endif

#include <esp_timer.h>

// Wall time inside every fully connected eval, and the subset whose shape is
// the DeiT-Tiny FFN down-projection (row 768, out 192). The application
// zeroes these around the invoke it wants to report.
long long fc_total_time = 0;
long long fc_ffn_down_time = 0;
int fc_ffn_down_calls = 0;
int fc_batch_calls = 0;
int fc_row_calls = 0;

// Strong override in the fast firmware returns 1. The slow firmware leaves
// this weak definition, which keeps the per-row PIE kernel.
extern "C" int __attribute__((weak)) esp_tflm_fc_use_batch(void) { return 0; }

namespace tflite {
namespace micro {
namespace {

void* FullyConnectedInit(TfLiteContext* context, const char* buffer,
                         size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataFullyConnected));
}

TfLiteStatus FullyConnectedPrepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* data = static_cast<OpDataFullyConnected*>(node->user_data);
  data->fc_batch_scratch_index = -1;
  data->fc_batch_scratch_bytes = 0;
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if ((input->type == kTfLiteFloat32 && filter->type != kTfLiteFloat32) ||
      (input->type == kTfLiteInt8 &&
       (filter->type != kTfLiteInt8 && filter->type != kTfLiteInt4)) ||
      (input->type == kTfLiteInt16 && filter->type != kTfLiteInt8)) {
    MicroPrintf("Input type: %s with filter type : %s not supported.",
                TfLiteTypeGetName(input->type),
                TfLiteTypeGetName(filter->type));
    return kTfLiteError;
  }

  if (filter->type == kTfLiteInt4) {
    int filter_size =
        RuntimeShape(filter->dims->size,
                     reinterpret_cast<const int32_t*>(filter->dims->data))
            .FlatSize();
    context->RequestScratchBufferInArena(context, filter_size,
                                         &data->filter_buffer_index);
  }

  TF_LITE_ENSURE_OK(context, CalculateOpDataFullyConnected(
                                 context, params->activation, input->type,
                                 input, filter, bias, output, data));

#if defined(ESP_NN) && defined(ESP_NN_HAS_FC_S8_BATCH) && \
    defined(ESP_NN_HAS_FC_SCRATCH_BUF)
  // The batched kernel evaluates every row as one 1x1 convolution. That path
  // needs a fully connected scratch buffer registered immediately before the
  // call. Request it here so the arena planner can reuse it across nodes.
  if (esp_tflm_fc_use_batch() != 0 && data->is_per_channel &&
      input->type == kTfLiteInt8 && filter->type == kTfLiteInt8 &&
      output->dims->size >= 1 && filter->dims->size >= 2) {
    int32_t batches = 1;
    for (int i = 0; i < output->dims->size - 1; ++i) {
      batches *= output->dims->data[i];
    }
    const int32_t out_depth = output->dims->data[output->dims->size - 1];
    const int32_t accum = filter->dims->data[filter->dims->size - 1];
    if (batches > 0 && batches <= 65535 && accum > 0 && accum <= 65535 &&
        out_depth > 0 && out_depth <= 65535) {
      const int need = esp_nn_get_fc_batch_scratch_size(
          batches, static_cast<uint16_t>(accum),
          static_cast<uint16_t>(out_depth));
      if (need > 0) {
        TF_LITE_ENSURE_OK(
            context, context->RequestScratchBufferInArena(
                         context, need + 16, &data->fc_batch_scratch_index));
        data->fc_batch_scratch_bytes = need;
      }
    }
  }
#endif

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus FullyConnectedEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);

  const auto& data =
      *(static_cast<const OpDataFullyConnected*>(node->user_data));

  long long start_time = esp_timer_get_time();
  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
    case kTfLiteFloat32: {
      tflite::micro::reference_ops::FullyConnected(
          FullyConnectedParamsFloat(params->activation),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    }

    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
          int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
          tflite::micro::tensor_utils::UnpackDenseInt4IntoInt8(
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(filter).FlatSize(),
              unpacked_filter_data);
          tflite::micro::reference_integer_ops::FullyConnected(
              FullyConnectedParamsQuantized(data),
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter), unpacked_filter_data,
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        case kTfLiteInt8: {
          if (data.is_per_channel) {
#if ESP_NN
            const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
            const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);

            TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
            TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
            const int filter_dim_count = filter_shape.DimensionsCount();
            const int output_dim_count = output_shape.DimensionsCount();
            const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
            const int output_depth = output_shape.Dims(output_dim_count - 1);
            TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
            const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

            const int32_t* bias_data =
                tflite::micro::GetOptionalTensorData<int32_t>(bias);

            const int8_t *input_data = tflite::micro::GetTensorData<int8_t>(input);
            int8_t *output_data = tflite::micro::GetTensorData<int8_t>(output);
            const int8_t *filter_data = tflite::micro::GetTensorData<int8_t>(filter);

            const int64_t fc_t0 = esp_timer_get_time();
#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
            // P4 fast path: a per-channel batch with weight zero-point 0 is a
            // 1x1 convolution when a fully connected scratch buffer is
            // registered. The row loop below is the kernel that stays in
            // upstream esp-nn (and in this tree when the app does not ask
            // for the batch).
            const bool batch_ok =
                data.fc_batch_scratch_index >= 0 &&
                data.filter_zero_point == 0 &&
                ((reinterpret_cast<uintptr_t>(input_data) & 7u) == 0) &&
                ((reinterpret_cast<uintptr_t>(filter_data) & 7u) == 0);
            if (batch_ok) {
              uint8_t* raw = static_cast<uint8_t*>(context->GetScratchBuffer(
                  context, data.fc_batch_scratch_index));
              const uintptr_t aligned =
                  (reinterpret_cast<uintptr_t>(raw) + 15u) & ~uintptr_t{15};
              esp_nn_set_fc_scratch_buf(reinterpret_cast<void*>(aligned),
                                        static_cast<size_t>(data.fc_batch_scratch_bytes));
              esp_nn_fully_connected_per_ch_s8_batch(
                  input_data, -data.input_zero_point, accum_depth, filter_data,
                  -data.filter_zero_point, bias_data, output_data, output_depth,
                  data.output_zero_point, data.per_channel_output_shift,
                  data.per_channel_output_multiplier, data.output_activation_min,
                  data.output_activation_max, batches);
              fc_batch_calls += 1;
            } else
#endif
            {
              for (int b = 0; b < batches; ++b) {
                esp_nn_fully_connected_per_ch_s8(input_data, -data.input_zero_point,
                                          accum_depth,
                                          filter_data, -data.filter_zero_point,
                                          bias_data, output_data, output_depth,
                                          data.output_zero_point,
                                          data.per_channel_output_shift, data.per_channel_output_multiplier,
                                          data.output_activation_min,
                                          data.output_activation_max);
                input_data += accum_depth;
                output_data += output_depth;
              }
              fc_row_calls += 1;
            }
            const int64_t fc_dt = esp_timer_get_time() - fc_t0;
            if (accum_depth == 768 && output_depth == 192) {
              fc_ffn_down_time += fc_dt;
              fc_ffn_down_calls += 1;
            }
#else
            tflite::micro::reference_integer_ops::FullyConnectedPerChannel(
                FullyConnectedParamsQuantized(data),
                data.per_channel_output_multiplier,
                reinterpret_cast<const int*>(data.per_channel_output_shift),
                tflite::micro::GetTensorShape(input),
                tflite::micro::GetTensorData<int8_t>(input),
                tflite::micro::GetTensorShape(filter),
                tflite::micro::GetTensorData<int8_t>(filter),
                tflite::micro::GetTensorShape(bias),
                tflite::micro::GetOptionalTensorData<int32_t>(bias),
                tflite::micro::GetTensorShape(output),
                tflite::micro::GetTensorData<int8_t>(output));
#endif
          } else {
#if ESP_NN
            const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
            const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);

            TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
            TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
            const int filter_dim_count = filter_shape.DimensionsCount();
            const int output_dim_count = output_shape.DimensionsCount();
            const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
            const int output_depth = output_shape.Dims(output_dim_count - 1);
            TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
            const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

            const int32_t* bias_data =
                tflite::micro::GetOptionalTensorData<int32_t>(bias);

            const int8_t *input_data = tflite::micro::GetTensorData<int8_t>(input);
            int8_t *output_data = tflite::micro::GetTensorData<int8_t>(output);
            const int8_t *filter_data = tflite::micro::GetTensorData<int8_t>(filter);

            for (int b = 0; b < batches; ++b) {
              esp_nn_fully_connected_s8(input_data, -data.input_zero_point,
                                        accum_depth,
                                        filter_data, -data.filter_zero_point,
                                        bias_data, output_data, output_depth,
                                        data.output_zero_point,
                                        data.output_shift, data.output_multiplier,
                                        data.output_activation_min,
                                        data.output_activation_max);
              input_data += accum_depth;
              output_data += output_depth;
            }
#else
            tflite::micro::reference_integer_ops::FullyConnected(
                FullyConnectedParamsQuantized(data),
                tflite::micro::GetTensorShape(input),
                tflite::micro::GetTensorData<int8_t>(input),
                tflite::micro::GetTensorShape(filter),
                tflite::micro::GetTensorData<int8_t>(filter),
                tflite::micro::GetTensorShape(bias),
                tflite::micro::GetOptionalTensorData<int32_t>(bias),
                tflite::micro::GetTensorShape(output),
                tflite::micro::GetTensorData<int8_t>(output));
#endif
          }
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), input->type);
          return kTfLiteError;
        }
      }
      break;
    }

    case kTfLiteInt16: {
      switch (filter->type) {
        case kTfLiteInt8: {
          tflite::micro::reference_integer_ops::FullyConnected(
              FullyConnectedParamsQuantized(data),
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int64_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), input->type);
          return kTfLiteError;
        }
      }
      break;
    }

    default: {
      MicroPrintf("Input type %s (%d) not supported.",
                  TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
    }
  }
  fc_total_time += esp_timer_get_time() - start_time;
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(FullyConnectedInit, FullyConnectedPrepare,
                                   FullyConnectedEval);
}

TFLMInferenceRegistration RegisterInference_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(FullyConnectedEval);
}

}  // namespace micro
}  // namespace tflite
