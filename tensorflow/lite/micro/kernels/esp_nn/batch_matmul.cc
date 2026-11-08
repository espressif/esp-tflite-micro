/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/internal/reference/batch_matmul.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/batch_matmul.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#if ESP_NN
#include <esp_nn.h>
#include <esp_timer.h>
#endif

#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
extern "C" {
long long bmm_fc_us = 0;
int bmm_fc_calls = 0;
int bmm_fc_skip = 0;

/* Bench probe for one BATCH_MATMUL eval. rhs_copy/lhs_copy:
 * 0 = adj flag set, no data movement
 * 1 = reference_ops::Transpose ran this call
 * 2 = constant tensor, transpose already done on an earlier call */
struct BmmProfCall {
  int32_t lhs_rank;
  int32_t lhs_dims[5];
  int32_t rhs_rank;
  int32_t rhs_dims[5];
  int adj_x;
  int adj_y;
  int rhs_copy;
  int lhs_copy;
  int32_t rhs_bytes;
  int32_t lhs_bytes;
  long long rhs_tp_us;
  long long lhs_tp_us;
  long long kernel_us;
  long long eval_us;
  int align_copies;
};

constexpr int kBmmProfMax = 32;
long long bmm_eval_us = 0;
long long bmm_rhs_tp_us = 0;
long long bmm_lhs_tp_us = 0;
long long bmm_align_us = 0;
int bmm_align_copies = 0;
int bmm_prof_n = 0;
BmmProfCall bmm_prof_calls[kBmmProfMax];

void bmm_prof_reset(void) {
  bmm_eval_us = 0;
  bmm_rhs_tp_us = 0;
  bmm_lhs_tp_us = 0;
  bmm_align_us = 0;
  bmm_align_copies = 0;
  bmm_prof_n = 0;
}
}
#endif

namespace tflite {
namespace micro {
namespace {

#if ESP_NN
// Row sums that fit this many int32 stay on the stack (512 bytes).
constexpr int32_t kBatchMatmulStackSums = 128;

// The int8 fast path needs one row-sum per operand row/column. Those are
// sized and requested in Prepare, so Eval keeps them off the task stack.
struct OpDataBatchMatmulEsp {
  OpDataBatchMatmul base;  // must stay first: shared code casts through it
  int scratch_idx;
  int32_t scratch_elems;
  int fc_scratch_idx;
  int fc_scratch_bytes;
};

void ReshapeToFlattenBatchDimsIfPossible(RuntimeShape* lhs_shape,
                                         RuntimeShape* rhs_shape);
#endif  // ESP_NN

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node)
      : params(static_cast<TfLiteBatchMatMulParams*>(node->builtin_data)),
        op_data(static_cast<OpDataBatchMatmul*>(node->user_data)) {}

  TfLiteBatchMatMulParams* params;
  OpDataBatchMatmul* op_data;
};

struct PrepareOpContext : OpContext {
  PrepareOpContext(TfLiteContext* context, TfLiteNode* node)
      : OpContext(context, node),
        micro_context_(GetMicroContext(context)),
        lhs(micro_context_->AllocateTempInputTensor(
            node, kBatchMatmulInputLhsTensor)),
        rhs(micro_context_->AllocateTempInputTensor(
            node, kBatchMatmulInputRhsTensor)),
        output(micro_context_->AllocateTempOutputTensor(
            node, kBatchMatmulOutputTensor)) {}

  ~PrepareOpContext() {
    if (lhs != nullptr) {
      micro_context_->DeallocateTempTfLiteTensor(lhs);
    }
    if (rhs != nullptr) {
      micro_context_->DeallocateTempTfLiteTensor(rhs);
    }
    if (output != nullptr) {
      micro_context_->DeallocateTempTfLiteTensor(output);
    }
  }

 private:
  MicroContext* micro_context_;

 public:
  TfLiteTensor* lhs;
  TfLiteTensor* rhs;
  TfLiteTensor* output;
};

struct EvalOpContext : OpContext {
  EvalOpContext(TfLiteContext* context, TfLiteNode* node)
      : OpContext(context, node),
        lhs(tflite::micro::GetEvalInput(context, node,
                                        kBatchMatmulInputLhsTensor)),
        rhs(tflite::micro::GetEvalInput(context, node,
                                        kBatchMatmulInputRhsTensor)),
        output(tflite::micro::GetEvalOutput(context, node,
                                            kBatchMatmulOutputTensor)) {}

  const TfLiteEvalTensor* lhs;
  const TfLiteEvalTensor* rhs;
  TfLiteEvalTensor* output;
};

TfLiteEvalTensor* AllocInitTransposeTensorFromTfLiteTensor(
    TfLiteContext* context, const TfLiteTensor& tensor) {
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteEvalTensor* eval_tensor = static_cast<TfLiteEvalTensor*>(
      micro_context->AllocatePersistentBuffer(sizeof(TfLiteEvalTensor)));
  if (eval_tensor == nullptr) {
    return nullptr;
  }

  eval_tensor->type = tensor.type;

  const int tensor_rank = NumDimensions(&tensor);
  const size_t eval_dims_size = TfLiteIntArrayGetSizeInBytes(tensor_rank);
  eval_tensor->dims = static_cast<TfLiteIntArray*>(
      micro_context->AllocatePersistentBuffer(eval_dims_size));
  if (eval_tensor->dims == nullptr) {
    return nullptr;
  }
  eval_tensor->dims->size = tensor_rank;
  for (int i = 0; i < tensor_rank - 2; ++i) {
    eval_tensor->dims->data[i] = tensor.dims->data[i];
  }
  // Swap last two dimensions.
  eval_tensor->dims->data[tensor_rank - 2] = tensor.dims->data[tensor_rank - 1];
  eval_tensor->dims->data[tensor_rank - 1] = tensor.dims->data[tensor_rank - 2];

  const size_t eval_data_size = static_cast<size_t>(NumElements(&tensor)) *
                                TfLiteTypeGetSize(tensor.type);
  eval_tensor->data.data =
      micro_context->AllocatePersistentBuffer(eval_data_size);
  if (eval_tensor->data.data == nullptr) {
    return nullptr;
  }

  return eval_tensor;
}

// Initializes tensors to store transposed operands.
// Allocate storage for hybrid quantization if needed.
// Allocate normal quantization data if needed.
TfLiteStatus InitializeTemporaries(TfLiteContext* context, TfLiteNode* node,
                                   const PrepareOpContext& op_context) {
  OpDataBatchMatmul* op_data = op_context.op_data;
  const TfLiteTensor* lhs = op_context.lhs;
  const TfLiteTensor* rhs = op_context.rhs;
  MicroContext* micro_context = GetMicroContext(context);

  op_data->quantization = nullptr;
  op_data->lhs_transposed_tensor = nullptr;
  op_data->rhs_transposed_tensor = nullptr;

  if (lhs->type == kTfLiteInt8 || lhs->type == kTfLiteInt16) {
    op_data->quantization = static_cast<decltype(op_data->quantization)>(
        micro_context->AllocatePersistentBuffer(
            sizeof(*op_data->quantization)));
    TF_LITE_ENSURE(context, op_data->quantization != nullptr);
  }

  // tensor for Transposed LHS;
  if (op_context.params->adj_x) {
    op_data->lhs_transposed_tensor =
        AllocInitTransposeTensorFromTfLiteTensor(context, *lhs);
    TF_LITE_ENSURE(context, op_data->lhs_transposed_tensor != nullptr);
  }

  // We need a buffer for the RHS if we need to transpose the RHS. We
  // transpose by default, so that the two inputs (LHS and RHS) are in a proper
  // layout for our fast matrix multiplication routines. If the transpose flag
  // is set by the caller, the data is already in the desired layout.
  if (!op_context.params->adj_y) {
    op_data->rhs_transposed_tensor =
        AllocInitTransposeTensorFromTfLiteTensor(context, *rhs);
    TF_LITE_ENSURE(context, op_data->rhs_transposed_tensor != nullptr);
  }

  return kTfLiteOk;
}

void* BatchMatMulInit(TfLiteContext* context, const char* buffer,
                      size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  MicroContext* micro_context = GetMicroContext(context);
#if ESP_NN
  void* raw =
      micro_context->AllocatePersistentBuffer(sizeof(OpDataBatchMatmulEsp));
  if (raw != nullptr) {
    /* Prepare may bail out before it gets to request the scratch */
    OpDataBatchMatmulEsp* esp_data = static_cast<OpDataBatchMatmulEsp*>(raw);
    esp_data->scratch_idx = -1;
    esp_data->scratch_elems = 0;
    esp_data->fc_scratch_idx = -1;
    esp_data->fc_scratch_bytes = 0;
  }
  return raw;
#else
  return micro_context->AllocatePersistentBuffer(sizeof(OpDataBatchMatmul));
#endif
}

#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
/* Scratch for one 1x1 call, then one aligned copy of a head that missed
 * the 8-byte pointer check. The copy sits past the quant tables so the
 * kernel can use the front of the buffer while reading the copy. */
struct BmmFcScratch {
  int conv;
  int stage_at;
  int input_stage;
  int filter_stage;
  int total;
};

static int BmmRound16(int64_t bytes) {
  return static_cast<int>((bytes + 15) & ~int64_t{15});
}

static BmmFcScratch BmmFcScratchPlan(int32_t batches, int32_t depth,
                                    int32_t out_ch) {
  BmmFcScratch plan = {0, 0, 0, 0, 0};
  if (batches <= 0 || depth <= 0 || out_ch <= 0 || batches > 65535 ||
      depth > 65535 || out_ch > 65535) {
    return plan;
  }
  plan.conv = esp_nn_get_fc_batch_scratch_size(
      batches, static_cast<uint16_t>(depth), static_cast<uint16_t>(out_ch));
  if (plan.conv <= 0) {
    return plan;
  }
  const int addend_at = (plan.conv + 15) & ~15;
  const int quant_at = addend_at + batches * static_cast<int>(sizeof(int32_t));
  const int need = quant_at + out_ch * 2 * static_cast<int>(sizeof(int32_t));
  plan.stage_at = (need + 15) & ~15;
  plan.input_stage = BmmRound16(static_cast<int64_t>(batches) * depth);
  plan.filter_stage = BmmRound16(static_cast<int64_t>(out_ch) * depth);
  plan.total = plan.stage_at + plan.input_stage + plan.filter_stage + 16;
  return plan;
}
#endif

TfLiteStatus BatchMatMulPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  PrepareOpContext op_context(context, node);
  const TfLiteTensor* lhs_data = op_context.lhs;
  TF_LITE_ENSURE(context, lhs_data != nullptr);
  const TfLiteTensor* rhs_data = op_context.rhs;
  TF_LITE_ENSURE(context, rhs_data != nullptr);
  TfLiteTensor* output = op_context.output;
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE(context, lhs_data->type == kTfLiteFloat32 ||
                              lhs_data->type == kTfLiteInt8 ||
                              lhs_data->type == kTfLiteInt16);
  TF_LITE_ENSURE(context, rhs_data->type == kTfLiteFloat32 ||
                              rhs_data->type == kTfLiteInt8 ||
                              rhs_data->type == kTfLiteInt16);
  // Both inputs should be of the same type.
  // Hybrid input (FLOAT32 LHS, INT8 RHS) is not supported.
  TF_LITE_ENSURE(context, lhs_data->type == rhs_data->type);
  // LHS input must match output type.  INT32 output not supported.
  TF_LITE_ENSURE(context, lhs_data->type == output->type);

  const int lhs_rank = NumDimensions(lhs_data);
  const int rhs_rank = NumDimensions(rhs_data);
  // Support dimensions between 2 and 5, inclusive.
  TF_LITE_ENSURE(context, lhs_rank >= 2);
  TF_LITE_ENSURE(context, lhs_rank <= 5);
  TF_LITE_ENSURE(context, rhs_rank >= 2);
  TF_LITE_ENSURE(context, rhs_rank <= 5);

  TF_LITE_ENSURE_OK(context, InitializeTemporaries(context, node, op_context));

  OpDataBatchMatmul* op_data = op_context.op_data;
  // If the RHS is constant, we only transpose once.
  op_data->rhs_is_transposed = false;
  op_data->lhs_is_constant_tensor = IsConstantTensor(lhs_data);
  op_data->rhs_is_constant_tensor = IsConstantTensor(rhs_data);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (lhs_data->type == kTfLiteInt8 || lhs_data->type == kTfLiteInt16) {
    TF_LITE_ENSURE(context, op_data->quantization != nullptr);
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, lhs_data, rhs_data, output, &real_multiplier));
    QuantizeMultiplier(real_multiplier,
                       &op_data->quantization->output_multiplier,
                       &op_data->quantization->output_shift);
    // BatchMatMul has no fused activation functions. Therefore, set
    // output activation min and max to min and max of int8_t or int16_t type.
    if (lhs_data->type == kTfLiteInt8) {
      op_data->quantization->output_activation_min =
          std::numeric_limits<int8_t>::min();
      op_data->quantization->output_activation_max =
          std::numeric_limits<int8_t>::max();
    } else {
      op_data->quantization->output_activation_min =
          std::numeric_limits<int16_t>::min();
      op_data->quantization->output_activation_max =
          std::numeric_limits<int16_t>::max();

      TF_LITE_ENSURE_EQ(context, lhs_data->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, rhs_data->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    }

    op_data->quantization->lhs_zero_point = lhs_data->params.zero_point;
    op_data->quantization->rhs_zero_point = rhs_data->params.zero_point;
    op_data->quantization->output_zero_point = output->params.zero_point;
  }

  const int output_rank = std::max(lhs_rank, rhs_rank);
  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(output_rank, GetTensorShape(lhs_data));
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(output_rank, GetTensorShape(rhs_data));

  // Ensure any batch dimensions obey broadcasting rules.
  for (int i = 0; i < output_rank - 2; ++i) {
    const int lhs_dim = extended_lhs_shape.Dims(i);
    const int rhs_dim = extended_rhs_shape.Dims(i);
    if (lhs_dim != rhs_dim) {
      if (lhs_dim != 1) {
        TF_LITE_ENSURE_EQ(context, rhs_dim, 1);
      }
    }
  }
  bool adj_x = op_context.params->adj_x;
  bool adj_y = op_context.params->adj_y;
  // Ensure other dimensions work for matrix multiplication.
  int accum_dim_lhs = adj_x ? extended_lhs_shape.Dims(output_rank - 2)
                            : extended_lhs_shape.Dims(output_rank - 1);
  int accum_dim_rhs = adj_y ? extended_rhs_shape.Dims(output_rank - 1)
                            : extended_rhs_shape.Dims(output_rank - 2);

  TF_LITE_ENSURE_EQ(context, accum_dim_lhs, accum_dim_rhs);
  TfLiteStatus status =
      ReshapeOutputTensor(context, node, extended_lhs_shape, extended_rhs_shape,
                          adj_x, adj_y, output_rank, output);

#if ESP_NN
  // Wide operands need their row sums in the arena. Size that from the
  // shapes Eval will derive: the same swap and flattening, done here once.
  OpDataBatchMatmulEsp* esp_data =
      static_cast<OpDataBatchMatmulEsp*>(node->user_data);
  esp_data->scratch_idx = -1;
  esp_data->scratch_elems = 0;
  esp_data->fc_scratch_idx = -1;
  esp_data->fc_scratch_bytes = 0;
  if (status == kTfLiteOk && lhs_data->type == kTfLiteInt8) {
    RuntimeShape fast_lhs_shape =
        adj_x ? GetTensorShape(lhs_data)
              : SwapRowColumnDims(GetTensorShape(lhs_data));
    RuntimeShape fast_rhs_shape =
        adj_y ? GetTensorShape(rhs_data)
              : SwapRowColumnDims(GetTensorShape(rhs_data));
    ReshapeToFlattenBatchDimsIfPossible(&fast_lhs_shape, &fast_rhs_shape);
    // Eval passes RHS args first, so rows come from the RHS shape and
    // columns from the LHS shape.
    const RuntimeShape fast_rhs_ext =
        RuntimeShape::ExtendedShape(5, fast_rhs_shape);
    const RuntimeShape fast_lhs_ext =
        RuntimeShape::ExtendedShape(5, fast_lhs_shape);
    const int32_t rows = fast_rhs_ext.Dims(3);
    const int32_t cols = fast_lhs_ext.Dims(4);
    const int32_t depth = fast_rhs_ext.Dims(4);
    if (rows > 0 && cols > 0 && rows + cols > kBatchMatmulStackSums) {
      int idx = -1;
      if (context->RequestScratchBufferInArena(
              context, (size_t)(rows + cols) * sizeof(int32_t), &idx) ==
          kTfLiteOk) {
        esp_data->scratch_idx = idx;
        esp_data->scratch_elems = rows + cols;
      }
    }
#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
    if (rows > 0 && rows <= 65535 && cols > 0 && cols <= 65535 && depth > 0 &&
        depth <= 65535) {
      const BmmFcScratch plan = BmmFcScratchPlan(cols, depth, rows);
      if (plan.conv > 0) {
        int idx = -1;
        if (context->RequestScratchBufferInArena(
                context, static_cast<size_t>(plan.total), &idx) == kTfLiteOk) {
          esp_data->fc_scratch_idx = idx;
          esp_data->fc_scratch_bytes = plan.total;
        }
      }
    }
#endif
  }
#endif  // ESP_NN

  return status;
}


#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
// One head of QK or attention×V is a batched fully connected layer: the
// query/attention rows are the batch and the key/value rows are the filter.
// A nonzero filter zero-point is applied inside the 1x1 kernel.
static bool BatchMatMulViaFc(TfLiteContext* context,
                             const OpDataBatchMatmul& data,
                             const FullyConnectedParams& params,
                             const RuntimeShape& lhs_shape,
                             const int8_t* lhs_data,
                             const RuntimeShape& rhs_shape,
                             const int8_t* rhs_data,
                             int8_t* output_data) {
  const RuntimeShape elhs = RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape erhs = RuntimeShape::ExtendedShape(5, rhs_shape);
  for (int d = 0; d < 3; d++) {
    if (elhs.Dims(d) != erhs.Dims(d)) {
      return false;
    }
  }
  const int32_t heads = elhs.Dims(0) * elhs.Dims(1) * elhs.Dims(2);
  const int32_t lhs_rows = elhs.Dims(3);
  const int32_t rhs_cols = erhs.Dims(4);
  const int32_t depth = elhs.Dims(4);
  if (heads < 1 || lhs_rows < 1 || rhs_cols < 1 || depth < 1 ||
      erhs.Dims(3) * erhs.Dims(4) != rhs_cols * depth ||
      lhs_rows > 65535 || rhs_cols > 65535 || depth > 65535) {
    return false;
  }
  const OpDataBatchMatmulEsp* esp_data =
      reinterpret_cast<const OpDataBatchMatmulEsp*>(&data);
  if (esp_data->fc_scratch_idx < 0) {
    return false;
  }
  uint8_t* raw = static_cast<uint8_t*>(
      context->GetScratchBuffer(context, esp_data->fc_scratch_idx));
  const uintptr_t aligned =
      (reinterpret_cast<uintptr_t>(raw) + 15u) & ~uintptr_t{15};
  const int align_loss =
      static_cast<int>(aligned - reinterpret_cast<uintptr_t>(raw));
  if (align_loss + 16 > esp_data->fc_scratch_bytes) {
    return false;
  }
  uint8_t* scratch = reinterpret_cast<uint8_t*>(aligned);
  const int usable = esp_data->fc_scratch_bytes - align_loss;
  const BmmFcScratch plan = BmmFcScratchPlan(rhs_cols, depth, lhs_rows);
  if (plan.conv <= 0 || plan.total - 16 > usable) {
    return false;
  }
  const int quant_at =
      ((plan.conv + 15) & ~15) + rhs_cols * static_cast<int>(sizeof(int32_t));
  if (((reinterpret_cast<uintptr_t>(lhs_data) | reinterpret_cast<uintptr_t>(rhs_data)) &
       7u) != 0) {
    return false;
  }
  int32_t* mult = reinterpret_cast<int32_t*>(scratch + quant_at);
  int32_t* shift = mult + lhs_rows;
  for (int32_t oc = 0; oc < lhs_rows; ++oc) {
    mult[oc] = params.output_multiplier;
    shift[oc] = params.output_shift;
  }
  uint8_t* input_stage = scratch + plan.stage_at;
  uint8_t* filter_stage = input_stage + plan.input_stage;
  const size_t input_bytes = static_cast<size_t>(rhs_cols) * depth;
  const size_t filter_bytes = static_cast<size_t>(lhs_rows) * depth;
  const int64_t t0 = esp_timer_get_time();
  for (int32_t h = 0; h < heads; ++h) {
    const int8_t* filter = lhs_data + static_cast<size_t>(h) * lhs_rows * depth;
    const int8_t* input = rhs_data + static_cast<size_t>(h) * rhs_cols * depth;
    int8_t* out = output_data + static_cast<size_t>(h) * lhs_rows * rhs_cols;
    if ((reinterpret_cast<uintptr_t>(input) & 7u) != 0) {
      const int64_t c0 = esp_timer_get_time();
      memcpy(input_stage, input, input_bytes);
      bmm_align_us += esp_timer_get_time() - c0;
      bmm_align_copies += 1;
      input = reinterpret_cast<const int8_t*>(input_stage);
    }
    if ((reinterpret_cast<uintptr_t>(filter) & 7u) != 0) {
      const int64_t c0 = esp_timer_get_time();
      memcpy(filter_stage, filter, filter_bytes);
      bmm_align_us += esp_timer_get_time() - c0;
      bmm_align_copies += 1;
      filter = reinterpret_cast<const int8_t*>(filter_stage);
    }
    esp_nn_set_fc_scratch_buf(scratch, static_cast<size_t>(usable));
    esp_nn_fully_connected_per_ch_s8_batch(
        input, params.input_offset, static_cast<uint16_t>(depth), filter,
        params.weights_offset, nullptr, out, static_cast<uint16_t>(lhs_rows),
        params.output_offset, shift, mult, params.quantized_activation_min,
        params.quantized_activation_max, rhs_cols);
  }
  bmm_fc_us += esp_timer_get_time() - t0;
  bmm_fc_calls += 1;
  return true;
}
#endif

#if ESP_NN
// Espressif fast path for the quantized reference BatchMatMul. Mirrors the
// reference exactly (positional args and dim conventions: lhs rows along
// Dims(3), rhs COLS along Dims(4) - the kernel pre-transposes RHS data so
// both operands are row-contiguous along the contraction axis). Zero-point
// offsets are folded out of the inner loop via row sums -
//   sum_k (l+fo)(r+io) = dot(l,r) + fo*sum(r) + io*sum(l) + K*fo*io
// - with the same integer accumulation and the same
// MultiplyByQuantizedMultiplier: bit-identical results. Handles the
// no-broadcast batch case; returns false otherwise.
static bool BatchMatMulInt8Fast(const FullyConnectedParams& params,
                                const RuntimeShape& lhs_shape,
                                const int8_t* lhs_data,
                                const RuntimeShape& rhs_shape,
                                const int8_t* rhs_data,
                                const RuntimeShape& output_shape,
                                int8_t* output_data, int32_t* scratch,
                                int32_t scratch_elems) {
  const RuntimeShape elhs = RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape erhs = RuntimeShape::ExtendedShape(5, rhs_shape);
  for (int d = 0; d < 3; d++) {
    if (elhs.Dims(d) != erhs.Dims(d)) {
      return false;  // broadcast batches: keep the reference path
    }
  }
  const int32_t batches = elhs.Dims(0) * elhs.Dims(1) * elhs.Dims(2);
  const int32_t lhs_rows = elhs.Dims(3);
  const int32_t rhs_cols = erhs.Dims(4);
  const int32_t depth = elhs.Dims(4);
  if (erhs.Dims(3) * erhs.Dims(4) != rhs_cols * depth || depth < 1) {
    return false;
  }
  const int32_t fo = params.weights_offset;  // reference adds fo to lhs vals
  const int32_t io = params.input_offset;    // reference adds io to rhs vals
  const int64_t kfoio = (int64_t)depth * fo * io;
  if (kfoio > INT32_MAX || kfoio < INT32_MIN) {
    return false;
  }

  // Row sums are re-read across the whole output, so keep the small ones in
  // a bounded stack buffer; wider operands take the scratch from the arena.
  int32_t stack_sums[kBatchMatmulStackSums];
  int32_t* sums;
  if (lhs_rows + rhs_cols <= kBatchMatmulStackSums) {
    sums = stack_sums;
  } else if (scratch != nullptr && lhs_rows + rhs_cols <= scratch_elems) {
    sums = scratch;
  } else {
    return false;
  }
  /* the int8 output never overlaps the sums; say so or every read reloads */
  int32_t* __restrict lhs_sums = sums;
  int32_t* __restrict rhs_sums = sums + lhs_rows;
  for (int32_t b = 0; b < batches; b++) {
    const int8_t* lhs = lhs_data + (size_t)b * lhs_rows * depth;
    const int8_t* rhs = rhs_data + (size_t)b * rhs_cols * depth;
    int8_t* out = output_data + (size_t)b * lhs_rows * rhs_cols;
    for (int32_t i = 0; i < lhs_rows; i++) {
      int32_t s = 0;
      const int8_t* p = lhs + (size_t)i * depth;
      for (int32_t k = 0; k < depth; k++) s += p[k];
      /* lhs vals carry fo; cross terms: io*sum(lhs_row) + K*fo*io */
      lhs_sums[i] = s * io + (int32_t)kfoio;
    }
    for (int32_t j = 0; j < rhs_cols; j++) {
      int32_t s = 0;
      const int8_t* p = rhs + (size_t)j * depth;
      for (int32_t k = 0; k < depth; k++) s += p[k];
      rhs_sums[j] = s * fo;
    }
    for (int32_t j = 0; j < rhs_cols; j++) {
      const int8_t* rp = rhs + (size_t)j * depth;
      /* int8 output may alias the sums, so hoist the invariant read */
      const int32_t rsum = rhs_sums[j];
      for (int32_t i = 0; i < lhs_rows; i++) {
        const int8_t* lp = lhs + (size_t)i * depth;
        int32_t acc = 0;
        int32_t k = 0;
        for (; k <= depth - 4; k += 4) {
          acc += lp[k] * rp[k];
          acc += lp[k + 1] * rp[k + 1];
          acc += lp[k + 2] * rp[k + 2];
          acc += lp[k + 3] * rp[k + 3];
        }
        for (; k < depth; k++) acc += lp[k] * rp[k];
        acc += lhs_sums[i] + rsum;
        int32_t r = MultiplyByQuantizedMultiplier(acc, params.output_multiplier,
                                                  params.output_shift);
        r += params.output_offset;
        r = std::max(r, params.quantized_activation_min);
        r = std::min(r, params.quantized_activation_max);
        out[(size_t)lhs_rows * j + i] = static_cast<int8_t>(r);
      }
    }
  }
  return true;
}
#endif  // ESP_NN



TfLiteStatus EvalInt8(TfLiteContext* context, const OpDataBatchMatmul& data,
                      const RuntimeShape& lhs_shape,
                      const TfLiteEvalTensor& lhs,
                      const RuntimeShape& rhs_shape,
                      const TfLiteEvalTensor& rhs,
                      const RuntimeShape& output_shape,
                      TfLiteEvalTensor* output) {
  TF_LITE_ENSURE(context, data.quantization != nullptr);
  // Reuse params struct from FullyConnected Op.
  FullyConnectedParams op_params;
  op_params.input_offset = -data.quantization->lhs_zero_point;
  op_params.weights_offset =
      -data.quantization->rhs_zero_point;  // filter offset
  op_params.output_offset = data.quantization->output_zero_point;
  op_params.output_multiplier = data.quantization->output_multiplier;
  op_params.output_shift = data.quantization->output_shift;
  op_params.quantized_activation_min = data.quantization->output_activation_min;
  op_params.quantized_activation_max = data.quantization->output_activation_max;
  op_params.lhs_cacheable = data.lhs_is_constant_tensor;
  op_params.rhs_cacheable = data.rhs_is_constant_tensor;

  // Note we pass RHS args first, LHS args second. See note for Eval.
#if ESP_NN
  /* base is the first member, so the shared OpData aliases the esp one */
  const OpDataBatchMatmulEsp* esp_data =
      reinterpret_cast<const OpDataBatchMatmulEsp*>(&data);
  int32_t* sums =
      esp_data->scratch_idx >= 0
          ? static_cast<int32_t*>(
                context->GetScratchBuffer(context, esp_data->scratch_idx))
          : nullptr;
#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
  if (BatchMatMulViaFc(
          context, data, op_params, rhs_shape,
          tflite::micro::GetTensorData<int8_t>(&rhs), lhs_shape,
          tflite::micro::GetTensorData<int8_t>(&lhs),
          tflite::micro::GetTensorData<int8_t>(output))) {
    return kTfLiteOk;
  }
  bmm_fc_skip += 1;
#endif
  if (!BatchMatMulInt8Fast(
          op_params, rhs_shape, tflite::micro::GetTensorData<int8_t>(&rhs),
          lhs_shape, tflite::micro::GetTensorData<int8_t>(&lhs), output_shape,
          tflite::micro::GetTensorData<int8_t>(output), sums,
          esp_data->scratch_elems))
#endif  // ESP_NN
  {
    reference_ops::BatchMatMul<int8_t, int32_t>(
        op_params, rhs_shape, tflite::micro::GetTensorData<int8_t>(&rhs),
        lhs_shape, tflite::micro::GetTensorData<int8_t>(&lhs), output_shape,
        tflite::micro::GetTensorData<int8_t>(output));
  }

  return kTfLiteOk;
}

TfLiteStatus EvalInt16(TfLiteContext* context, const OpDataBatchMatmul& data,
                       const RuntimeShape& lhs_shape,
                       const TfLiteEvalTensor& lhs,
                       const RuntimeShape& rhs_shape,
                       const TfLiteEvalTensor& rhs,
                       const RuntimeShape& output_shape,
                       TfLiteEvalTensor* output) {
  TF_LITE_ENSURE(context, data.quantization != nullptr);
  // Reuse params struct from FullyConnected Op.
  FullyConnectedParams op_params;
  op_params.input_offset = -data.quantization->lhs_zero_point;
  op_params.weights_offset =
      -data.quantization->rhs_zero_point;  // filter offset
  op_params.output_offset = data.quantization->output_zero_point;
  op_params.output_multiplier = data.quantization->output_multiplier;
  op_params.output_shift = data.quantization->output_shift;
  op_params.quantized_activation_min = data.quantization->output_activation_min;
  op_params.quantized_activation_max = data.quantization->output_activation_max;
  op_params.lhs_cacheable = data.lhs_is_constant_tensor;
  op_params.rhs_cacheable = data.rhs_is_constant_tensor;

  // Note we pass RHS args first, LHS args second. See note for Eval.
  reference_ops::BatchMatMul<int16_t, int64_t>(
      op_params, rhs_shape, tflite::micro::GetTensorData<int16_t>(&rhs),
      lhs_shape, tflite::micro::GetTensorData<int16_t>(&lhs), output_shape,
      tflite::micro::GetTensorData<int16_t>(output));

  return kTfLiteOk;
}

void ReshapeToFlattenBatchDimsIfPossible(RuntimeShape* lhs_shape,
                                         RuntimeShape* rhs_shape) {
  // Compress BatchMatMul when third from last RHS dimension is one.
  int32_t rhs_dims_count = rhs_shape->DimensionsCount();
  int32_t lhs_dims_count = lhs_shape->DimensionsCount();

  // Compress ops where rhs shape is [..., 1, X, Y] and lhs shape is
  // [..., Q, R, S].
  // After shape assignment, rhs shape represents
  // [..., 1, lhs_rows, accum_depth] and lhs shape represents
  // [..., batch, accum_depth, rhs_cols].
  //
  // When rhs has a batch dimension of 1, we can collapse the batch dimension
  // of lhs into rhs_cols, treating [..., batch, accum_depth, rhs_cols] as
  // [..., accum_depth, batch * rhs_cols].
  //
  // This works because whether adj_x is false (where orig_lhs is
  // [batch, rhs_cols, accum_depth]) or adj_x is true (where orig_lhs is
  // [batch, accum_depth, rhs_cols] but transposed in memory), the underlying
  // buffer in memory has batch and rhs_cols as contiguous outer dimensions,
  // which can be physically collapsed into batch * rhs_cols.
  if (rhs_dims_count > 2 && lhs_dims_count > 2) {
    int rhs_one = rhs_shape->DimsData()[rhs_dims_count - 3];
    if (rhs_one == 1) {
      int32_t* lhs_dims = lhs_shape->DimsData();
      int32_t* rhs_dims = rhs_shape->DimsData();
      RuntimeShape tmp_l(lhs_dims_count - 1, lhs_dims);
      tmp_l.SetDim(lhs_dims_count - 3, lhs_dims[lhs_dims_count - 2]);
      tmp_l.SetDim(lhs_dims_count - 2,
                   lhs_dims[lhs_dims_count - 3] * lhs_dims[lhs_dims_count - 1]);
      lhs_shape->ReplaceWith(tmp_l.DimensionsCount(), tmp_l.DimsData());
      RuntimeShape tmp_r(rhs_dims_count - 1, rhs_shape->DimsData());
      tmp_r.SetDim(rhs_dims_count - 3, rhs_dims[rhs_dims_count - 2]);
      tmp_r.SetDim(rhs_dims_count - 2, rhs_dims[rhs_dims_count - 1]);
      rhs_shape->ReplaceWith(tmp_r.DimensionsCount(), tmp_r.DimsData());
    }
  }
}

// Perform a batch matrix multiply on
// LHS <..., A, B>  X  RHS<..., B, C>
// where the leading dimensions of LHS and RHS obey broadcasting rules
// (this Op will apply broadcasting rules).
// We assume that LHS and RHS are both row oriented (adjacent values in memory
// are in the same row) and will output in the same memory layout. However,
// our fast GEMM libraries assume RCC layout (LHS row oriented,
// RHS column oriented, output column oriented). Therefore, we perform
// RHS <..., C, B> X LHS <..., B, A>
// where output is a C X A column-oriented, which is equivalent to
// A X C row-oriented.
#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
static void BmmCopyDims(const TfLiteEvalTensor* tensor, int32_t* rank,
                        int32_t* dims) {
  const int n = tensor->dims->size;
  *rank = n;
  for (int i = 0; i < 5; ++i) {
    dims[i] = i < n ? tensor->dims->data[i] : 0;
  }
}

static void BmmProfPrintDims(const int32_t rank, const int32_t* dims) {
  const int n = rank < 5 ? rank : 5;
  for (int i = 0; i < n; ++i) {
    printf("%s%ld", i == 0 ? "" : "x", static_cast<long>(dims[i]));
  }
}

extern "C" void bmm_prof_print(void) {
  const long long other_us =
      bmm_eval_us - bmm_rhs_tp_us - bmm_lhs_tp_us - bmm_fc_us;
  printf(
      "BMM_PROFILE calls=%d eval_us=%lld rhs_tp_us=%lld lhs_tp_us=%lld "
      "kernel_us=%lld align_us=%lld align_copies=%d other_us=%lld "
      "note=reference_transpose\n",
      bmm_prof_n, bmm_eval_us, bmm_rhs_tp_us, bmm_lhs_tp_us, bmm_fc_us,
      bmm_align_us, bmm_align_copies, other_us);
  for (int i = 0; i < bmm_prof_n; ++i) {
    const BmmProfCall& call = bmm_prof_calls[i];
    printf(
        "BMM_CALL i=%d adj_x=%d adj_y=%d rhs_copy=%d lhs_copy=%d "
        "rhs_bytes=%ld lhs_bytes=%ld rhs_tp_us=%lld lhs_tp_us=%lld "
        "kernel_us=%lld eval_us=%lld lhs=",
        i, call.adj_x, call.adj_y, call.rhs_copy, call.lhs_copy,
        static_cast<long>(call.rhs_bytes), static_cast<long>(call.lhs_bytes),
        call.rhs_tp_us, call.lhs_tp_us, call.kernel_us, call.eval_us);
    BmmProfPrintDims(call.lhs_rank, call.lhs_dims);
    printf(" rhs=");
    BmmProfPrintDims(call.rhs_rank, call.rhs_dims);
    printf("\n");
  }
}
#endif

TfLiteStatus BatchMatMulEval(TfLiteContext* context, TfLiteNode* node) {
#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
  const int64_t eval_t0 = esp_timer_get_time();
  const long long kernel_before = bmm_fc_us;
  const int align_before = bmm_align_copies;
  long long rhs_tp_us = 0;
  long long lhs_tp_us = 0;
  int rhs_copy = 0;
  int lhs_copy = 0;
  int32_t rhs_bytes = 0;
  int32_t lhs_bytes = 0;
#endif
  EvalOpContext op_context(context, node);
  OpDataBatchMatmul* op_data = op_context.op_data;
  const TfLiteEvalTensor* lhs = op_context.lhs;
  const TfLiteEvalTensor* rhs = op_context.rhs;
  TfLiteEvalTensor* output = op_context.output;
  RuntimeShape orig_lhs_shape = tflite::micro::GetTensorShape(lhs);
  RuntimeShape orig_rhs_shape = tflite::micro::GetTensorShape(rhs);

  bool adj_y = op_context.params->adj_y;
  bool adj_x = op_context.params->adj_x;

  TfLiteEvalTensor* rhs_tensor = adj_y ? const_cast<TfLiteEvalTensor*>(rhs)
                                       : op_data->rhs_transposed_tensor;
  TfLiteEvalTensor* lhs_tensor = adj_x ? op_data->lhs_transposed_tensor
                                       : const_cast<TfLiteEvalTensor*>(lhs);
  TF_LITE_ENSURE(context, rhs_tensor != nullptr);
  TF_LITE_ENSURE(context, lhs_tensor != nullptr);
  if (!adj_y) {
    // TODO(b/154760341): Constant tensors should already be transposed, but
    // we transpose once if necessary for now.
    if (!(op_data->rhs_is_constant_tensor && op_data->rhs_is_transposed)) {
#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
      rhs_bytes = static_cast<int32_t>(NumElements(rhs));
      const int64_t tp0 = esp_timer_get_time();
#endif
      TransposeRowsColumns(*rhs, rhs_tensor);
#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
      rhs_tp_us = esp_timer_get_time() - tp0;
      rhs_copy = 1;
#endif
      op_data->rhs_is_transposed = true;
    }
#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
    else {
      rhs_copy = 2;
    }
#endif
  }
  if (adj_x) {
#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
    lhs_bytes = static_cast<int32_t>(NumElements(lhs));
    const int64_t tp0 = esp_timer_get_time();
#endif
    TransposeRowsColumns(*lhs, lhs_tensor);
#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
    lhs_tp_us = esp_timer_get_time() - tp0;
    lhs_copy = 1;
#endif
  }
  RuntimeShape rhs_shape =
      adj_y ? orig_rhs_shape : SwapRowColumnDims(orig_rhs_shape);
  RuntimeShape lhs_shape =
      adj_x ? orig_lhs_shape : SwapRowColumnDims(orig_lhs_shape);

  ReshapeToFlattenBatchDimsIfPossible(&lhs_shape, &rhs_shape);

  TfLiteStatus status = kTfLiteOk;
  switch (lhs->type) {
    case kTfLiteFloat32:
      // Note we pass RHS args first, LHS args second. See note above.
      reference_ops::BatchMatMul(
          rhs_shape, tflite::micro::GetTensorData<float>(rhs_tensor), lhs_shape,
          tflite::micro::GetTensorData<float>(lhs_tensor),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      status = EvalInt8(context, *op_data, lhs_shape, *lhs_tensor, rhs_shape,
                        *rhs_tensor, tflite::micro::GetTensorShape(output),
                        output);
      break;
    case kTfLiteInt16:
      status = EvalInt16(context, *op_data, lhs_shape, *lhs_tensor, rhs_shape,
                         *rhs_tensor, tflite::micro::GetTensorShape(output),
                         output);
      break;
    default:
      MicroPrintf("BATCH_MATMUL doesn't support input type %s",
                  TfLiteTypeGetName(lhs->type));
      status = kTfLiteError;
      break;
  }
#if defined(ESP_NN_HAS_FC_S8_BATCH) && defined(ESP_NN_HAS_FC_SCRATCH_BUF)
  const long long eval_us = esp_timer_get_time() - eval_t0;
  const long long kernel_us = bmm_fc_us - kernel_before;
  bmm_eval_us += eval_us;
  bmm_rhs_tp_us += rhs_tp_us;
  bmm_lhs_tp_us += lhs_tp_us;
  if (bmm_prof_n < kBmmProfMax) {
    BmmProfCall* call = &bmm_prof_calls[bmm_prof_n++];
    BmmCopyDims(lhs, &call->lhs_rank, call->lhs_dims);
    BmmCopyDims(rhs, &call->rhs_rank, call->rhs_dims);
    call->adj_x = adj_x ? 1 : 0;
    call->adj_y = adj_y ? 1 : 0;
    call->rhs_copy = rhs_copy;
    call->lhs_copy = lhs_copy;
    call->rhs_bytes = rhs_bytes;
    call->lhs_bytes = lhs_bytes;
    call->rhs_tp_us = rhs_tp_us;
    call->lhs_tp_us = lhs_tp_us;
    call->kernel_us = kernel_us;
    call->eval_us = eval_us;
    call->align_copies = bmm_align_copies - align_before;
  }
#endif
  return status;
}

}  // namespace

TFLMRegistration Register_BATCH_MATMUL() {
  return tflite::micro::RegisterOp(BatchMatMulInit, BatchMatMulPrepare,
                                   BatchMatMulEval);
}

}  // namespace micro
}  // namespace tflite
