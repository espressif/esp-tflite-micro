// Copyright 2020-2021 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stddef.h>
#include <stdint.h>

#include <common_functions.h>
#include <esp_nn.h>
#include <esp_nn_multicore.h>
#include "esp_nn_fc_internal.h"

/* Measured ESP32-S3 floors. One core beats the S3 fully connected kernel
 * from batch 3, input 8, and output 8. Batch 1 and 2 are faster only when
 * the shared 1x1 kernel splits across both cores: the row length is a
 * multiple of 8, filter bytes are above 24 KiB, and there are at least 32
 * outputs. The filter and output numbers match the split in
 * esp_nn_conv_esp32s3.c. esp_nn_dual_core_active() means the worker exists.
 * That kernel still runs the same shape on one core when its worker scratch
 * does not fit, or when esp_nn_dual_core_run() refuses a nested call or a
 * call already on the worker's core. Those one-core shapes lose to fully
 * connected. This gate cannot see that refusal. */
enum {
    ESP_NN_FC_S3_MIN_BATCH = 3,
    ESP_NN_FC_S3_MIN_ROW = 8,
    ESP_NN_FC_S3_MIN_OUT = 8,
    ESP_NN_FC_S3_DUAL_MIN_OUT = 32,
    ESP_NN_FC_S3_DUAL_MIN_FILTER_BYTES = 24 * 1024,
};

/* Measured ESP32-P4 floors. ESP32-S31 uses the same PIE kernels, so it
 * uses these floors too. Batch 1 already wins. Widths under 8 were not
 * measured. Output 4 wins. The shared convolution still starts its second
 * core at batch 4; this gate does not change that. */
enum {
    ESP_NN_FC_P4_MIN_BATCH = 1,
    ESP_NN_FC_P4_MIN_ROW = 8,
    ESP_NN_FC_P4_MIN_OUT = 4,
};

/* Performance only. Correctness checks stay in the caller. Targets that
 * were not swept keep the original cutoffs. */
static int fc_conv_shape_wins(int32_t batches, uint16_t row_len,
                              uint16_t out_channels)
{
#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)
    if (row_len < ESP_NN_FC_S3_MIN_ROW) {
        return 0;
    }
    if (batches >= ESP_NN_FC_S3_MIN_BATCH &&
            out_channels >= ESP_NN_FC_S3_MIN_OUT) {
        return 1;
    }
    if (batches >= 1 && batches < ESP_NN_FC_S3_MIN_BATCH &&
            (row_len & 7) == 0 &&
            out_channels >= ESP_NN_FC_S3_DUAL_MIN_OUT &&
            (uint32_t) row_len * (uint32_t) out_channels >
                (uint32_t) ESP_NN_FC_S3_DUAL_MIN_FILTER_BYTES &&
            esp_nn_dual_core_active()) {
        return 1;
    }
    return 0;
#elif defined(CONFIG_NN_OPTIMIZED) && \
        (defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S31))
    return batches >= ESP_NN_FC_P4_MIN_BATCH &&
            row_len >= ESP_NN_FC_P4_MIN_ROW &&
            out_channels >= ESP_NN_FC_P4_MIN_OUT;
#else
    return batches >= 4 && out_channels >= 16 &&
            row_len >= 64 && (row_len & 7) == 0;
#endif
}

void esp_nn_fully_connected_s8_ansi(const int8_t *input_data,
                                    const int32_t input_offset,
                                    const uint16_t row_len,
                                    const int8_t *filter_data,
                                    const int32_t filter_offset,
                                    const int32_t *bias,
                                    int8_t *out_data,
                                    const uint16_t out_channels,
                                    const int32_t out_offset,
                                    const int32_t out_shift,
                                    const int32_t out_mult,
                                    const int32_t activation_min,
                                    const int32_t activation_max)
{
    for (int32_t out_c = 0; out_c < out_channels; ++out_c) {
        int32_t result = 0;
        for (int32_t data_idx = 0; data_idx < row_len; data_idx++) {
            int32_t filter_index = row_len * out_c + data_idx;
            int32_t input_val = input_data[data_idx];
            int32_t filter_val = filter_data[filter_index];
            result += (filter_val + filter_offset) * (input_val + input_offset);
        }
        if (bias) {
            result += bias[out_c];
        }
        result = esp_nn_multiply_by_quantized_mult(result, out_mult, out_shift);
        result += out_offset;
        result = max(result, activation_min);
        result = min(result, activation_max);
        out_data[out_c] = (int8_t) result;
    }
}

void esp_nn_fully_connected_per_ch_s8_ansi(const int8_t *input_data,
                                    const int32_t input_offset,
                                    const uint16_t row_len,
                                    const int8_t *filter_data,
                                    const int32_t filter_offset,
                                    const int32_t *bias,
                                    int8_t *out_data,
                                    const uint16_t out_channels,
                                    const int32_t out_offset,
                                    const int32_t* out_shift,
                                    const int32_t* out_mult,
                                    const int32_t activation_min,
                                    const int32_t activation_max)
{
    for (int32_t out_c = 0; out_c < out_channels; ++out_c) {
        int32_t result = 0;
        for (int32_t data_idx = 0; data_idx < row_len; data_idx++) {
            int32_t filter_index = row_len * out_c + data_idx;
            int32_t input_val = input_data[data_idx];
            int32_t filter_val = filter_data[filter_index];
            result += (filter_val + filter_offset) * (input_val + input_offset);
        }
        if (bias) {
            result += bias[out_c];
        }
        result = esp_nn_multiply_by_quantized_mult(result, out_mult[out_c], out_shift[out_c]);
        result += out_offset;
        result = max(result, activation_min);
        result = min(result, activation_max);
        out_data[out_c] = (int8_t) result;
    }
}

/* Fully connected scratch. Separate from the convolution scratch pointer.
 * The batched call clears both fields before it returns. */
static int8_t *fc_scratch;
static size_t fc_scratch_size;

void esp_nn_set_fc_scratch_buf(const void *buf, size_t size)
{
    fc_scratch = (int8_t *) buf;
    fc_scratch_size = (buf != NULL) ? size : 0;
}

/* The 1x1 scratch formulas do not read the offsets or the activation, so
 * those fields are fixed. The batch count is the convolution height: S3
 * drops its transpose buffer below 8 positions, and PIE pads the input when
 * the channel count is not a multiple of 8. */
int esp_nn_get_fc_batch_scratch_size(int32_t batches, uint16_t row_len,
                                     uint16_t out_ch)
{
    if (batches <= 0 || batches > UINT16_MAX || row_len == 0 || out_ch == 0) {
        return 0;
    }
    const data_dims_t input_dims = {
        .width = 1, .height = batches, .channels = row_len,
    };
    const data_dims_t filter_dims = {
        .width = 1, .height = 1, .channels = row_len,
    };
    const data_dims_t output_dims = {
        .width = 1, .height = batches, .channels = out_ch,
    };
    const conv_params_t conv_params = {
        .in_offset = 0,
        .out_offset = 0,
        .stride = { .width = 1, .height = 1 },
        .padding = { .width = 0, .height = 0 },
        .dilation = { .width = 1, .height = 1 },
        .activation = { .min = -128, .max = 127 },
    };
    return esp_nn_get_conv_scratch_size(&input_dims, &filter_dims,
                                        &output_dims, &conv_params);
}

void esp_nn_fc_scratch_clear(void)
{
    fc_scratch = NULL;
    fc_scratch_size = 0;
}

/* A registered buffer is usable when this 1x1 shape needs workspace, the
 * pointer is 16-byte aligned, and the recorded length covers that workspace.
 * A zero need is the ANSI convolution, which was not swept, so it stays off
 * this path. */
static int fc_scratch_fits(int bytes)
{
    if (bytes <= 0 || fc_scratch == NULL) {
        return 0;
    }
    return (((uintptr_t) fc_scratch) & 15) == 0 &&
            fc_scratch_size >= (size_t) bytes;
}

/* A per-channel batch with weight zero-point 0 is a 1x1 convolution:
 * height = batches, width = 1, in_ch = row_len, out_ch = out_channels.
 * Convolution supplies the second core when its own split applies, so
 * esp_nn_dual_core_enable() still has to have been called. The input
 * zero-point may be nonzero.
 *
 * Correctness: a nonzero weight zero-point stays on fully connected because
 * convolution has no weight-zero-point term. Batches above UINT16_MAX stay
 * there too: the S3 and PIE kernels store height in uint16_t, and this path
 * maps batches to that height. The input and filter pointers must be 8-byte
 * aligned. A fully connected scratch buffer that is missing, not 16-byte
 * aligned, or shorter than esp_nn_get_conv_scratch_size for this 1x1 shape
 * also stays on fully connected. The convolution scratch pointer is not
 * read. For the duration of the convolution call, that pointer is set to
 * the fully connected buffer and then restored.
 *
 * Performance: fc_conv_shape_wins is the per-chip floor from the board
 * sweeps. The S3 fast assembly still requires channels % 8 inside the
 * convolution kernel and uses its existing fallback for other widths.
 * Batch 1 and 2 require that multiple of 8 here as well. A live worker
 * is also required, and the convolution can still decline the split and
 * finish on one core. */
int esp_nn_fc_per_ch_s8_batch_via_conv(const int8_t *input_data,
                                    const int32_t input_offset,
                                    const uint16_t row_len,
                                    const int8_t *filter_data,
                                    const int32_t filter_offset,
                                    const int32_t *bias,
                                    int8_t *out_data,
                                    const uint16_t out_channels,
                                    const int32_t out_offset,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t activation_min,
                                    const int32_t activation_max,
                                    const int32_t batches)
{
    if (batches > UINT16_MAX ||
            (((uintptr_t) input_data) & 7) != 0 ||
            (((uintptr_t) filter_data) & 7) != 0 ||
            !fc_conv_shape_wins(batches, row_len, out_channels)) {
        return 0;
    }
#if !defined(CONFIG_NN_OPTIMIZED) || \
        !(defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S31) || \
          defined(CONFIG_IDF_TARGET_ESP32S3))
    /* The 1x1 kernel has no weight-zero-point term except on P4, S31, and
     * S3, where the missing per-row product is added before requantize. */
    if (filter_offset != 0) {
        return 0;
    }
#endif

    const data_dims_t input_dims = {
        .width = 1, .height = batches, .channels = row_len,
    };
    const data_dims_t filter_dims = {
        .width = 1, .height = 1, .channels = row_len,
    };
    const data_dims_t output_dims = {
        .width = 1, .height = batches, .channels = out_channels,
    };
    const conv_params_t conv_params = {
        .in_offset = input_offset,
        .out_offset = out_offset,
        .stride = { .width = 1, .height = 1 },
        .padding = { .width = 0, .height = 0 },
        .dilation = { .width = 1, .height = 1 },
        .activation = { .min = activation_min, .max = activation_max },
    };
    const int scratch_need = esp_nn_get_conv_scratch_size(
            &input_dims, &filter_dims, &output_dims, &conv_params);
    /* Weight zero-point folds into a per-row term stored after the
     * convolution workspace: filter_offset * (sum(input) + K * input_offset). */
    const int addend_at = (scratch_need + 15) & ~15;
    const int scratch_bytes = filter_offset == 0
            ? scratch_need
            : addend_at + batches * (int) sizeof(int32_t);
    if (!fc_scratch_fits(scratch_bytes)) {
        return 0;
    }
    if (filter_offset != 0) {
        int32_t *addend = (int32_t *) (fc_scratch + addend_at);
        for (int32_t n = 0; n < batches; n++) {
            int32_t sum = 0;
            const int8_t *row = input_data + (size_t) n * row_len;
            for (uint16_t c = 0; c < row_len; c++) {
                sum += row[c];
            }
            const int64_t term = (int64_t) filter_offset *
                    ((int64_t) sum + (int64_t) row_len * input_offset);
            if (term > INT32_MAX || term < INT32_MIN) {
                return 0;
            }
            addend[n] = (int32_t) term;
        }
#if defined(CONFIG_NN_OPTIMIZED) && \
        (defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S31))
        esp_nn_conv_set_row_addend_riscv_pie(addend);
#elif defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)
        esp_nn_conv_set_row_addend_esp32s3(addend);
#else
        (void) addend;
        return 0;
#endif
    }
    quant_data_t quant = {
        .shift = (int32_t *) out_shift,
        .mult = (int32_t *) out_mult,
    };
    void *saved_scratch = NULL;
    void *saved_preferred = NULL;
    size_t saved_preferred_size = 0;
    esp_nn_conv_scratch_borrow(fc_scratch, &saved_scratch, &saved_preferred,
                               &saved_preferred_size);
    esp_nn_conv_s8(&input_dims, input_data, &filter_dims, filter_data, bias,
                   &output_dims, out_data, &conv_params, &quant);
    esp_nn_conv_scratch_restore(saved_scratch, saved_preferred,
                                saved_preferred_size);
#if defined(CONFIG_NN_OPTIMIZED) && \
        (defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S31))
    esp_nn_conv_set_row_addend_riscv_pie(NULL);
#elif defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)
    esp_nn_conv_set_row_addend_esp32s3(NULL);
#endif
    return 1;
}

/* Portable reference for the batched per-channel FC API: loop the per-row
 * kernel. Optimized targets override the macro with a weight-stationary impl.
 * A qualifying zero-point-0 batch is the 1x1 convolution above. */
void esp_nn_fully_connected_per_ch_s8_batch_ansi(const int8_t *input_data,
                                    const int32_t input_offset,
                                    const uint16_t row_len,
                                    const int8_t *filter_data,
                                    const int32_t filter_offset,
                                    const int32_t *bias,
                                    int8_t *out_data,
                                    const uint16_t out_channels,
                                    const int32_t out_offset,
                                    const int32_t* out_shift,
                                    const int32_t* out_mult,
                                    const int32_t activation_min,
                                    const int32_t activation_max,
                                    const int32_t batches)
{
    if (esp_nn_fc_per_ch_s8_batch_via_conv(input_data, input_offset, row_len,
                                           filter_data, filter_offset, bias,
                                           out_data, out_channels, out_offset,
                                           out_shift, out_mult, activation_min,
                                           activation_max, batches)) {
        esp_nn_fc_scratch_clear();
        return;
    }
    for (int32_t b = 0; b < batches; b++) {
        esp_nn_fully_connected_per_ch_s8_ansi(input_data + b * row_len, input_offset,
                                    row_len, filter_data, filter_offset, bias,
                                    out_data + b * out_channels, out_channels,
                                    out_offset, out_shift, out_mult,
                                    activation_min, activation_max);
    }
    esp_nn_fc_scratch_clear();
}
