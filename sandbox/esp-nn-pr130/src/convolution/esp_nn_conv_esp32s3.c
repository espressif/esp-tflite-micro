/*
 * SPDX-FileCopyrightText: 2020-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Optimizations strategies used:
 * Below optimizations are capable of any size of input/filter:
 *
 * 1. For filter wdxht = 1x1 (Refer esp_nn_conv_s8_mult8_1x1_esp32s3 function)
 *      - For this specific version, the strategy we employ:
 *          > This particular filter has only the channel
 *              dimension and we have `out_ch` number of such filters.
 *          > We take 8 input lines at a time and transpose those.
 *          > Keep loading and multiplying filter values one by one,
 *              to produce 8 outputs in parallel
 *
 * 2. General version: (Refer esp_nn_conv_s8_filter_aligned_input_padded_esp32s3)
 *      - For all other cases:
 *          > Consider `filter_wd * in_ch` as a single row. These many values can
 *              be continuosly loaded from inputs as well.
 *          > multiply accumulate into a single filter output.
 *          > To speed things up further, we pre-calculate
 *              (filter * in_offset + bias term) earlier and add it at the end of filter
 *
 *      About ((filter * in_offset + bias term)) accumulate term:
 *          > The conv operation before requantization is as follows:
 *              for i in filter_size:
 *                  conv_out += (input + input_offset) * filter;
 *               conv_out += bias
 *
 *          > where input_offset is constant term hence, we can see that
 *              this term can be precalculated as:
 *                  for i in filter_size:
 *                      acc_term += input_offset * filter[i];
 *                  acc_term += bias
 *              OR
 *                   for i in filter_size:
 *                      acc_term += filter[i]; // accumulate filter values
 *                  acc_term = acc_term * input_offset + bias
 *
 *
 * In both the above versions we align the filter if needed, pad the input with
 *       -input_offset if needed and extend the channels to make those multiple
 *       of 8/16 as per function needs
 *
 * 3. Im2col version: (for small in_ch where filter_wd * in_ch < 16)
 *      - Inspired by ESP32-P4 im2col approach.
 *      - Instead of padding channels (wastes 81% of SIMD lanes for in_ch=3),
 *        flatten the entire filter window into one contiguous vector:
 *          window_len = filter_wd * filter_ht * in_ch (e.g., 3*3*3 = 27)
 *      - For each output pixel: copy the input window into a scratch buffer,
 *        then use ACCX dot product on the full window. No wasted MACs.
 */

#include <stdio.h>
#include "../common/esp_nn_filter_sum_esp32s3.h"
#include "esp_nn_conv_1x1_panel_esp32s3.h"
#include <esp_nn_multicore.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <esp_nn_defs.h>

#include <common_functions.h>

/* 3x3 optimized path — im2col per pixel, iterate OC with input in cache */
extern int esp_nn_conv_s8_3x3_can_use(int filter_wd, int filter_ht, int in_channels, int out_channels);
extern int esp_nn_conv_s8_3x3_scratch_size(int in_channels, int out_channels,
                                           int out_wd, int out_ht);
extern void esp_nn_conv_s8_3x3_opt(const int8_t *input,
    const uint16_t input_wd, const uint16_t input_ht,
    const uint16_t in_channels, const int32_t input_offset,
                             const uint16_t pad_wd,
                             const uint16_t pad_ht,
    const uint16_t stride_wd, const uint16_t stride_ht,
    const int8_t *filter_data, const int32_t *bias,
    int8_t *out_data, const uint16_t out_wd, const uint16_t out_ht,
    const uint16_t out_channels, const int32_t out_offset,
    const int32_t *out_shift, const int32_t *out_mult,
    const int32_t activation_min, const int32_t activation_max,
    void *scratch);

/* ANSI C reference conv for comparison */
extern void esp_nn_conv_s8_ansi(const data_dims_t *input_dims,
                                const int8_t *input_data,
                                const data_dims_t *filter_dims,
                                const int8_t *filter_data,
                                const int32_t *bias,
                                const data_dims_t *output_dims,
                                int8_t *out_data,
                                const conv_params_t *conv_params,
                                const quant_data_t *quant_data);

/* 1x1 conv — correct SIMD implementation */
extern int esp_nn_conv_s8_1x1_scratch_size(int in_channels);
extern void esp_nn_conv_s8_1x1(const int8_t *input,
                                const uint16_t input_wd,
                                const uint16_t input_ht,
                                const uint16_t in_channels,
                                const int32_t input_offset,
                                const int8_t *filter_data,
                                const int32_t *bias,
                                int8_t *out_data,
                                const uint16_t out_channels,
                                const int32_t out_offset,
                                const int32_t *out_shift,
                                const int32_t *out_mult,
                                const int32_t activation_min,
                                const int32_t activation_max,
                                void *scratch,
                                const int32_t *row_addend);

/* Debug heap checks - opt-in: a full heap walk per conv call is far too
 * expensive (and too fragile) to ship enabled. */
#ifdef ESP_NN_DEBUG_HEAP_CHECK
#include "esp_heap_caps.h"
#define CONV_HEAP_CHECK(tag) do { \
    if (!heap_caps_check_integrity_all(false)) { \
        printf("CONV HEAP CORRUPT: %s\n", tag); \
    } \
} while(0)
#else
#define CONV_HEAP_CHECK(tag)
#endif

static int16_t *scratch_buffer = NULL;
static uint8_t *preferred_scratch_buffer = NULL;
static size_t preferred_scratch_size = 0;
/* Per-row accumulator term for a fully connected weight zero-point.
 * NULL on every other convolution, including a zero weight zero-point. */
static const int32_t *fc_row_addend = NULL;

void esp_nn_conv_set_row_addend_esp32s3(const int32_t *row_addend)
{
    fc_row_addend = row_addend;
}

extern void esp_nn_conv_s8_mult8_1x1_esp32s3(
                const int8_t *input_data,
                const uint16_t input_wd,
                const uint16_t input_ht,
                const uint16_t in_channels,
                const int32_t input_offset,
                const int8_t *filter_aligned,
                const int32_t *bias,
                int8_t *out_data,
                const uint16_t out_wd,
                const uint16_t out_ht,
                const uint16_t out_channels,
                const int32_t out_offset,
                const int32_t *out_shift,
                const int32_t *out_mult,
                const int32_t activation_min,
                const int32_t activation_max,
                void *buffer, /* scratch buffer */
                const int32_t *row_addend);

extern void esp_nn_conv_s8_filter_aligned_input_padded_esp32s3(
                const int8_t *input_data,
                const uint16_t input_wd,
                const uint16_t input_ht,
                const uint16_t in_channels,
                const int32_t input_offset,
                const uint16_t stride_wd,
                const uint16_t stride_ht,
                const int8_t *filter_data,
                const uint16_t filter_wd,
                const uint16_t filter_ht,
                const int32_t *bias,
                int8_t *out_data,
                const uint16_t out_wd,
                const uint16_t out_ht,
                const uint16_t out_channels,
                const int32_t out_offset,
                const int32_t *out_shift,
                const int32_t *out_mult,
                const int32_t activation_min,
                const int32_t activation_max,
                                void *scratch_buffer);

/*
 * The assembly kernel handles spatial positions in groups of eight, but its
 * scalar remainder path walks the complete filter once per leftover position.
 * For large pointwise filters that makes a 15-position tensor read the weights
 * eight times (one vector group plus seven scalar passes).  Pad the remainder
 * to one vector group instead, so the weights are streamed only once.
 */
static void esp_nn_conv_s8_mult8_1x1_batched_tail(
        const int8_t *input, uint16_t input_wd, uint16_t input_ht,
        uint16_t in_channels, int32_t input_offset,
        const int8_t *filter_data, const int32_t *bias,
        int8_t *out_data, uint16_t out_channels, int32_t out_offset,
        const int32_t *out_shift, const int32_t *out_mult,
        int32_t activation_min, int32_t activation_max, void *scratch,
        const int32_t *row_addend)
{
    const int spatial_size = input_wd * input_ht;
    const int tail = spatial_size & 7;

    if (tail < 2) {
        esp_nn_conv_s8_mult8_1x1_esp32s3(
                input, input_wd, input_ht, in_channels, input_offset,
                filter_data, bias, out_data, input_wd, input_ht, out_channels,
                out_offset, out_shift, out_mult, activation_min, activation_max,
                scratch, row_addend);
        return;
    }

    uint8_t *work = (uint8_t *)(((uintptr_t)scratch + 15) & ~(uintptr_t)15);
    int8_t *tail_input = (int8_t *)(work + 16 * in_channels);
    int8_t *tail_output = tail_input + 8 * in_channels;
    const int vector_positions = spatial_size - tail;

    if (vector_positions != 0) {
        esp_nn_conv_s8_mult8_1x1_esp32s3(
                input, vector_positions, 1, in_channels, input_offset,
                filter_data, bias, out_data, vector_positions, 1, out_channels,
                out_offset, out_shift, out_mult, activation_min, activation_max,
                work, row_addend);
    }

    memcpy(tail_input, input + vector_positions * in_channels,
           tail * in_channels);
    memset(tail_input + tail * in_channels, (int8_t)-input_offset,
           (8 - tail) * in_channels);
    /* The padded positions are discarded. Their addend is zero so the
     * assembly load of eight values stays inside this array. */
    int32_t tail_addend_buf[8] __attribute__((aligned(16)));
    const int32_t *tail_addend = NULL;
    if (row_addend != NULL) {
        for (int i = 0; i < tail; i++) {
            tail_addend_buf[i] = row_addend[vector_positions + i];
        }
        for (int i = tail; i < 8; i++) {
            tail_addend_buf[i] = 0;
        }
        tail_addend = tail_addend_buf;
    }
    esp_nn_conv_s8_mult8_1x1_esp32s3(
            tail_input, 8, 1, in_channels, input_offset,
            filter_data, bias, tail_output, 8, 1, out_channels,
            out_offset, out_shift, out_mult, activation_min, activation_max,
            work, tail_addend);
    memcpy(out_data + vector_positions * out_channels, tail_output,
           tail * out_channels);
}

/*
 * GEBP-style OC-panel driver for the mult8 1x1 asm. The asm streams the
 * whole filter matrix once per 8-position batch, which thrashes the 64 KB
 * L1 D-cache whenever the filter exceeds it. Output channels are therefore
 * processed in panels sized to ~24 KB of filter rows: a panel stays
 * resident across every position batch, so filter data is fetched from
 * memory once per layer while the batched SIMD kernel does the math.
 * Each panel's output is staged contiguous and scattered into the NHWC
 * layout. Per-output-channel arithmetic is independent and unchanged:
 * results are bit-identical to a single batched call.
 */
void esp_nn_conv_s8_mult8_1x1_oc_panel(
        const int8_t *input, int spatial_size, uint16_t in_channels,
        int32_t input_offset, const int8_t *filter_data, const int32_t *bias,
        int8_t *out_data, uint16_t out_channels, uint16_t out_stride,
        int32_t out_offset, const int32_t *out_shift, const int32_t *out_mult,
        int32_t activation_min, int32_t activation_max, void *scratch,
        const int32_t *row_addend)
{
    int oc_tile = ESP_NN_S3_PANEL_BYTES / in_channels;
    oc_tile &= ~7;
    /* Every panel pass re-reads the staged positions. For wide windows the
     * byte budget alone leaves a panel of one or two channels, so the
     * re-reads cost more than the residency they buy: keep at least 16
     * channels per panel there. Shapes whose budget already allows more are
     * untouched, and a larger panel measured slower for them. */
    if (oc_tile < 16) {
        oc_tile = 16;
    }
    if (oc_tile > out_channels) {
        oc_tile = out_channels;
    }
    if (oc_tile < 8) {
        oc_tile = 8;
    }
    if (oc_tile > 1024) {
        oc_tile = 1024;
    }
    if (oc_tile > out_channels) {
        oc_tile = out_channels;
    }

    uint8_t *work = (uint8_t *)(((uintptr_t)scratch + 15) & ~(uintptr_t)15);
    if (oc_tile == out_channels && out_stride == out_channels) {
        /* Whole filter fits one panel: plain batched call, no staging. */
        esp_nn_conv_s8_mult8_1x1_batched_tail(
                input, spatial_size, 1, in_channels, input_offset,
                filter_data, bias, out_data, out_channels, out_offset,
                out_shift, out_mult, activation_min, activation_max, work,
                row_addend);
        return;
    }

    /* Staging area sits after the asm work region (transpose + tails). */
    int8_t *staging = (int8_t *)(work + 24 * in_channels + 8 * oc_tile + 32);
    int pos_chunk = (8 * 1024) / oc_tile;
    pos_chunk &= ~7;
    if (pos_chunk < 8) {
        pos_chunk = 8;
    }

    for (int oc_base = 0; oc_base < out_channels; oc_base += oc_tile) {
        const int oc_n = min(oc_tile, out_channels - oc_base);
        for (int pos0 = 0; pos0 < spatial_size; pos0 += pos_chunk) {
            const int pos_n = min(pos_chunk, spatial_size - pos0);
            esp_nn_conv_s8_mult8_1x1_batched_tail(
                    input + pos0 * in_channels, pos_n, 1, in_channels,
                    input_offset, filter_data + oc_base * in_channels,
                    bias ? bias + oc_base : NULL, staging, oc_n,
                    out_offset, out_shift + oc_base, out_mult + oc_base,
                    activation_min, activation_max, work,
                    row_addend == NULL ? NULL : row_addend + pos0);
            const int8_t *src = staging;
            int8_t *dst = out_data + pos0 * out_stride + oc_base;
            for (int p = 0; p < pos_n; p++) {
                memcpy(dst, src, oc_n);
                src += oc_n;
                dst += out_stride;
            }
        }
    }
}

#if ESP_NN_DUAL_CORE_SUPPORTED
typedef struct {
    const int8_t *input;
    int spatial_size;
    int in_channels;
    int32_t input_offset;
    const int8_t *filter_data;
    const int32_t *bias;
    int8_t *out_data;
    int out_channels;
    int out_stride;
    int32_t out_offset;
    const int32_t *out_shift;
    const int32_t *out_mult;
    int32_t activation_min;
    int32_t activation_max;
    uint8_t *copy_scratch;
    size_t copy_scratch_size;
    const int32_t *row_addend;
} conv_1x1_panel_mt_job_t;

static void conv_1x1_panel_mt_worker(void *p)
{
    const conv_1x1_panel_mt_job_t *j = (const conv_1x1_panel_mt_job_t *)p;
    esp_nn_conv_s8_mult8_1x1_oc_panel(
            j->input, j->spatial_size, j->in_channels, j->input_offset,
            j->filter_data, j->bias, j->out_data, j->out_channels,
            j->out_stride, j->out_offset, j->out_shift, j->out_mult,
            j->activation_min, j->activation_max, j->copy_scratch,
            j->row_addend);
}
#endif /* ESP_NN_DUAL_CORE_SUPPORTED */

/* Use shared dot product from common — see esp_nn_dot_s8_esp32s3.S */

/**
 * Im2col convolution for small in_ch (filter_wd * in_ch < 16).
 *
 * Instead of padding channels to 16 (wasting 81% MACs for in_ch=3),
 * flatten the entire filter window into one contiguous vector:
 *   window_len = filter_wd * filter_ht * in_ch (e.g., 3*3*3 = 27)
 *
 * For each output pixel: copy the input window into a contiguous scratch
 * buffer, then use ACCX dot product. No wasted MACs.
 *
 * Scratch layout: [filter_sum[out_ch] | im2col_buf[window_len_aligned]]
 */
/* Floor division for a positive divisor (C '/' truncates towards zero). */
static inline int32_t esp_nn_floor_div_s32(int32_t num, int32_t den)
{
    int32_t q = num / den;
    return (num % den != 0 && num < 0) ? q - 1 : q;
}

__attribute__ ((noinline))
static void esp_nn_conv_s8_im2col_s3(
        const data_dims_t *input_dims,
        const int8_t *input_data,
        const data_dims_t *filter_dims,
        const int8_t *filter_data,
        const int32_t *bias,
        const data_dims_t *output_dims,
        int8_t *out_data,
        const conv_params_t *conv_params,
        const quant_data_t *quant_data,
        void *scratch)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_ch = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_ch = output_dims->channels;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    const int32_t window_len = filter_wd * filter_ht * in_ch;
    /* Align to 16 for SIMD: zero-padded tail doesn't affect dot product */
    const int32_t window_len_aligned = (window_len + 15) & ~15;
    const int8_t pad_val = (int8_t)(-input_offset);

    /* Scratch layout (16-byte aligned):
     * [filter_sum: out_ch * 4]
     * [aligned_filter: out_ch * window_len_aligned]  -- zero-padded copy
     * [im2col_buf: window_len_aligned]
     */
    int32_t *filter_sum = (int32_t *)scratch;
    int8_t *aligned_filter = (int8_t *)((uintptr_t)((int8_t *)scratch + out_ch * sizeof(int32_t) + 15) & ~15);
    int8_t *im2col_buf = (int8_t *)((uintptr_t)(aligned_filter + out_ch * window_len_aligned + 15) & ~15);

    /* Pre-compute filter_sum * input_offset AND copy filter with zero-padded tail */
    const int8_t *fptr = filter_data;
    int8_t *af_ptr = aligned_filter;
    for (int32_t oc = 0; oc < out_ch; oc++) {
        int32_t sum = 0;
        for (int32_t fi = 0; fi < window_len; fi++) {
            sum += fptr[fi];
        }
        filter_sum[oc] = sum * input_offset;
        /* Copy filter + zero-pad tail for safe SIMD reads */
        memcpy(af_ptr, fptr, window_len);
        memset(af_ptr + window_len, 0, window_len_aligned - window_len);
        fptr += window_len;
        af_ptr += window_len_aligned;
    }

    /* Zero the tail of im2col buffer once (for aligned SIMD reads) */
    memset(im2col_buf + window_len, 0, window_len_aligned - window_len);

    /* Compute safe interior region where no bounds checking needed.
     * Interior: all filter taps fall within valid input. */
    const int32_t row_bytes = filter_wd * in_ch;
    int32_t safe_y_start = (pad_ht + stride_ht - 1) / stride_ht;
    /* floor, not C truncation: the numerator is negative when an input side
     * is smaller than the filter, and truncation would mark a row/column as
     * interior whose taps read past the end of the input. */
    int32_t safe_y_end = esp_nn_floor_div_s32(input_ht - filter_ht + pad_ht, stride_ht) + 1;
    int32_t safe_x_start = (pad_wd + stride_wd - 1) / stride_wd;
    int32_t safe_x_end = esp_nn_floor_div_s32(input_wd - filter_wd + pad_wd, stride_wd) + 1;
    if (safe_y_start > out_ht) safe_y_start = out_ht;
    if (safe_y_end > out_ht) safe_y_end = out_ht;
    if (safe_y_end < safe_y_start) safe_y_end = safe_y_start;
    if (safe_x_start > out_wd) safe_x_start = out_wd;
    if (safe_x_end > out_wd) safe_x_end = out_wd;
    if (safe_x_end < safe_x_start) safe_x_end = safe_x_start;

    /* Process each output pixel */
    int8_t *out_ptr = out_data;
    for (int32_t out_y = 0; out_y < out_ht; out_y++) {
        const int32_t base_y = out_y * stride_ht - pad_ht;
        int is_safe_y = (out_y >= safe_y_start && out_y < safe_y_end);

        for (int32_t out_x = 0; out_x < out_wd; out_x++) {
            const int32_t base_x = out_x * stride_wd - pad_wd;

            /* Copy input window into contiguous im2col buffer */
            int8_t *buf = im2col_buf;

            if (is_safe_y && out_x >= safe_x_start && out_x < safe_x_end) {
                /* FAST PATH: interior pixel — no bounds checking needed.
                 * All filter taps guaranteed to be within valid input. */
                for (int32_t fy = 0; fy < filter_ht; fy++) {
                    const int8_t *src = input_data + ((base_y + fy) * input_wd + base_x) * in_ch;
                    memcpy(buf, src, row_bytes);
                    buf += row_bytes;
                }
            } else {
                /* SLOW PATH: edge pixel — per-element bounds checking */
                for (int32_t fy = 0; fy < filter_ht; fy++) {
                    int32_t in_y = base_y + fy;
                    if (in_y >= 0 && in_y < input_ht) {
                        for (int32_t fx = 0; fx < filter_wd; fx++) {
                            int32_t in_x = base_x + fx;
                            if (in_x >= 0 && in_x < input_wd) {
                                const int8_t *src = input_data + (in_y * input_wd + in_x) * in_ch;
                                memcpy(buf, src, in_ch);
                            } else {
                                memset(buf, pad_val, in_ch);
                            }
                            buf += in_ch;
                        }
                    } else {
                        memset(buf, pad_val, row_bytes);
                        buf += row_bytes;
                    }
                }
            }

            /* Dot product against each output channel's filter (aligned copy) */
            const int32_t *out_mult_ptr = quant_data->mult;
            const int32_t *out_shift_ptr = quant_data->shift;
            const int8_t *filter_ptr = aligned_filter;

            for (int32_t oc = 0; oc < out_ch; oc++) {
                int32_t conv_out = esp_nn_dot_s8_aligned_esp32s3(im2col_buf, filter_ptr, window_len_aligned);
                conv_out += filter_sum[oc];
                if (bias) conv_out += bias[oc];
                conv_out = esp_nn_requantize(conv_out, *out_mult_ptr++, *out_shift_ptr++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_ptr++ = (int8_t) conv_out;
                filter_ptr += window_len_aligned;
            }
        }
    }
}

int esp_nn_get_conv_scratch_size_esp32s3(const data_dims_t *input_dims,
                                         const data_dims_t *filter_dims,
                                         const data_dims_t *output_dims,
                                         const conv_params_t *conv_params)
{
    /* Grouped conv runs each group through the standard path with repacked
     * slices: inner requirement (per-group dims) + input slice + output
     * staging. */
    if (filter_dims->channels && filter_dims->channels < input_dims->channels &&
            input_dims->channels % filter_dims->channels == 0) {
        const int32_t groups_ = input_dims->channels / filter_dims->channels;
        if (groups_ > 1 && output_dims->channels % groups_ == 0) {
            data_dims_t in_g_ = *input_dims;
            data_dims_t out_g_ = *output_dims;
            in_g_.channels = filter_dims->channels;
            out_g_.channels = output_dims->channels / groups_;
            const int inner_ = esp_nn_get_conv_scratch_size_esp32s3(&in_g_, filter_dims, &out_g_, conv_params);
            const int32_t in_slice_ = (int32_t)input_dims->width * input_dims->height * filter_dims->channels;
            const int32_t out_slice_ = (int32_t)output_dims->width * output_dims->height * out_g_.channels;
            return inner_ + in_slice_ + out_slice_ + 64;
        }
    }

    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_ch = input_dims->channels;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_ch = output_dims->channels;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;

    /* Mirrors the runtime dispatch: the 3x3 path takes precedence when it
     * qualifies, and needs im2col + an aligned zero-padded filter copy +
     * corrections. */
    if (esp_nn_conv_s8_3x3_can_use(filter_wd, filter_ht, in_ch, out_ch)) {
        return esp_nn_conv_s8_3x3_scratch_size(in_ch, out_ch,
                                               output_dims->width,
                                               output_dims->height);
    }

    int new_channels = (in_ch + 7) & ~7;

    int input_scratch = input_wd * input_ht * in_ch;
    int filter_scratch = filter_wd * filter_ht * in_ch * out_ch;

    int align_buf_size = 64; /* alignment (16) + assembly pre/post access margin (48) */
    if ((filter_wd == 1 && filter_ht == 1 && pad_wd == 0 && pad_ht == 0) &&
            (stride_wd == 1 && stride_ht == 1)) {
        /* Transpose buffer is used by both 1x1 kernels; the filter is not
         * copied by either, so no filter term. */
        int transpose_buf_size = 2 * (8 * new_channels);
        if (input_wd * input_ht < 8) {
            transpose_buf_size = 0;
        }
        /* Neither 1x1 kernel copies or pads the input: the SIMD path
         * transposes into the buffer above, the fallback reads in place
         * (any alignment, any channel count). No input term. */
        int existing_size = transpose_buf_size + align_buf_size;
        int batched_tail_size = 16 * in_ch + 8 * in_ch + 8 * out_ch +
                                align_buf_size;
        /* OC-panel driver (filter > 24 KB): asm work + staging areas */
        int panel_size = 24 * in_ch + 16 * 1024 + align_buf_size;
        if ((int32_t)in_ch * out_ch <= 24 * 1024) {
            panel_size = 0;
        }
        return max(max(existing_size, batched_tail_size), panel_size);
    } else {
        int32_t filter_row_size = filter_wd * in_ch;
        int32_t window_len = filter_wd * filter_ht * in_ch;

        /* Im2col path: filter_wd * in_ch < 16 but window_len >= 16 */
        if (filter_row_size < 16 && window_len >= 16) {
            int32_t window_len_aligned = (window_len + 15) & ~15;
            /* filter_sum + aligned_filter_copy + im2col_buf + alignment padding */
            int im2col_scratch = out_ch * 4 + 16 + out_ch * window_len_aligned + 16 + window_len_aligned;
            return im2col_scratch + align_buf_size;
        }

        new_channels = (in_ch + 15) & ~15;

        // Padded-input scratch: leading (top/left) padding from TFLite plus the
        // trailing (bottom/right) padding derived from the output extent.
        // Must match the padding applied in esp_nn_conv_s8_esp32s3().
        int pad_right = max(0, (output_dims->width - 1) * stride_wd + filter_wd - pad_wd - input_wd);
        int pad_bottom = max(0, (output_dims->height - 1) * stride_ht + filter_ht - pad_ht - input_ht);
        if (pad_wd == 0 && pad_ht == 0 && pad_right == 0 && pad_bottom == 0) {
            input_scratch = 0;
        } else {
            input_scratch = (input_wd + pad_wd + pad_right) * (input_ht + pad_ht + pad_bottom) * in_ch;
        }
        /* At most one of the two filter copies is ever made, so max(), not
         * the sum. */
        int32_t aligned_filter_row_size = ((filter_row_size + 15) / 16) * 16;
        int row_padded_copy = aligned_filter_row_size * filter_ht * out_ch;
        int pointer_align_copy = filter_wd * filter_ht * in_ch * out_ch;
        filter_scratch = max(row_padded_copy, pointer_align_copy);

        int offset_acc_scratch = out_ch * 4;
        return input_scratch + filter_scratch + align_buf_size + offset_acc_scratch;
    }
    return align_buf_size;
}

void esp_nn_set_conv_scratch_buf_esp32s3(void *buf)
{
    scratch_buffer = (int16_t *) buf;
}

void esp_nn_conv_scratch_borrow_esp32s3(void *buf, void **scratch,
                                        void **preferred, size_t *preferred_size)
{
    *scratch = scratch_buffer;
    *preferred = preferred_scratch_buffer;
    *preferred_size = preferred_scratch_size;
    scratch_buffer = (int16_t *) buf;
    /* The 1x1 kernel writes the preferred buffer when it is large enough.
     * Suspend it so this call writes the borrowed fully connected buffer,
     * then the caller restores the preferred registration. */
    preferred_scratch_buffer = NULL;
    preferred_scratch_size = 0;
}

void esp_nn_conv_scratch_restore_esp32s3(void *scratch, void *preferred,
                                         size_t preferred_size)
{
    scratch_buffer = (int16_t *) scratch;
    preferred_scratch_buffer = (uint8_t *) preferred;
    preferred_scratch_size = preferred_size;
}

void esp_nn_set_conv_preferred_scratch_buf_esp32s3(void *buf, size_t size)
{
    preferred_scratch_buffer = (uint8_t *)buf;
    preferred_scratch_size = size;
}

void esp_nn_conv_s8_esp32s3(const data_dims_t *input_dims,
                            const int8_t *input,
                            const data_dims_t *filter_dims,
                            const int8_t *filter_data,
                            const int32_t *bias,
                            const data_dims_t *output_dims,
                            int8_t *out_data,
                            const conv_params_t *conv_params,
                            const quant_data_t *quant_data)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_channels = output_dims->channels;
    const int32_t *out_shift = quant_data->shift;
    const int32_t *out_mult = quant_data->mult;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    /* Grouped conv (filter_ch < input_ch): run each group through the
     * optimized path via repacked slices; reference row-split otherwise. */
    if (channels != filter_dims->channels) {
        data_dims_t in_g = *input_dims;
        data_dims_t out_g = *output_dims;
        const int32_t groups = filter_dims->channels
                ? channels / filter_dims->channels : 0;
        if (groups > 1 && output_dims->channels % groups == 0 &&
                channels % filter_dims->channels == 0) {
            in_g.channels = filter_dims->channels;
            out_g.channels = output_dims->channels / groups;
            const int inner = esp_nn_get_conv_scratch_size_esp32s3(
                    &in_g, filter_dims, &out_g, conv_params);
            const int total = esp_nn_get_conv_scratch_size_esp32s3(
                    input_dims, filter_dims, output_dims, conv_params);
            if (esp_nn_conv_s8_grouped_repack(
                    esp_nn_conv_s8_esp32s3, input_dims, input,
                    filter_dims, filter_data, bias, output_dims, out_data,
                    conv_params, quant_data, scratch_buffer, total, inner)) {
                return;
            }
        }
        esp_nn_conv_s8_ansi_mt_split(input_dims, input, filter_dims, filter_data,
                                     bias, output_dims, out_data, conv_params, quant_data);
        return;
    }

    int filter_size = filter_wd * filter_ht * channels * out_channels;

    /* 1x1 stride-1 conv */
    if (filter_wd == 1 && filter_ht == 1 && pad_wd == 0 && pad_ht == 0 &&
            stride_wd == 1 && stride_ht == 1) {
        /* The mult8 asm loads the input and stores the output 8 bytes at a
         * time, so both must be 8-byte aligned. */
        if (channels % 8 == 0 &&
                (((uintptr_t)input | (uintptr_t)out_data) & 7) == 0) {
            int spatial_size = input_wd * input_ht;
            int filter_bytes = channels * out_channels;
            if (filter_bytes > 24 * 1024) {
                /* Panel work + staging (24 * in_ch + 16 KB + 64) goes to the
                 * preferred (internal) scratch when it fits, like the batched
                 * path below; otherwise a PSRAM arena would host the very
                 * buffers the L1-resident scheme depends on. */
                void *panel_scratch = scratch_buffer;
                if (preferred_scratch_buffer != NULL &&
                        (((uintptr_t)preferred_scratch_buffer & 15) == 0) &&
                        (size_t)(24 * channels + 16 * 1024 + 64) <= preferred_scratch_size) {
                    panel_scratch = preferred_scratch_buffer;
                }
                /* Filter exceeds the L1 panel budget: the OC-panel driver keeps
                 * each panel of filter rows hot across all position batches. */
                /* Cores split the output channels; outputs are disjoint,
                 * arithmetic unchanged. */
#if ESP_NN_DUAL_CORE_SUPPORTED
                if (esp_nn_dual_core_active() && out_channels >= 32) {
                    const int oc0 = ((out_channels / 2) + 7) & ~7;
                    const int wneed = 24 * channels + 16 * 1024 + 64;
                    uint8_t *wscr = (uint8_t *)esp_nn_dual_core_scratch(wneed);
                    if (wscr != NULL && oc0 > 0 && oc0 < out_channels) {
                        conv_1x1_panel_mt_job_t job = {
                            .input = input, .spatial_size = spatial_size,
                            .in_channels = channels, .input_offset = input_offset,
                            .filter_data = filter_data, .bias = bias,
                            .out_data = out_data, .out_channels = oc0,
                            .out_stride = out_channels,
                            .out_offset = out_offset, .out_shift = out_shift,
                            .out_mult = out_mult, .activation_min = activation_min,
                            .activation_max = activation_max,
                            .copy_scratch = wscr, .copy_scratch_size = (size_t)wneed,
                            .row_addend = fc_row_addend,
                        };
                        if (esp_nn_dual_core_run(conv_1x1_panel_mt_worker, &job)) {
                            esp_nn_conv_s8_mult8_1x1_oc_panel(
                                    input, spatial_size, channels, input_offset,
                                    filter_data + oc0 * channels,
                                    bias ? bias + oc0 : NULL,
                                    out_data + oc0, out_channels - oc0, out_channels,
                                    out_offset, out_shift + oc0, out_mult + oc0,
                                    activation_min, activation_max, panel_scratch,
                                    fc_row_addend);
                            esp_nn_dual_core_wait();
                            return;
                        }
                        /* worker declined (nested or same-core call): single-core below */
                    }
                }
#endif /* ESP_NN_DUAL_CORE_SUPPORTED */
                esp_nn_conv_s8_mult8_1x1_oc_panel(
                        input, spatial_size, channels, input_offset,
                        filter_data, bias, out_data, out_channels,
                        out_channels, out_offset, out_shift, out_mult,
                        activation_min, activation_max, panel_scratch,
                        fc_row_addend);
                return;
            }
            void *pointwise_scratch = scratch_buffer;
            int required = 24 * channels + 8 * out_channels + 16;
            if (preferred_scratch_buffer != NULL &&
                    (((uintptr_t)preferred_scratch_buffer & 15) == 0) &&
                    required <= preferred_scratch_size) {
                pointwise_scratch = preferred_scratch_buffer;
            }
            /* Full asm path — requires mult8 channels + 8-byte aligned filter */
            esp_nn_conv_s8_mult8_1x1_batched_tail(
                    input, input_wd, input_ht, channels, input_offset,
                    filter_data, bias, out_data, out_channels, out_offset,
                    out_shift, out_mult, activation_min, activation_max,
                    pointwise_scratch, fc_row_addend);
        } else {
            /* Fallback: handles any alignment + any channel count */
            esp_nn_conv_s8_1x1(input, input_wd, input_ht, channels, input_offset,
                               filter_data, bias, out_data, out_channels, out_offset,
                               out_shift, out_mult, activation_min, activation_max,
                               scratch_buffer, fc_row_addend);
        }
        return;
    }

    if (scratch_buffer == NULL) {
        printf("esp_nn_conv error! scratch_buffer not set!\n");
        return;
    }

    {
        int32_t filter_row_size = filter_wd * channels;
        int32_t window_len = filter_wd * filter_ht * channels;

        /* 3x3 optimized path: im2col per pixel, iterate OC with the input in
         * cache - avoids the general asm's per-output-channel input reload.
         * Rewritten around the aligned dot: one aligned zero-padded filter
         * copy per layer, then both operands are plain 16-byte-aligned
         * vectors. The previous inline-asm version MAC'd two uninitialized
         * q-registers on the first iteration of every pixel. */
        if (esp_nn_conv_s8_3x3_can_use(filter_wd, filter_ht, channels, out_channels)) {
            esp_nn_conv_s8_3x3_opt(input, input_wd, input_ht, channels,
                                    input_offset, pad_wd, pad_ht, stride_wd, stride_ht,
                                    filter_data, bias, out_data,
                                    out_wd, out_ht, out_channels, out_offset,
                                    out_shift, out_mult, activation_min, activation_max,
                                    (void *)scratch_buffer);
            return;
        }


        /* Im2col path: small in_ch where per-row SIMD is wasteful,
         * but entire window is large enough for SIMD dot product.
         * E.g., 3x3 conv with in_ch=3: row=9 (<16), window=27 (>=16). */
        if (filter_row_size < 16 && window_len >= 16) {
            esp_nn_conv_s8_im2col_s3(input_dims, input, filter_dims, filter_data,
                                      bias, output_dims, out_data, conv_params,
                                      quant_data, scratch_buffer);
            return;
        }

        // align the `filter width * channels` to 16 bytes. Do zero padding for the same
        int32_t filter_alignment_padding = 16 - (filter_row_size & 15);
        int8_t *filter_data_aligned = (int8_t *) filter_data;
        int8_t *input_padded = (int8_t *) input;
        int8_t *scratch_data = (int8_t *) scratch_buffer;
        int new_input_wd = input_wd, new_input_ht = input_ht;
        if (filter_alignment_padding != 16) {
            // pad filter_data
            int32_t new_row_size = filter_wd * channels + filter_alignment_padding;
            filter_data_aligned = scratch_data;
            int8_t *row_ptr = filter_data_aligned;
            const int8_t *filter_data_ptr = filter_data;
            for (int32_t ch_idx = 0; ch_idx < out_channels; ch_idx++) {
                for (int32_t row_idx = 0; row_idx < filter_ht; row_idx++) {
                    memcpy(row_ptr, filter_data_ptr, filter_row_size);
                    memset(row_ptr + filter_row_size, 0, new_row_size - filter_row_size);
                    filter_data_ptr += filter_row_size;
                    row_ptr += new_row_size;
                }
            }
            scratch_data += new_row_size * filter_ht * out_channels;
            filter_row_size = new_row_size;
        } else if ((int) filter_data & 15) {
            filter_data_aligned = scratch_data;
            memcpy(filter_data_aligned, filter_data, filter_size);
            scratch_data += filter_size;
        }
        // pad_wd/pad_ht carry only the leading (top/left) padding as passed by
        // TFLite. Derive the trailing (bottom/right) padding from the output
        // extent: it differs from the leading one whenever the total "SAME"
        // padding is odd, and is also needed when pad_wd/pad_ht are 0.
        int32_t pad_right = max(0, (out_wd - 1) * stride_wd + filter_wd - pad_wd - input_wd);
        int32_t pad_bottom = max(0, (out_ht - 1) * stride_ht + filter_ht - pad_ht - input_ht);

        if (pad_wd != 0 || pad_ht != 0 || pad_right > 0 || pad_bottom > 0) {
            input_padded = (int8_t *) scratch_data;
            esp_nn_aligned_s8_pad_asymmetric(input, input_padded, input_wd, input_ht, channels,
                                             -input_offset, pad_wd, pad_ht,
                                             (uint16_t) pad_right, (uint16_t) pad_bottom);
            new_input_wd = input_wd + pad_wd + pad_right;
            new_input_ht = input_ht + pad_ht + pad_bottom;
            scratch_data += new_input_wd * new_input_ht * channels;
        }


        int filter_total = filter_wd * filter_ht * channels * out_channels;
        if (input_offset != 0 && filter_total > 16384) {
            int32_t *corrections = (int32_t *)scratch_data;
            int32_t filter_ch_size = filter_wd * filter_ht * channels;
            const int8_t *f_src = filter_data; // use ORIGINAL (not aligned) filter for sum
            for (int ch = 0; ch < out_channels; ch++) {
                /* esp-nn#36: this was a scalar reduction the same algorithmic
                 * size as the conv itself, recomputed every call. */
                int32_t filter_sum = esp_nn_filter_sum_s8_esp32s3(f_src, filter_ch_size);
                corrections[ch] = filter_sum * input_offset;
                if (bias) {
                    corrections[ch] += bias[ch];
                }
                f_src += filter_ch_size;
            }
            // Pass input_offset=0 to assembly so it skips its pre-computation.
            // Pass scratch_data as "bias" pointer — the assembly's bias-copy loop
            // will read from scratch and write to scratch (identity, no-op).
            esp_nn_conv_s8_filter_aligned_input_padded_esp32s3(
                input_padded, new_input_wd, new_input_ht, channels, 0,
                stride_wd, stride_ht, filter_data_aligned, filter_wd, filter_ht,
                (const int32_t *)scratch_data, out_data, out_wd, out_ht, out_channels,
                out_offset, out_shift, out_mult, activation_min, activation_max,
                scratch_data);
            CONV_HEAP_CHECK("general: after asm (precomp)");
        } else {
            esp_nn_conv_s8_filter_aligned_input_padded_esp32s3(
                input_padded, new_input_wd, new_input_ht, channels, input_offset,
                stride_wd, stride_ht, filter_data_aligned, filter_wd, filter_ht,
                bias, out_data, out_wd, out_ht, out_channels, out_offset,
                out_shift, out_mult, activation_min, activation_max, scratch_data);
            CONV_HEAP_CHECK("general: after asm (normal)");
        }
    }
}
