/*
 * SPDX-FileCopyrightText: 2024-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Optimizations strategies used:
 * Below optimizations are capable of any size of input/filter:
 *
 * 1. For filter wdxht = 1x1 (Refer esp_nn_conv_s8_mult8_1x1_riscv_pie function)
 *      - For this specific version, the strategy we employ:
 *          > This particular filter has only the channel
 *              dimension and we have `out_ch` number of such filters.
 *          > We take 8 input lines at a time and transpose those.
 *          > Keep loading and multiplying filter values one by one,
 *              to produce 8 outputs in parallel
 *
 * 2. General version: (Refer esp_nn_conv_s8_filter_aligned_input_padded_riscv_pie)
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
 */

#include <stddef.h>
#include <stdio.h>
#include <esp_nn_defs.h>
#include <esp_nn_ansi_headers.h>
#include "esp_nn_generic_opt.h"

#include <common_functions.h>
#include "../common/esp_nn_filter_sum_riscv_pie.h"
#include <esp_nn_multicore.h>

int esp_nn_get_conv_scratch_size_riscv_pie(const data_dims_t *input_dims,
                                           const data_dims_t *filter_dims,
                                           const data_dims_t *output_dims,
                                           const conv_params_t *conv_params);
void esp_nn_conv_s8_riscv_pie(const data_dims_t *input_dims,
                              const int8_t *input,
                              const data_dims_t *filter_dims,
                              const int8_t *filter_data,
                              const int32_t *bias,
                              const data_dims_t *output_dims,
                              int8_t *out_data,
                              const conv_params_t *conv_params,
                              const quant_data_t *quant_data);

static int16_t *scratch_buffer = NULL;
/* Per-pixel accumulator addend for a fully connected weight zero-point.
 * NULL on the weight-zero-point-0 path. Indexed by the pixel this call owns. */
static const int32_t *fc_row_addend = NULL;

void esp_nn_conv_set_row_addend_riscv_pie(const int32_t *row_addend)
{
    fc_row_addend = row_addend;
}

/*
 * XACC dot of `rows` filter rows against strided input rows, double-buffered
 * like pie_dot_s8 (q0/q2 + q1/q3) so vld results are consumed three
 * instructions later instead of one - the single-buffered per-row loop
 * stalled on every load-use pair. Requires row_size a multiple of 16 and
 * >= 32. Filter rows are contiguous; input rows are in_stride apart.
 * Integer accumulation, same values in a reordered sum: bit-identical.
 */
static int32_t conv_dot_rows_pie(const int8_t *in, const int8_t *filt,
                                 int32_t rows, int32_t row_size,
                                 int32_t in_stride)
{
    const int32_t c32 = (row_size >> 5) - 1;
    const int32_t rem16 = row_size & 16;
    int32_t result;
    asm volatile ("esp.zero.xacc\n\t");
    for (int32_t r = 0; r < rows; r++) {
        asm volatile (
            "mv     x30, %[inp]                     \n\t"
            "mv     x31, %[flt]                     \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q2, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vld.128.ip  q3, x31, 16            \n\t"
            "beqz   %[c32], 2f                      \n\t"
            /* zero-overhead loop; end label ON last body insn */
            "esp.lp.setup 0, %[c32], 1f             \n\t"
            "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
            "1:                                     \n\t"
            "esp.vld.128.ip  q3, x31, 16            \n\t"
            "2:                                     \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "esp.vmulas.s8.xacc  q2, q3             \n\t"
            "beqz   %[rem], 3f                      \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "3:                                     \n\t"
            :
            : [inp] "r"(in + r * in_stride), [flt] "r"(filt + r * row_size),
              [c32] "r"(c32), [rem] "r"(rem16)
            : "x30", "x31"
        );
    }
    asm volatile (
        "esp.movx.r.xacc.l  x30   \n\t"
        "mv %0, x30               \n\t"
        : "=r" (result)
        :
        : "x30"
    );
    return result;
}


/* Sum of `len` int8 values via the PIE dot against an all-ones vector.
 * Integer addition is associative, so the result is bit-identical to a
 * scalar loop; used to vectorize the filter_sum * input_offset prepasses. */
/* Sum via the shared PIE dot-against-ones helper (see the header for why
 * this must not be re-derived privately). */
static int32_t conv_filter_byte_sum(const int8_t *data, int32_t len)
{
    return esp_nn_filter_sum_s8_riscv_pie(data, len);
}



/**
 * Reusable PIE-accelerated dot product (same as FC version).
 * Processes 32 elements/iter (double-pump) for len >= 32,
 * 16 elements/iter for len >= 16, scalar remainder.
 */
static inline __attribute__((always_inline))
int32_t pie_dot_s8(const int8_t *a, const int8_t *b, int32_t len)
{
    int32_t result = 0;

    if (len >= 32) {
        int32_t c32 = (len >> 5) - 1;
        int32_t rem16 = len & 16;
        asm volatile (
            "esp.zero.xacc                          \n\t"
            "mv     x30, %[in]                      \n\t"
            "mv     x31, %[flt]                     \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q2, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vld.128.ip  q3, x31, 16            \n\t"
            "beqz   %[c32], 2f                      \n\t"
            /* zero-overhead loop; end label ON last body insn */
            "esp.lp.setup 0, %[c32], 1f             \n\t"
            "esp.vmulas.s8.xacc.ld.ip q0, x30, 16, q0, q1 \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc.ld.ip q2, x30, 16, q2, q3 \n\t"
            "1:                                     \n\t"
            "esp.vld.128.ip  q3, x31, 16            \n\t"
            "2:                                     \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "esp.vmulas.s8.xacc  q2, q3             \n\t"
            "beqz   %[rem16], 3f                    \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "3:                                     \n\t"
            "esp.movx.r.xacc.l   x30                \n\t"
            "mv     %[res], x30                     \n\t"
            : [res] "=r"(result)
            : [in] "r"(a), [flt] "r"(b), [c32] "r"(c32), [rem16] "r"(rem16)
            : "x30", "x31"
        );
    } else if (len >= 16) {
        /* exactly one full 16-element block */
        asm volatile (
            "esp.zero.xacc                          \n\t"
            "mv     x30, %[in]                      \n\t"
            "mv     x31, %[flt]                     \n\t"
            "esp.vld.128.ip  q0, x30, 16            \n\t"
            "esp.vld.128.ip  q1, x31, 16            \n\t"
            "esp.vmulas.s8.xacc  q0, q1             \n\t"
            "esp.movx.r.xacc.l   x30                \n\t"
            "mv     %[res], x30                     \n\t"
            : [res] "=r"(result)
            : [in] "r"(a), [flt] "r"(b)
            : "x30", "x31"
        );
    }

    for (int32_t idx = len & ~15; idx < len; idx++) {
        result += (int32_t)a[idx] * (int32_t)b[idx];
    }
    return result;
}

/* Large pointwise filters do not fit in the target's data cache. The usual
 * pixel-major traversal walks the complete filter once per output pixel,
 * repeatedly fetching the same weights from flash. For small spatial maps,
 * keep one output-channel filter row hot and apply it to every pixel before
 * advancing to the next row. Output writes become strided, but the avoided
 * weight traffic is much larger (for example 15x for a 3x5 output map). */
__attribute__((noinline))
static void conv_1x1_filter_major(const data_dims_t *input_dims,
                                  const int8_t *input_data,
                                  const int8_t *filter_data,
                                  const int32_t *bias,
                                  const data_dims_t *output_dims,
                                  int8_t *out_data,
                                  const conv_params_t *conv_params,
                                  const quant_data_t *quant_data)
{
    const int32_t in_channels = input_dims->channels;
    const int32_t out_channels = output_dims->channels;
    const int32_t pixels = output_dims->width * output_dims->height;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t output_offset = conv_params->out_offset;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    for (int32_t out_ch = 0; out_ch < out_channels; ++out_ch) {
        const int8_t *filter = filter_data + out_ch * in_channels;
        int32_t filter_sum = 0;
        if (input_offset != 0) {
            /* Vectorized like the other prepasses. It matters more here: this
             * path is gated on a small spatial map, so the per-channel sum is
             * a sizeable fraction of the work, not a negligible prepass. */
            filter_sum = conv_filter_byte_sum(filter, in_channels) * input_offset;
        }
        const int32_t base = filter_sum + (bias ? bias[out_ch] : 0);
        const int32_t multiplier = quant_data->mult[out_ch];
        const int32_t shift = quant_data->shift[out_ch];

        for (int32_t pixel = 0; pixel < pixels; ++pixel) {
            const int8_t *input = input_data + pixel * in_channels;
            int32_t result = pie_dot_s8(input, filter, in_channels) + base;
            result = esp_nn_requantize(result, multiplier, shift);
            result += output_offset;
            result = max(result, activation_min);
            result = min(result, activation_max);
            out_data[pixel * out_channels + out_ch] = (int8_t) result;
        }
    }
}

/**
 * Batched 1x1 conv using QACC per-lane: processes 16 pixels simultaneously.
 * Transposes input so each QACC lane = one pixel, then broadcasts filter
 * coefficients for per-lane accumulation. Critical for small in_ch where
 * XACC can't be used (in_ch < 16).
 *
 * For in_ch=8: 4.5x faster than scalar per-pixel approach.
 */
__attribute__((noinline))
static void conv_1x1_batch16(const int8_t *pixel_ptrs[16],
                      const int8_t *filter_data,
                      const int32_t *filter_sum,
                      const int32_t *bias,
                      int8_t *out_ptrs[16],
                      int32_t in_ch, int32_t out_ch,
                      int32_t out_offset,
                      const int32_t *out_mult, const int32_t *out_shift,
                      int32_t act_min, int32_t act_max,
                      const int32_t *row_addend)
{
    /* Transpose: arrange 16 pixels' data as ch0[p0..p15], ch1[p0..p15], ... */
    int8_t transposed[16 * 16] __attribute__((aligned(16)));  /* in_ch <= 16 for this path */
    for (int c = 0; c < in_ch; c++) {
        for (int p = 0; p < 16; p++) {
            transposed[c * 16 + p] = pixel_ptrs[p][c];
        }
    }

    /* For each output channel: QACC per-lane MAC with broadcast filter.
     * Use single asm block for zero + accumulate loop to prevent
     * q register clobber between separate asm blocks. */
    const int8_t *filt = filter_data;
    for (int32_t oc = 0; oc < out_ch; oc++) {
        /* QACC accumulate over in_ch. Hardware loop (esp.lp.setup) is deliberately
         * not used - trip count (< 16) is too small to amortize its arming cost;
         * software-pipelined instead: esp.vmulas.s8.qacc.ld.ip folds the next pixel
         * load into the MAC, and esp.vldbc.8.xp advances the filter pointer in the
         * load (no addi, assembler-safe). Prologue loads pair 0; a tail MAC finishes
         * the last pair. ~4.06 vs 6.0 cyc/iter on S31. */
        asm volatile (
            "li     t3, 1                        \n\t"  /* filter byte-stride */
            "esp.zero.qacc                       \n\t"
            "mv     x30, %[trans]                \n\t"  /* transposed base */
            "mv     x31, %[flt]                  \n\t"  /* filter base */
            "mv     s7,  %[cnt]                  \n\t"  /* in_ch count */
            "esp.vld.128.ip  q0, x30, 16         \n\t"  /* prologue: pixel[0] */
            "esp.vldbc.8.xp  q1, x31, t3         \n\t"  /* prologue: filter[0], x31 += 1 */
            "addi   s7, s7, -1                   \n\t"  /* fused-loop trip = in_ch - 1 */
            "beqz   s7, 2f                       \n\t"  /* in_ch == 1: straight to tail */
            "1:                                  \n\t"
            "esp.vmulas.s8.qacc.ld.ip q0, x30, 16, q0, q1 \n\t"  /* mac pair; load next pixel */
            "esp.vldbc.8.xp  q1, x31, t3         \n\t"            /* load next filter, x31 += 1 */
            "addi   s7, s7, -1                   \n\t"
            "bnez   s7, 1b                       \n\t"
            "2:                                  \n\t"
            "esp.vmulas.s8.qacc q0, q1           \n\t"            /* tail: last pair */
            :
            : [trans] "r"(transposed), [flt] "r"(filt), [cnt] "r"(in_ch)
            : "x30", "x31", "s7", "t3"
        );

        /* Extract 16 results */
        int32_t results[16] __attribute__((aligned(16)));
        ESP_NN_QACC_EXTRACT_S32(results);

        /* Add filter_sum + bias, requant, clamp, store for each pixel */
        int32_t fs = filter_sum[oc];
        int32_t b = bias ? bias[oc] : 0;
        int32_t combined = fs + b;
        int32_t m = out_mult[oc];
        int32_t s = out_shift[oc];

        for (int p = 0; p < 16; p++) {
            int32_t r = results[p] + combined;
            if (row_addend != NULL) {
                r += row_addend[p];
            }
            r = esp_nn_multiply_by_quantized_mult(r, m, s);
            r += out_offset;
            r = max(r, act_min);
            r = min(r, act_max);
            out_ptrs[p][oc] = (int8_t) r;
        }

        filt += in_ch;
    }
}



/* MAC continuation for the woven pass: same double-buffered accumulate as
 * the plain loop but WITHOUT zeroing the QACC (partial sums are live). */
static void qacc16_mac_continue(const int8_t *trans, const int8_t *filt,
                                int32_t cnt)
{
    if (cnt & 1) {
        /* Odd tap first: the paired loop below loads one pair ahead and
         * would otherwise read one chunk past the end of trans/filt. */
        asm volatile (
            "li     t3, 1                        \n\t"
            "mv     x30, %[trans]                \n\t"
            "mv     x31, %[flt]                  \n\t"
            "esp.vld.128.ip  q0, x30, 16         \n\t"
            "esp.vldbc.8.xp  q1, x31, t3         \n\t"
            "esp.vmulas.s8.qacc q0, q1           \n\t"
            :
            : [trans] "r"(trans), [flt] "r"(filt)
            : "x30", "x31", "t3"
        );
        trans += 16;
        filt += 1;
        cnt -= 1;
        if (cnt == 0) {
            return;
        }
    }
    /* cnt is even and >= 2: pairs = cnt / 2, the last pair is MAC'd after
     * the loop so no load runs past the end. */
    asm volatile (
        "li     t3, 1                        \n\t"
        "mv     x30, %[trans]                \n\t"
        "mv     x31, %[flt]                  \n\t"
        "esp.vld.128.ip  q0, x30, 16         \n\t"
        "esp.vldbc.8.xp  q1, x31, t3         \n\t"
        "esp.vld.128.ip  q2, x30, 16         \n\t"
        "esp.vldbc.8.xp  q3, x31, t3         \n\t"
        "beqz   %[cpair], 2f                 \n\t"
        "esp.lp.setup 0, %[cpair], 1f        \n\t"
        "esp.vmulas.s8.qacc.ld.ip q0, x30, 16, q0, q1 \n\t"
        "esp.vldbc.8.xp  q1, x31, t3         \n\t"
        "esp.vmulas.s8.qacc.ld.ip q2, x30, 16, q2, q3 \n\t"
        "1:                                  \n\t"
        "esp.vldbc.8.xp  q3, x31, t3         \n\t"
        "2:                                  \n\t"
        "esp.vmulas.s8.qacc q0, q1           \n\t"
        "esp.vmulas.s8.qacc q2, q3           \n\t"
        :
        : [trans] "r"(trans), [flt] "r"(filt),
          [cpair] "r"(cnt / 2 - 1)
        : "x30", "x31", "t3"
    );
}

/*
 * Software-pipelined variant of the deep 16-pixel QACC pass: while the
 * current output channel's MACs stream through the PIE slot, the PREVIOUS
 * channel's 16 results are requantized in the scalar slot (the core
 * dual-issues one PIE and one scalar op per cycle, and PIE loads/MACs have
 * multi-cycle latency the scalar stream hides in).
 *
 * The requant uses the doubling-high-multiply with an unconditional
 * +2^30 nudge; a 200M-trial host sweep shows it bit-identical to
 * esp_nn_multiply_by_quantized_mult for |acc + combined| < 2^27 and
 * shift <= 0, which the caller guarantees before selecting this path.
 *
 * Consumes exactly 8 * min(in_ch / 8, 16) input channels and requantizes
 * min(in_ch / 8, 16) elements of prev_res; the caller MACs the remaining
 * channels and drains the remaining elements. Requires in_ch % 8 == 0.
 */
static int32_t qacc16_mac_woven_requant(const int8_t *trans,
                                        const int8_t *filt, int32_t in_ch,
                                        const int32_t *prev_res,
                                        int32_t combined, int32_t mult,
                                        int32_t rs, int32_t out_offset,
                                        int32_t act_min, int32_t act_max,
                                        int8_t *prev_out, int32_t out_stride)
{
    int32_t woven = in_ch >> 3;          /* iterations: 8 channels each */
    if (woven > 16) {
        woven = 16;
    }
    const int32_t rnd = rs ? (1 << (rs - 1)) : 0;
    const int32_t rsnz = rs ? 1 : 0;
    const int32_t *rp = prev_res;
    int8_t *op = prev_out;
    int32_t n = woven;
    asm volatile (
        "li     t3, 1                          \n\t"
        "mv     x30, %[tp]                     \n\t"
        "mv     x31, %[fp]                     \n\t"
        "1:                                    \n\t"
        /* group 1 (channels k, k+1) + requant head */
        "esp.vld.128.ip  q0, x30, 16           \n\t"
        "lw     t0, 0(%[rp])                   \n\t"
        "esp.vldbc.8.xp  q1, x31, t3           \n\t"
        "add    t0, t0, %[comb]                \n\t"
        "esp.vld.128.ip  q2, x30, 16           \n\t"
        "mulh   t1, t0, %[mult]                \n\t"
        "esp.vldbc.8.xp  q3, x31, t3           \n\t"
        "mul    t0, t0, %[mult]                \n\t"
        "esp.vmulas.s8.qacc q0, q1             \n\t"
        "add    t2, t0, %[nud]                 \n\t"
        "esp.vmulas.s8.qacc q2, q3             \n\t"
        "sltu   t0, t2, t0                     \n\t"
        /* group 2 (channels k+2, k+3) + doubling-high compose */
        "esp.vld.128.ip  q0, x30, 16           \n\t"
        "add    t1, t1, t0                     \n\t"
        "esp.vldbc.8.xp  q1, x31, t3           \n\t"
        "slli   t1, t1, 1                      \n\t"
        "esp.vld.128.ip  q2, x30, 16           \n\t"
        "srli   t2, t2, 31                     \n\t"
        "esp.vldbc.8.xp  q3, x31, t3           \n\t"
        "or     t1, t1, t2                     \n\t"
        "esp.vmulas.s8.qacc q0, q1             \n\t"
        "srli   t0, t1, 31                     \n\t"
        "esp.vmulas.s8.qacc q2, q3             \n\t"
        "and    t0, t0, %[rsnz]                \n\t"
        /* group 3 (channels k+4, k+5) + rounding shift */
        "esp.vld.128.ip  q0, x30, 16           \n\t"
        "add    t1, t1, %[rnd]                 \n\t"
        "esp.vldbc.8.xp  q1, x31, t3           \n\t"
        "sub    t1, t1, t0                     \n\t"
        "esp.vld.128.ip  q2, x30, 16           \n\t"
        "sra    t1, t1, %[rs]                  \n\t"
        "esp.vldbc.8.xp  q3, x31, t3           \n\t"
        "add    t1, t1, %[ooff]                \n\t"
        "esp.vmulas.s8.qacc q0, q1             \n\t"
        "blt    t1, %[amax], 2f                \n\t"
        "mv     t1, %[amax]                    \n\t"
        "2:                                    \n\t"
        "esp.vmulas.s8.qacc q2, q3             \n\t"
        "blt    %[amin], t1, 3f                \n\t"
        "mv     t1, %[amin]                    \n\t"
        "3:                                    \n\t"
        /* group 4 (channels k+6, k+7) + store/advance */
        "esp.vld.128.ip  q0, x30, 16           \n\t"
        "sb     t1, 0(%[op])                   \n\t"
        "esp.vldbc.8.xp  q1, x31, t3           \n\t"
        "addi   %[rp], %[rp], 4                \n\t"
        "esp.vld.128.ip  q2, x30, 16           \n\t"
        "add    %[op], %[op], %[ostr]          \n\t"
        "esp.vldbc.8.xp  q3, x31, t3           \n\t"
        "addi   %[n], %[n], -1                 \n\t"
        "esp.vmulas.s8.qacc q0, q1             \n\t"
        "esp.vmulas.s8.qacc q2, q3             \n\t"
        "bnez   %[n], 1b                       \n\t"
        : [rp] "+r"(rp), [op] "+r"(op), [n] "+r"(n)
        : [tp] "r"(trans), [fp] "r"(filt), [comb] "r"(combined),
          [mult] "r"(mult), [nud] "r"(0x40000000), [rnd] "r"(rnd),
          [rsnz] "r"(rsnz), [rs] "r"(rs), [ooff] "r"(out_offset),
          [amin] "r"(act_min), [amax] "r"(act_max), [ostr] "r"(out_stride)
        : "x30", "x31", "t0", "t1", "t2", "t3", "memory"
    );
    return woven;
}


/*
 * Position-batched 1x1 for in_ch >= 16: 16 consecutive pixels are
 * transposed to channel-major in scratch, then each output channel runs
 * one QACC pass (16 lanes = 16 pixels, broadcast filter byte per input
 * channel). Two instructions per input channel move 16 MACs, and the
 * per-pixel dot setup of the XACC path disappears. Accumulation order per
 * output element (input channels ascending) and the scalar requant are
 * unchanged: bit-identical results.
 */

/*
 * Runs output channels [oc_start, oc_end) of a 16-pixel channel-major
 * staging: one QACC pass per channel (dot length `count`), requantize,
 * scatter to out_base[pixel * out_stride + oc]. Pipelines the previous
 * channel's requant into the current channel's MAC stream when the fast
 * requant is provably exact (see qacc16_mac_woven_requant); otherwise a
 * plain double-buffered pass per channel.
 */
static void qacc16_run_channels(const int8_t *t, const int8_t *filt_base,
                                int32_t count, int32_t oc_start,
                                int32_t oc_end, const int32_t *filter_sum,
                                const int32_t *bias, const int32_t *out_mult,
                                const int32_t *out_shift, int8_t *out_base,
                                int32_t out_stride, int32_t out_offset,
                                int32_t act_min, int32_t act_max,
                                int32_t lanes, const int32_t *row_addend)
{
    const int8_t *filt = filt_base;

    /* The woven requant writes all 16 lanes, so partial batches take the
     * plain path (which stores `lanes` results). */
    if (row_addend == NULL && lanes == 16 && (count & 7) == 0 && count >= 128) {
        bool woven_ok = true;
        const int32_t acc_bound = count * 127 * 128;
        for (int32_t oc = oc_start; oc < oc_end; oc++) {
            const int32_t comb = filter_sum[oc] + (bias ? bias[oc] : 0);
            const int32_t acomb = comb < 0 ? -comb : comb;
            if (out_shift[oc] > 0 || acc_bound + acomb >= (1 << 27)) {
                woven_ok = false;
                break;
            }
        }
        if (woven_ok) {
            int32_t resA[16] __attribute__((aligned(16)));
            int32_t resB[16] __attribute__((aligned(16)));
            int32_t *prev = NULL, *cur = resA;
            int32_t prev_oc = -1;
            for (int32_t oc = oc_start; oc < oc_end; oc++) {
                asm volatile ("esp.zero.qacc\n\t");
                if (prev == NULL) {
                    qacc16_mac_continue(t, filt, count);
                } else {
                    const int32_t comb = filter_sum[prev_oc]
                            + (bias ? bias[prev_oc] : 0);
                    const int32_t w = qacc16_mac_woven_requant(
                            t, filt, count, prev, comb,
                            out_mult[prev_oc], -out_shift[prev_oc],
                            out_offset, act_min, act_max,
                            out_base + prev_oc, out_stride);
                    if (8 * w < count) {
                        qacc16_mac_continue(t + 8 * w * 16, filt + 8 * w,
                                            count - 8 * w);
                    }
                    for (int32_t e = w; e < 16; e++) {
                        int32_t r = prev[e] + comb;
                        r = esp_nn_requantize(r, out_mult[prev_oc],
                                              out_shift[prev_oc]);
                        r += out_offset;
                        r = max(r, act_min);
                        r = min(r, act_max);
                        out_base[e * out_stride + prev_oc] = (int8_t) r;
                    }
                }
                ESP_NN_QACC_EXTRACT_S32(cur);
                prev = cur;
                cur = (cur == resA) ? resB : resA;
                prev_oc = oc;
                filt += count;
            }
            const int32_t comb = filter_sum[prev_oc]
                    + (bias ? bias[prev_oc] : 0);
            for (int32_t e = 0; e < 16; e++) {
                int32_t r = prev[e] + comb;
                r = esp_nn_requantize(r, out_mult[prev_oc],
                                      out_shift[prev_oc]);
                r += out_offset;
                r = max(r, act_min);
                r = min(r, act_max);
                out_base[e * out_stride + prev_oc] = (int8_t) r;
            }
            return;
        }
    }

    for (int32_t oc = oc_start; oc < oc_end; oc++) {
        asm volatile ("esp.zero.qacc\n\t");
        qacc16_mac_continue(t, filt, count);
        int32_t results[16] __attribute__((aligned(16)));
        ESP_NN_QACC_EXTRACT_S32(results);
        const int32_t combined = filter_sum[oc] + (bias ? bias[oc] : 0);
        const int32_t m = out_mult[oc];
        const int32_t sh = out_shift[oc];
        int8_t *out_ptr = out_base + oc;
        for (int p = 0; p < lanes; p++) {
            int32_t r = results[p] + combined;
            if (row_addend != NULL) {
                r += row_addend[p];
            }
            r = esp_nn_requantize(r, m, sh);
            r += out_offset;
            r = max(r, act_min);
            r = min(r, act_max);
            out_ptr[p * out_stride] = (int8_t) r;
        }
        filt += count;
    }
}

static void conv_1x1_batch16_deep(const int8_t *input, /* 16*in_ch bytes */
                                  const int8_t *filter_data,
                                  const int32_t *filter_sum,
                                  const int32_t *bias,
                                  int8_t *out_data, /* 16 pixels, stride out_stride */
                                  int32_t in_ch,
                                  int32_t oc_base, int32_t oc_end,
                                  int32_t out_stride,
                                  int32_t out_offset,
                                  const int32_t *out_mult,
                                  const int32_t *out_shift,
                                  int32_t act_min, int32_t act_max,
                                  int8_t *transposed /* in_ch*16, 16-aligned */,
                                  const int32_t *row_addend)
{
    /* Transpose 16 pixels to channel-major: t[c][p] = in[p][c]. */
    for (int32_t c = 0; c < in_ch; c++) {
        int8_t *t = transposed + c * 16;
        const int8_t *src = input + c;
        for (int p = 0; p < 16; p++) {
            t[p] = src[p * in_ch];
        }
    }

    qacc16_run_channels(transposed, filter_data + oc_base * in_ch, in_ch,
                        oc_base, oc_end, filter_sum, bias, out_mult,
                        out_shift, out_data, out_stride, out_offset,
                        act_min, act_max, 16, row_addend);
}

__attribute__ ((noinline))
static void esp_nn_conv_s8_1x1(const data_dims_t *input_dims,
                               const int8_t *input_data,
                               const int8_t *filter_data,
                               const int32_t *bias,
                               const data_dims_t *output_dims,
                               int8_t *out_data,
                               const conv_params_t *conv_params,
                               const quant_data_t *quant_data,
                               void *scratch,
                               const int32_t *row_addend)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t in_channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_channels = output_dims->channels;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    const int32_t filter_bytes = in_channels * out_channels;
    const int32_t output_pixels = out_wd * out_ht;
    if (row_addend == NULL && filter_bytes >= 64 * 1024 && output_pixels <= 16) {
        conv_1x1_filter_major(input_dims, input_data, filter_data, bias,
                              output_dims, out_data, conv_params, quant_data);
        return;
    }

    int32_t *filter_sum = (int32_t *) scratch; // alignment of 4 bytes assumed

    /* pre-calculate filter_sum * input_offset */
    const int8_t *filter_ptr = filter_data;
    for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
        filter_sum[out_ch_idx] = conv_filter_byte_sum(
                filter_data + out_ch_idx * in_channels, in_channels) * input_offset;
    }

    /* When in_ch < 16: use QACC batch path (16 pixels at once) or channel padding.
     * QACC batch: transpose pixels, broadcast filter, per-lane MAC.
     * Channel pad: pad in/filter to 16 ch for XACC. */
    /* When in_ch < 16: use QACC batch (16 pixels at a time with broadcast filter).
     * Falls back to channel-padding for remaining pixels. */
    if (in_channels < 16) {
        int32_t total_pixels = out_wd * out_ht;
        int32_t pix = 0;

        /* Process batches of 16 pixels using QACC per-lane */
        for (; pix <= total_pixels - 16; pix += 16) {
            const int8_t *pp[16];
            int8_t *op[16];
            for (int p = 0; p < 16; p++) {
                pp[p] = input_data + (pix + p) * in_channels;
                op[p] = out_data + (pix + p) * out_channels;
            }
            conv_1x1_batch16(pp, filter_data, filter_sum, bias, op,
                             in_channels, out_channels, out_offset,
                             quant_data->mult, quant_data->shift,
                             activation_min, activation_max,
                             row_addend == NULL ? NULL : row_addend + pix);
        }

        /* Remaining pixels (< 16): scalar fallback */
        for (; pix < total_pixels; pix++) {
            const int8_t *inp = input_data + pix * in_channels;
            filter_ptr = filter_data;
            for (int32_t oc = 0; oc < out_channels; oc++) {
                int32_t conv_out = 0;
                for (int32_t ic = 0; ic < in_channels; ic++) {
                    conv_out += inp[ic] * filter_ptr[ic];
                }
                conv_out += filter_sum[oc];
                if (row_addend != NULL) {
                    conv_out += row_addend[pix];
                }
                if (bias) conv_out += bias[oc];
                conv_out = esp_nn_multiply_by_quantized_mult(conv_out,
                    quant_data->mult[oc], quant_data->shift[oc]);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                out_data[pix * out_channels + oc] = (int8_t) conv_out;
                filter_ptr += in_channels;
            }
        }
        return;
    }

    /* OC-panel tiling (see esp_nn_conv_s8_padded): sweep a panel of output
     * channels across all pixels so the panel's filter rows stay resident in
     * the 64 KB L1 D-cache; filter data is fetched from memory once per
     * layer instead of once per pixel. Arithmetic order per output element
     * is unchanged: bit-identical results. */
    int32_t oc_tile = out_channels;
    if ((int32_t)out_channels * in_channels > 24 * 1024) {
        oc_tile = (24 * 1024) / in_channels;
        if (oc_tile < 4) {
            oc_tile = 4;
        }
    }

    int32_t loop_ht = out_ht;
    int32_t loop_wd = out_wd;
    int32_t loop_in_wd = input_wd;

    /* Position-batched fast path: 16 pixels per QACC pass (lanes = pixels,
     * broadcast filter byte per input channel). Requires contiguous pixels,
     * which holds for the stride-1 unpadded 1x1 (input_wd == out_wd). The
     * per-pixel dot setup of the XACC path below disappears; the pixel tail
     * (< 16) falls through to that path. */
    if (input_wd == out_wd) {
        const int32_t total_pixels = (int32_t)out_wd * out_ht;
        const int32_t batched = total_pixels & ~15;
        if (batched > 0) {
            int8_t *transposed = (int8_t *)(((uintptr_t)((int8_t *)scratch
                    + out_channels * 4 + 15)) & ~(uintptr_t)15);
            for (int32_t pix = 0; pix < batched; pix += 16) {
                const int8_t *in_batch = input_data + pix * in_channels;
                int8_t *out_batch = out_data + pix * out_channels;
                for (int32_t oc_base = 0; oc_base < out_channels;
                        oc_base += oc_tile) {
                    const int32_t oc_end = min(oc_base + oc_tile, out_channels);
                    conv_1x1_batch16_deep(in_batch, filter_data, filter_sum,
                                          bias, out_batch, in_channels,
                                          oc_base, oc_end, out_channels,
                                          out_offset, quant_data->mult,
                                          quant_data->shift,
                                          activation_min, activation_max,
                                          transposed,
                                          row_addend == NULL ? NULL
                                                             : row_addend + pix);
                }
            }
            if (batched == total_pixels) {
                return;
            }
            /* Tail (< 16 pixels). The XACC hardware loop below faults on
             * the 768-wide FFN remainder (loop exit lands off an
             * instruction), so finish these pixels in scalar code. */
            input_data += batched * in_channels;
            out_data += batched * out_channels;
            const int32_t tail = total_pixels - batched;
            for (int32_t pix = 0; pix < tail; ++pix) {
                const int8_t *inp = input_data + pix * in_channels;
                const int8_t *fp = filter_data;
                for (int32_t oc = 0; oc < out_channels; ++oc) {
                    int32_t conv_out = 0;
                    for (int32_t ic = 0; ic < in_channels; ++ic) {
                        conv_out += inp[ic] * fp[ic];
                    }
                    conv_out += filter_sum[oc];
                    if (row_addend != NULL) {
                        conv_out += row_addend[batched + pix];
                    }
                    if (bias) {
                        conv_out += bias[oc];
                    }
                    conv_out = esp_nn_requantize(conv_out,
                            quant_data->mult[oc], quant_data->shift[oc]);
                    conv_out += out_offset;
                    conv_out = max(conv_out, activation_min);
                    conv_out = min(conv_out, activation_max);
                    out_data[pix * out_channels + oc] = (int8_t) conv_out;
                    fp += in_channels;
                }
            }
            return;
        }
    }

    for (int32_t oc_base = 0; oc_base < out_channels; oc_base += oc_tile) {
    const int32_t oc_end = min(oc_base + oc_tile, out_channels);
    for (int32_t in_row = 0; in_row < loop_ht; in_row++) {
        for (int32_t in_col = 0; in_col < loop_wd; in_col++) {
            const int32_t *out_mult = quant_data->mult + oc_base;
            const int32_t *out_shift = quant_data->shift + oc_base;
            filter_ptr = filter_data + oc_base * in_channels;
            const int8_t *input_base_ptr = input_data + (in_row * loop_in_wd + in_col) * in_channels;
            int8_t *out_ptr = out_data
                    + ((int32_t)(in_row * loop_wd + in_col)) * out_channels + oc_base;
            for (int32_t out_ch_idx = oc_base; out_ch_idx < oc_end; out_ch_idx++) {
                /* initializations */
                int32_t conv_out = 0;
                const int8_t *input_ptr = input_base_ptr;

                int32_t in_ch_idx = 0;
#if 1 // inline asm
                // for now check for the alignment as well
                if (in_channels < 16) {// || ((uint32_t) input_ptr & 15) || ((uint32_t) filter_ptr & 15)) {
                    goto skip_asm;
                }

                int32_t c16 = (in_channels >> 4) - 1;
                asm volatile (
                    "mv x30, %[inp]                 \n\t"
                    "mv x31, %[flt]                 \n\t"
                    "esp.zero.xacc                  \n\t"
                    "esp.vld.128.ip  q0, x30, 16    \n\t"
                    "esp.vld.128.ip  q1, x31, 16    \n\t"

                    "beqz %[c16], 2f                \n\t"
                    /* zero-overhead loop; end label ON last body insn */
                    "esp.lp.setup 0, %[c16], 1f     \n\t"
                    "esp.vmulas.s8.xacc.ld.ip  q0, x30, 16, q0, q1   \n\t"
                    "1:                             \n\t"
                    "esp.vld.128.ip  q1, x31, 16    \n\t"
                    "2:                             \n\t"

                    // move input_ptr, filter_ptr and conv_out
                    "mv %[inp], x30                 \n\t"
                    "mv %[flt], x31                 \n\t"
                    "esp.vmulas.s8.xacc  q0, q1     \n\t"
                    /* esp.movx GPR operand must be x26-x31 (required on S31) */
                    "esp.movx.r.xacc.l  x29         \n\t"
                    "mv %[out], x29                 \n\t"

                    : [inp] "+r" (input_ptr), [flt] "+r" (filter_ptr), [out] "=r" (conv_out)
                    : [c16] "r"(c16)
                    : "x29", "x30", "x31"
                );
                in_ch_idx = in_channels & ~15;
skip_asm:
#endif
                for (; in_ch_idx < in_channels - 3; in_ch_idx += 4) {
                    conv_out += *input_ptr++ * *filter_ptr++;
                    conv_out += *input_ptr++ * *filter_ptr++;
                    conv_out += *input_ptr++ * *filter_ptr++;
                    conv_out += *input_ptr++ * *filter_ptr++;
                }

                for (; in_ch_idx < in_channels; in_ch_idx++) {
                    conv_out += *input_ptr++ * *filter_ptr++;
                }
                conv_out = conv_out + filter_sum[out_ch_idx];
                if (row_addend != NULL) {
                    conv_out += row_addend[in_row * loop_wd + in_col];
                }
                if (bias) {
                    conv_out += bias[out_ch_idx];
                }
                conv_out = esp_nn_requantize(conv_out, *out_mult++, *out_shift++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_ptr++ = (int8_t) conv_out;
            }
        }
    }
    }
}

__attribute__ ((noinline))
static void esp_nn_conv_s8_padded(
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
    const uint16_t in_channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
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

    int32_t *filter_sum = (int32_t *) scratch; // alignment of 4 bytes assumed

    /* pre-calculate filter_sum * input_offset */
    {
        const int32_t filter_len = filter_wd * filter_ht * in_channels;
        for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
            filter_sum[out_ch_idx] = conv_filter_byte_sum(
                    filter_data + out_ch_idx * filter_len, filter_len) * input_offset;
        }
    }

    const int32_t row_size = filter_wd * in_channels;

    /* Interior extent: output columns/rows whose whole filter window lies
     * inside the input. Anything beyond falls in TFLite's implicit trailing
     * padding (clipped taps), handled by the edge paths below. */
    /* Guard the truncating division: an input smaller than the filter has
     * no fully-inside output at all, not one. */
    int32_t eff_wd = (input_wd >= filter_wd) ? (input_wd - filter_wd) / stride_wd + 1 : 0;
    int32_t eff_ht = (input_ht >= filter_ht) ? (input_ht - filter_ht) / stride_ht + 1 : 0;
    if (eff_wd > out_wd) eff_wd = out_wd;
    if (eff_ht > out_ht) eff_ht = out_ht;
    const bool right_pad = eff_wd < out_wd;
    const bool bottom_pad = eff_ht < out_ht;

    /*
     * OC-panel tiling: sweep a panel of output channels across all output
     * pixels before moving to the next panel. The panel's filters
     * (oc_tile * filter_size bytes) stay resident in the L1 D-cache
     * (P4 TRM 9.3.3.2: 64 KB dcache, 2-way, 64 B lines) for the whole pixel
     * sweep, so filter data is fetched from memory once per layer instead of
     * once per output pixel. Arithmetic order per output element is
     * unchanged: bit-identical results.
     */
    const int32_t filter_size = filter_ht * row_size;
    int32_t oc_tile = out_channels;
    if ((int32_t)out_channels * filter_size > 24 * 1024 && filter_size > 0) {
        oc_tile = (24 * 1024) / filter_size;
        if (oc_tile < 4) {
            oc_tile = 4;
        }
    }


    for (int32_t oc_base = 0; oc_base < out_channels; oc_base += oc_tile) {
        const int32_t oc_end = min(oc_base + oc_tile, out_channels);
        const int8_t *panel_filter = filter_data + oc_base * filter_size;
        for (int32_t out_y = 0; out_y < eff_ht; out_y++) {
            const int32_t base_y = stride_ht * out_y;
            for (int32_t out_x = 0; out_x < eff_wd; out_x++) {
                const int32_t base_x = stride_wd * out_x;
                const int8_t *filter_data_ptr = panel_filter;
                int8_t *out_ptr = out_data
                        + ((int32_t)out_y * out_wd + out_x) * out_channels + oc_base;
                for (int32_t out_ch_idx = oc_base; out_ch_idx < oc_end; out_ch_idx++) {
                    int32_t conv_out = 0, filter_y_idx;
                    if (row_size >= 32 && (row_size & 15) == 0) {
                        /* Double-buffered fused-rows dot (no per-row scalar
                         * tail needed when row_size is a multiple of 16). */
                        conv_out = conv_dot_rows_pie(
                                input_data + ((int32_t)base_y * input_wd
                                              + base_x) * in_channels,
                                filter_data_ptr, filter_ht, row_size,
                                (int32_t)input_wd * in_channels);
                        filter_data_ptr += (int32_t)filter_ht * row_size;
                        conv_out += filter_sum[out_ch_idx];
                        if (bias) {
                            conv_out += bias[out_ch_idx];
                        }
                        conv_out = esp_nn_requantize(conv_out,
                                                     out_mult[out_ch_idx],
                                                     out_shift[out_ch_idx]);
                        conv_out += out_offset;
                        conv_out = max(conv_out, activation_min);
                        conv_out = min(conv_out, activation_max);
                        *out_ptr++ = (int8_t) conv_out;
                        continue;
                    }
                    if (row_size >= 16) {
                        asm volatile("esp.zero.xacc                  \n\t");
                    }

                    for (filter_y_idx = 0; filter_y_idx < filter_ht; filter_y_idx++) {
                        const int32_t in_row = base_y + filter_y_idx;
                        const int32_t in_col = base_x;
                        const int8_t *input_data_ptr =
                                input_data + (in_row * input_wd + in_col) * in_channels;
                        int32_t row_idx = 0;
#if 1 // inline asm
                    // for now check for the alignment as well
                    if (row_size < 16) {// || ((uint32_t) input_ptr & 15) || ((uint32_t) filter_ptr & 15)) {
                        goto skip_asm_pad0;
                    }

                    {
                    int32_t c16 = (row_size >> 4) - 1;
                    /* Two loop forms, identical arithmetic: the zero-overhead
                     * hardware loop (as in the 1x1 kernel) drops the addi/bnez
                     * pair per 16-byte chunk and pays off once the trip count
                     * amortizes its setup; the software loop stays for short
                     * rows (small in_ch), where esp.lp.setup measured slower
                     * on ESP32-S31. */
                    if (c16 >= 4) {
                        asm volatile (
                            "mv x30, %[inp]                 \n\t"
                            "mv x31, %[flt]                 \n\t"
                            "esp.vld.128.ip  q0, x30, 16    \n\t"
                            "esp.vld.128.ip  q1, x31, 16    \n\t"

                            /* zero-overhead loop; end label ON last body insn */
                            "esp.lp.setup 0, %[c16], 1f     \n\t"
                            "esp.vmulas.s8.xacc.ld.ip  q0, x30, 16, q0, q1   \n\t"
                            "1:                             \n\t"
                            "esp.vld.128.ip  q1, x31, 16    \n\t"

                            // move input_ptr and filter_ptr
                            "mv %[inp], x30                 \n\t"
                            "mv %[flt], x31                 \n\t"
                            "esp.vmulas.s8.xacc  q0, q1     \n\t"

                            : [inp] "+r" (input_data_ptr), [flt] "+r" (filter_data_ptr)
                            : [c16] "r"(c16)
                            : "x30", "x31"
                        );
                    } else {
                        asm volatile (
                            "mv x30, %[inp]                 \n\t"
                            "mv x31, %[flt]                 \n\t"
                            "esp.vld.128.ip  q0, x30, 16    \n\t"
                            "esp.vld.128.ip  q1, x31, 16    \n\t"

                            "beqz %[c16], 2f                \n\t"
                            "mv   s7, %[c16]                \n\t"
                            "1:                             \n\t"
                            "esp.vmulas.s8.xacc.ld.ip  q0, x30, 16, q0, q1   \n\t"
                            "esp.vld.128.ip  q1, x31, 16    \n\t"
                            "addi s7, s7, -1                \n\t"
                            "bnez s7, 1b                    \n\t"
                            "2:                             \n\t"

                            // move input_ptr and filter_ptr
                            "mv %[inp], x30                 \n\t"
                            "mv %[flt], x31                 \n\t"
                            "esp.vmulas.s8.xacc  q0, q1     \n\t"

                            : [inp] "+r" (input_data_ptr), [flt] "+r" (filter_data_ptr)
                            : [c16] "r"(c16)
                            : "x30", "x31", "s7"
                        );
                    }
                    row_idx = row_size & ~15;
                    }
skip_asm_pad0:
#endif
                        for (; row_idx < row_size - 3; row_idx += 4) {
                            conv_out += *input_data_ptr++ * *filter_data_ptr++;
                            conv_out += *input_data_ptr++ * *filter_data_ptr++;
                            conv_out += *input_data_ptr++ * *filter_data_ptr++;
                            conv_out += *input_data_ptr++ * *filter_data_ptr++;
                        }
                        for (; row_idx < row_size; row_idx++) {
                            conv_out += *input_data_ptr++ * *filter_data_ptr++;
                        }
                    }
                    if (row_size >= 16) {
                        asm volatile (
                            "esp.movx.r.xacc.l  x30   \n\t"
                            "add %0, %0, x30          \n\t"
                            : "+r" (conv_out)
                            :
                            : "x30"
                        );
                    }
                    /* add input_offset term */
                    conv_out += filter_sum[out_ch_idx];

                    if (bias) {
                        conv_out += bias[out_ch_idx];
                    }
                    conv_out = esp_nn_requantize(conv_out, out_mult[out_ch_idx],
                                                 out_shift[out_ch_idx]);
                    conv_out += out_offset;
                    conv_out = max(conv_out, activation_min);
                    conv_out = min(conv_out, activation_max);
                    *out_ptr++ = (int8_t) conv_out;
                }
            }
        }
    }

    /* Right-edge columns whose filter window runs past the input: scalar
     * path with the clipped tap count, as before (few pixels). */
    for (int32_t out_y = 0; out_y < eff_ht && right_pad; out_y++) {
        for (int32_t out_x = eff_wd; out_x < out_wd; out_x++) {
            const int32_t base_y = stride_ht * out_y;
            const int32_t base_x = stride_wd * out_x;
            const int32_t *out_mult_ptr = out_mult;
            const int32_t *out_shift_ptr = out_shift;
            const int32_t *bias_ptr = bias;
            int8_t *out_ptr = out_data
                    + ((int32_t)out_y * out_wd + out_x) * out_channels;
            for (int32_t out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
                int32_t conv_out = 0, filter_y_idx;
                /* Clip taps per pixel: the overhang grows toward the edge. */
                const int32_t fx_end = min(filter_wd, input_wd - base_x);
                for (filter_y_idx = 0; filter_y_idx < filter_ht; filter_y_idx++) {
                    for (int32_t filter_x_idx = 0; filter_x_idx < fx_end; filter_x_idx++) {
                        const int32_t in_row = base_y + filter_y_idx;
                        const int32_t in_col = base_x + filter_x_idx;

                        const int8_t *input_ptr = input_data +
                                        (in_row * input_wd + in_col) * in_channels;
                        const int8_t *filter_ptr = filter_data +
                                        out_ch_idx * in_channels * filter_ht * filter_wd +
                                        (filter_y_idx * filter_wd + filter_x_idx) * in_channels;
                        int32_t in_ch_idx = 0;
                        for (; in_ch_idx < in_channels - 3; in_ch_idx += 4) {
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                        }
                        for (; in_ch_idx < in_channels; in_ch_idx ++) {
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                        }
                    }
                }

                if (bias) {
                    conv_out += *bias_ptr++;
                }
                conv_out = esp_nn_requantize(conv_out, *out_mult_ptr++, *out_shift_ptr++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_ptr++ = (int8_t) conv_out;
            }
        }
    }

    // Calculate the last row if needed. Hand the remaining input rows to the
    // generic kernel, which clamps filter windows to the input extent (the
    // rows falling in the implicit trailing padding contribute zero).
    if (bottom_pad) {
        const int32_t in_row = eff_ht * stride_ht;
        const int32_t rows = out_ht - eff_ht;
        esp_nn_conv_s8_opt(&(data_dims_t){input_dims->width, input_dims->height - in_row,
                                          input_dims->channels, 0},
                            input_data + in_row * input_dims->width * input_dims->channels,
                            filter_dims, filter_data, bias,
                            &(data_dims_t){output_dims->width, (uint16_t)rows, output_dims->channels, 0},
                            out_data + (int32_t)eff_ht * out_wd * out_channels,
                            conv_params, quant_data);
    }
}

/* L1D cache budget: use half of 64KB to leave room for filter streaming */
#define L1D_BUDGET 32768

/* Largest filter window the im2col QACC batch path stages for 16 pixels
 * (16 * 2304 = 36 KB of staging). Covers 3x3 windows up to 256 channels. */
#define ESP_NN_IM2COL_BATCH_MAX_WINDOW 2304

/**
 * Im2col convolution for small in_ch where filter_wd * in_ch < 16.
 *
 * Instead of padding channels (81% wasted MACs for in_ch=3),
 * concatenates the entire filter window into one contiguous vector:
 *   window_len = filter_wd * filter_ht * in_ch (e.g., 3*3*3 = 27)
 *
 * For each output pixel: copy the input window into a contiguous scratch
 * buffer, then use PIE dot product on the full window. No wasted MACs.
 *
 * Scratch layout: [filter_sum | im2col_buf]
 *   im2col_buf = filter_wd * filter_ht * in_ch bytes
 */
__attribute__ ((noinline))
static void esp_nn_conv_s8_im2col(
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
    const int8_t pad_val = (int8_t)(-input_offset);

    /* Scratch: filter_sum[out_ch] + im2col_buf[window_len] */
    int32_t *filter_sum = (int32_t *)scratch;
    int8_t *im2col_buf = (int8_t *)scratch + out_ch * sizeof(int32_t);

    /* Pre-compute filter_sum * input_offset */
    for (int32_t oc = 0; oc < out_ch; oc++) {
        filter_sum[oc] = conv_filter_byte_sum(
                filter_data + oc * window_len, window_len) * input_offset;
    }

    const int32_t total_pixels = (int32_t)out_wd * out_ht;

    /* QACC batch path. A pixel's window is fully inside the input when
     *   0 <= ox*stride_wd - pad_wd  and  ox*stride_wd - pad_wd + filter_wd <= input_wd
     * (same in y). Batch 16 consecutive pixels along each interior output
     * row - padded layers keep a fast path for their interior, which is the
     * bulk of the work (a 24x24 output with pad 1 is 84% interior); only the
     * border pixels fall to the per-pixel path below. Accumulation order per
     * output element is unchanged: bit-identical. */
    int32_t bx_lo = (pad_wd + stride_wd - 1) / stride_wd;
    int32_t by_lo = (pad_ht + stride_ht - 1) / stride_ht;
    /* Last output column/row whose window lies entirely inside the input.
     * C division truncates toward zero, so a negative numerator (input
     * narrower than the filter, TFLite SAME on e.g. 2x2 -> 3x3 s2) would
     * yield 0 and batch pixel 0 as "interior" although its window overruns
     * the input; there is no interior at all in that case. */
    int32_t bx_hi = (input_wd + pad_wd >= filter_wd)
                    ? (input_wd + pad_wd - filter_wd) / stride_wd : -1;
    int32_t by_hi = (input_ht + pad_ht >= filter_ht)
                    ? (input_ht + pad_ht - filter_ht) / stride_ht : -1;
    if (bx_hi > out_wd - 1) {
        bx_hi = out_wd - 1;
    }
    if (by_hi > out_ht - 1) {
        by_hi = out_ht - 1;
    }
    /* number of whole 16-pixel groups per interior row, and where they end */
    int32_t row_groups = (bx_hi >= bx_lo) ? 1 : 0;   /* any interior width */
    int32_t bx_end = (bx_hi >= bx_lo) ? (bx_hi + 1) : bx_lo;  /* exclusive */

    /* When the whole tensor is interior (no padding at all) the batch can run
     * over pixels in raster order, which also covers outputs narrower than
     * 16. Padded layers batch per interior row instead. */
    const bool all_interior = (pad_wd == 0 && pad_ht == 0 &&
            (int32_t)(out_wd - 1) * stride_wd + filter_wd <= input_wd &&
            (int32_t)(out_ht - 1) * stride_ht + filter_ht <= input_ht);
    int32_t raster_end = 0;     /* exclusive, in pixels */

    if (all_interior && window_len <= ESP_NN_IM2COL_BATCH_MAX_WINDOW &&
            total_pixels >= 16) {
        const int32_t row_size = filter_wd * in_ch;
        int8_t *t = (int8_t *)(((uintptr_t)(im2col_buf + window_len) + 15)
                               & ~(uintptr_t)15);
        raster_end = (total_pixels / 16) * 16;
        for (int32_t pix = 0; pix < raster_end; pix += 16) {
            const int8_t *bases[16];
            for (int p = 0; p < 16; p++) {
                const int32_t px = pix + p;
                const int32_t oy = px / out_wd;
                const int32_t ox = px % out_wd;
                bases[p] = input_data + ((int32_t)(oy * stride_ht) * input_wd
                                         + ox * stride_wd) * in_ch;
            }
            for (int32_t fy = 0; fy < filter_ht; fy++) {
                const int32_t row_off = (int32_t)fy * input_wd * in_ch;
                int8_t *dst = t + (int32_t)fy * row_size * 16;
                for (int32_t i = 0; i < row_size; i++, dst += 16) {
                    const int32_t off = row_off + i;
                    dst[0]  = bases[0][off];   dst[1]  = bases[1][off];
                    dst[2]  = bases[2][off];   dst[3]  = bases[3][off];
                    dst[4]  = bases[4][off];   dst[5]  = bases[5][off];
                    dst[6]  = bases[6][off];   dst[7]  = bases[7][off];
                    dst[8]  = bases[8][off];   dst[9]  = bases[9][off];
                    dst[10] = bases[10][off];  dst[11] = bases[11][off];
                    dst[12] = bases[12][off];  dst[13] = bases[13][off];
                    dst[14] = bases[14][off];  dst[15] = bases[15][off];
                }
            }
            qacc16_run_channels(t, filter_data, window_len, 0, out_ch,
                                filter_sum, bias, quant_data->mult,
                                quant_data->shift,
                                out_data + (int32_t)pix * out_ch, out_ch,
                                out_offset, activation_min, activation_max, 16,
                                NULL);
        }
        row_groups = 0;          /* per-pixel skips by raster_end instead */
        bx_end = bx_lo;
    } else if (window_len <= ESP_NN_IM2COL_BATCH_MAX_WINDOW && row_groups > 0) {
        const int32_t row_size = filter_wd * in_ch;
        int8_t *t = (int8_t *)(((uintptr_t)(im2col_buf + window_len) + 15)
                               & ~(uintptr_t)15);
        for (int32_t oy = by_lo; oy <= by_hi; oy++) {
            const int32_t base_y = oy * stride_ht - pad_ht;
            for (int32_t ox = bx_lo; ox < bx_end; ox += 16) {
                const int32_t lanes = min(16, bx_end - ox);
                /* Stage the windows transposed: t[tap * 16 + p]. Tap-outer so
                 * each tap's 16 stores fill one contiguous 16-byte line.
                 * Lanes past `lanes` repeat pixel 0 - they are multiplied but
                 * never stored, and reading in-bounds keeps it safe. */
                const int8_t *bases[16];
                for (int p = 0; p < 16; p++) {
                    const int32_t src_x = (p < lanes) ? (ox + p) : ox;
                    bases[p] = input_data +
                            ((int32_t)base_y * input_wd
                             + src_x * stride_wd - pad_wd) * in_ch;
                }
                for (int32_t fy = 0; fy < filter_ht; fy++) {
                    const int32_t row_off = (int32_t)fy * input_wd * in_ch;
                    int8_t *dst = t + (int32_t)fy * row_size * 16;
                    for (int32_t i = 0; i < row_size; i++, dst += 16) {
                        const int32_t off = row_off + i;
                        dst[0]  = bases[0][off];
                        dst[1]  = bases[1][off];
                        dst[2]  = bases[2][off];
                        dst[3]  = bases[3][off];
                        dst[4]  = bases[4][off];
                        dst[5]  = bases[5][off];
                        dst[6]  = bases[6][off];
                        dst[7]  = bases[7][off];
                        dst[8]  = bases[8][off];
                        dst[9]  = bases[9][off];
                        dst[10] = bases[10][off];
                        dst[11] = bases[11][off];
                        dst[12] = bases[12][off];
                        dst[13] = bases[13][off];
                        dst[14] = bases[14][off];
                        dst[15] = bases[15][off];
                    }
                }
                qacc16_run_channels(t, filter_data, window_len, 0, out_ch,
                                    filter_sum, bias, quant_data->mult,
                                    quant_data->shift,
                                    out_data + ((int32_t)oy * out_wd + ox) * out_ch,
                                    out_ch, out_offset,
                                    activation_min, activation_max, lanes,
                                    NULL);
            }
        }
    } else {
        row_groups = 0;      /* nothing batched: per-pixel covers everything */
        bx_end = bx_lo;
    }

    /* Per-pixel path: border pixels the interior batch above did not cover
     * (and everything, when no batch ran). */
    for (int32_t pixel = 0; pixel < total_pixels; pixel++) {
        {
            const int32_t out_y = pixel / out_wd;
            const int32_t out_x = pixel % out_wd;
            if (pixel < raster_end) {
                continue;       /* produced by the raster batch */
            }
            if (row_groups > 0 && out_y >= by_lo && out_y <= by_hi &&
                    out_x >= bx_lo && out_x < bx_end) {
                continue;       /* produced by the interior-row batch */
            }
            int8_t *out_ptr = out_data + (int32_t)pixel * out_ch;
            const int32_t base_y = out_y * stride_ht - pad_ht;
            const int32_t base_x = out_x * stride_wd - pad_wd;

            /* Copy input window into contiguous im2col buffer */
            int8_t *buf = im2col_buf;
            for (int32_t fy = 0; fy < filter_ht; fy++) {
                int32_t in_y = base_y + fy;
                for (int32_t fx = 0; fx < filter_wd; fx++) {
                    int32_t in_x = base_x + fx;
                    if (in_y >= 0 && in_y < input_ht && in_x >= 0 && in_x < input_wd) {
                        const int8_t *src = input_data + (in_y * input_wd + in_x) * in_ch;
                        for (int c = 0; c < in_ch; c++) {
                            *buf++ = src[c];
                        }
                    } else {
                        /* Padding pixel */
                        for (int c = 0; c < in_ch; c++) {
                            *buf++ = pad_val;
                        }
                    }
                }
            }

            /* Dot product against each output channel's filter */
            const int32_t *out_mult = quant_data->mult;
            const int32_t *out_shift = quant_data->shift;
            const int8_t *filter_ptr = filter_data;

            for (int32_t oc = 0; oc < out_ch; oc++) {
                int32_t conv_out = pie_dot_s8(im2col_buf, filter_ptr, window_len);
                conv_out += filter_sum[oc];
                if (bias) conv_out += bias[oc];
                conv_out = esp_nn_requantize(conv_out, *out_mult++, *out_shift++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_ptr++ = (int8_t) conv_out;
                filter_ptr += window_len;
            }
        }
    }
}

/**
 * Tiled convolution: process T output rows at a time.
 * Converts padded conv into a series of no-pad sub-problems by
 * copying/padding input tiles into the scratch buffer.
 *
 * This keeps the working set in L1D for large input tensors.
 * Reuses the existing esp_nn_conv_s8_padded PIE inner loop per tile.
 */
/* Tile plan for the padded tiny-window conv. Computed by BOTH the scratch
 * getter and the kernel from the same inputs, so the two can never disagree
 * on the path taken or on the bytes staged: a divergence here silently
 * overruns the caller's arena. */
typedef struct {
    int eff_ch;        /* channels after PIE lane padding (in_ch if none) */
    int filt_aligned;  /* bytes of the channel-padded filter copy, 0 if none */
    int row_bytes;     /* one staged (padded-width) input row */
    int tile_T;        /* output rows per tile */
    int staged_rows;   /* input rows staged for the tallest tile */
    int scratch_bytes; /* filter_sum + filt_aligned + staging */
} conv_tile_plan_t;

static void conv_plan_tiles(int input_wd, int in_ch,
                            int filter_wd, int filter_ht, int out_ch, int out_ht,
                            int pad_wd, int stride_ht, conv_tile_plan_t *p)
{
    p->eff_ch = in_ch;
    p->filt_aligned = 0;
    if (filter_wd * in_ch < 16) {
        /* PIE row dot needs 16 lanes: pad channels up */
        p->eff_ch = ((16 + filter_wd - 1) / filter_wd + 15) & ~15;
        p->filt_aligned = filter_wd * filter_ht * p->eff_ch * out_ch;
    }
    p->row_bytes = (input_wd + 2 * pad_wd) * p->eff_ch;
    const int fixed = out_ch * 4 + p->filt_aligned; /* filter_sum + filter copy */

    /* Monolithic by default. The kernel stages every input row the output
     * needs, (out_ht - 1) * stride + filter_ht of them; for TFLite SAME shapes
     * with trailing pad > leading pad that exceeds input_ht + 2 * pad_ht, so
     * size from the rows actually staged, not from the padded input. */
    p->tile_T = out_ht;
    if (((out_ht - 1) * stride_ht + filter_ht) * p->row_bytes + fixed > L1D_BUDGET) {
        const int budget = L1D_BUDGET - fixed;
        if (filter_ht * p->row_bytes <= budget) {
            p->tile_T = (budget - filter_ht * p->row_bytes)
                        / (stride_ht * p->row_bytes) + 1;
        } else {
            /* Even one filter-height band overflows L1: take the smallest
             * tile and let it spill rather than staging the whole input. */
            p->tile_T = 1;
        }
        if (p->tile_T < 1) p->tile_T = 1;
        if (p->tile_T > out_ht) p->tile_T = out_ht;
    }
    p->staged_rows = (p->tile_T - 1) * stride_ht + filter_ht;
    p->scratch_bytes = fixed + p->staged_rows * p->row_bytes;
}

__attribute__ ((noinline))
static void esp_nn_conv_s8_tiled(
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
    const uint16_t stride_ht = conv_params->stride.height;
    const int32_t input_offset = conv_params->in_offset;

    /* Shared plan: channel padding for PIE (row_size must be >= 16) and the
     * tile height. The scratch getter runs the identical computation. */
    conv_tile_plan_t plan;
    conv_plan_tiles(input_wd, in_ch, filter_wd, filter_ht, out_ch, out_ht,
                    pad_wd, stride_ht, &plan);
    int new_ch = plan.eff_ch;
    int need_ch_pad = (plan.filt_aligned != 0);
    int padded_input_wd = input_wd + 2 * pad_wd;

    /* Scratch layout:
     * [0] filter_sum: out_ch * 4 bytes
     * [after filter_sum] aligned_filter (if ch padding): filter_wd * filter_ht * new_ch * out_ch
     * [after filter] tile_input_buf: variable per tile
     */
    int32_t *filter_sum = (int32_t *) scratch;
    int filter_sum_size = out_ch * sizeof(int32_t);

    /* Pre-compute filter_sum * input_offset (once for entire layer) */
    {
        const int32_t flen = filter_wd * filter_ht * in_ch;
        for (int32_t oc = 0; oc < out_ch; oc++) {
            filter_sum[oc] = conv_filter_byte_sum(
                    filter_data + oc * flen, flen) * input_offset;
        }
    }

    /* Channel-pad filter if needed (pad with 0s - doesn't affect filter_sum) */
    int8_t *aligned_filter = NULL;
    int aligned_filter_size = 0;
    if (need_ch_pad) {
        aligned_filter = (int8_t *)scratch + filter_sum_size;
        aligned_filter_size = filter_wd * filter_ht * new_ch * out_ch;
        memset(aligned_filter, 0, aligned_filter_size);
        const int8_t *src_f = filter_data;
        int8_t *dst_f = aligned_filter;
        for (int oc = 0; oc < out_ch; oc++) {
            for (int fh = 0; fh < filter_ht; fh++) {
                for (int fw = 0; fw < filter_wd; fw++) {
                    memcpy(dst_f, src_f, in_ch);
                    src_f += in_ch;
                    dst_f += new_ch;  /* zero-padded channels */
                }
            }
        }
    }

    /* Tile input buffer starts after filter_sum + aligned_filter */
    int8_t *tile_buf = (int8_t *)scratch + filter_sum_size + aligned_filter_size;

    /* Use effective channel count for tile buffer sizing */
    int eff_ch = need_ch_pad ? new_ch : in_ch;
    int tile_input_row_bytes = padded_input_wd * eff_ch;

    /* Tile height T (output rows per tile) comes from the shared plan */
    const int tile_T = plan.tile_T;
    (void)tile_input_row_bytes;

    /* Process tiles */
    const int8_t *use_filter = need_ch_pad ? aligned_filter : filter_data;
    data_dims_t eff_filter_dims = {filter_wd, filter_ht, eff_ch, 0};

    for (int32_t tile_y = 0; tile_y < out_ht; tile_y += tile_T) {
        int32_t actual_T = min(tile_T, out_ht - tile_y);

        /* Input rows feeding output rows [tile_y, tile_y + actual_T):
         * exactly (actual_T - 1) * stride + filter_ht of them, which is what
         * conv_plan_tiles() sized the staging buffer for. */
        int32_t in_row_start = tile_y * stride_ht - pad_ht;
        int32_t in_row_end = (tile_y + actual_T - 1) * stride_ht - pad_ht + filter_ht - 1;
        int32_t tile_input_ht = in_row_end - in_row_start + 1;

        /* Copy/pad input rows into tile buffer, with channel padding if needed */
        int8_t pad_val = (int8_t)(-input_offset);
        int8_t *dst = tile_buf;

        for (int32_t row = in_row_start; row <= in_row_end; row++) {
            if (row < 0 || row >= input_ht) {
                memset(dst, pad_val, padded_input_wd * eff_ch);
            } else {
                /* For each pixel in padded row */
                int8_t *row_dst = dst;
                /* Left padding */
                for (int px = 0; px < pad_wd; px++) {
                    memset(row_dst, pad_val, eff_ch);
                    row_dst += eff_ch;
                }
                /* Valid pixels - with optional channel padding */
                const int8_t *row_src = input_data + row * input_wd * in_ch;
                if (need_ch_pad) {
                    for (int px = 0; px < input_wd; px++) {
                        memcpy(row_dst, row_src, in_ch);
                        if (eff_ch > in_ch) {
                            memset(row_dst + in_ch, pad_val, eff_ch - in_ch);
                        }
                        row_src += in_ch;
                        row_dst += eff_ch;
                    }
                } else {
                    memcpy(row_dst, row_src, input_wd * in_ch);
                    row_dst += input_wd * in_ch;
                }
                /* Right padding */
                for (int px = 0; px < pad_wd; px++) {
                    memset(row_dst, pad_val, eff_ch);
                    row_dst += eff_ch;
                }
            }
            dst += padded_input_wd * eff_ch;
        }

        /* Sub-problem with pad=0, effective channels */
        data_dims_t tile_input_dims = {padded_input_wd, tile_input_ht, eff_ch, 0};
        data_dims_t tile_output_dims = {out_wd, actual_T, out_ch, 0};
        conv_params_t tile_conv_params = *conv_params;
        tile_conv_params.padding.width = 0;
        tile_conv_params.padding.height = 0;

        esp_nn_conv_s8_padded(&tile_input_dims, tile_buf,
                              &eff_filter_dims, use_filter, bias,
                              &tile_output_dims,
                              out_data + tile_y * out_wd * out_ch,
                              &tile_conv_params, quant_data,
                              filter_sum);
    }
}

int esp_nn_get_conv_scratch_size_riscv_pie(const data_dims_t *input_dims,
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
            const int inner_ = esp_nn_get_conv_scratch_size_riscv_pie(&in_g_, filter_dims, &out_g_, conv_params);
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

    int new_channels = (in_ch + 7) & ~7;

    int input_scratch = input_wd * input_ht * in_ch;
    int filter_scratch = filter_wd * filter_ht * in_ch * out_ch;

    int align_buf_size = 32; /* extra buffer for alignment */
    if ((filter_wd == 1 && filter_ht == 1 && pad_wd == 0 && pad_ht == 0) &&
            (stride_wd == 1 && stride_ht == 1)) {
        if (in_ch < 16) {
            /* Channel-padding path: filter_sum + padded_filter + padded_input */
            int filter_sum_sz = out_ch * 4;
            int padded_filter_sz = 16 * out_ch;
            int padded_input_sz = 32; /* 16 bytes + alignment */
            return filter_sum_sz + padded_filter_sz + padded_input_sz + align_buf_size;
        }
        int transpose_buf_size = 2 * (8 * new_channels);
        if (input_wd * input_ht < 8) {
            transpose_buf_size = 0;
        }
        if (in_ch % 8) {
            input_scratch = input_wd * input_ht * new_channels;
        } else {
            input_scratch = 0;
        }
        filter_scratch = new_channels * out_ch;
        return input_scratch + filter_scratch + transpose_buf_size + align_buf_size;
    } else {
        new_channels = (in_ch + 15) & ~15;
        int offset_acc_scratch = out_ch * 4;

        if (pad_wd == 0 && pad_ht == 0 && filter_wd * in_ch >= 16 &&
                !(filter_wd * filter_ht * in_ch >= 128 &&
                  filter_wd * filter_ht * in_ch <= ESP_NN_IM2COL_BATCH_MAX_WINDOW &&
                  out_ch >= 16)) {
            /* Direct no-pad path: no input scratch needed. Shapes whose
             * window fits the im2col QACC batch are routed there instead
             * (see the dispatcher) and sized by the im2col branch below. */
            input_scratch = 0;
            filter_scratch = filter_wd * filter_ht * new_channels * out_ch;
            return input_scratch + filter_scratch + align_buf_size + offset_acc_scratch;
        }

        /* Im2col path: scratch = filter_sum + im2col_buf, plus the
         * 16-pixel transposed staging for the QACC batch path (bounded to
         * window_len <= ESP_NN_IM2COL_BATCH_MAX_WINDOW, matching the
         * kernel's batch condition).
         * Padded convs with SIMD-wide rows and a filter beyond the L1 panel
         * budget route to the tiled path (sized by the padded-case block
         * below), matching the dispatcher. */
        if (filter_wd * filter_ht * in_ch >= 16 &&
                !((pad_wd != 0 || pad_ht != 0) && filter_wd * in_ch >= 16 &&
                  (int32_t)filter_wd * filter_ht * in_ch * out_ch > 96 * 1024)) {
            int window_len = filter_wd * filter_ht * in_ch;
            int im2col_scratch = window_len;  /* one window buffer */
            if (window_len <= ESP_NN_IM2COL_BATCH_MAX_WINDOW) {
                im2col_scratch += window_len * 16 + 16;
            }
            return offset_acc_scratch + im2col_scratch + align_buf_size;
        }

        if (pad_wd == 0 && pad_ht == 0) {
            /* Very small window (< 16 elements total): tiled path */
            int eff_ch = ((16 + filter_wd - 1) / filter_wd + 15) & ~15;
            int filt_aligned = filter_wd * filter_ht * eff_ch * out_ch;
            int tile_input = input_wd * input_ht * eff_ch;
            return offset_acc_scratch + filt_aligned + tile_input + align_buf_size;
        }

        /* Padded tiny-window case: the tiled kernel and this getter share
         * conv_plan_tiles(), so the staging buffer is sized with the same
         * eff_ch, the same tile decision and the same staged-row count the
         * kernel will use. */
        conv_tile_plan_t plan;
        conv_plan_tiles(input_wd, in_ch, filter_wd, filter_ht, out_ch,
                        output_dims->height, pad_wd, stride_ht, &plan);
        return plan.scratch_bytes + align_buf_size;
    }
    return align_buf_size;
}

void esp_nn_set_conv_scratch_buf_riscv_pie(void *buf)
{
    scratch_buffer = (int16_t *) buf;
}

void esp_nn_conv_scratch_borrow_riscv_pie(void *buf, void **scratch,
                                          void **preferred, size_t *preferred_size)
{
    (void) preferred;
    (void) preferred_size;
    *scratch = scratch_buffer;
    scratch_buffer = (int16_t *) buf;
}

void esp_nn_conv_scratch_restore_riscv_pie(void *scratch, void *preferred,
                                           size_t preferred_size)
{
    (void) preferred;
    (void) preferred_size;
    scratch_buffer = (int16_t *) scratch;
}

typedef void (*conv_s8_pie_fn_t)(const data_dims_t *, const int8_t *,
                                 const data_dims_t *, const int8_t *,
                                 const int32_t *, const data_dims_t *,
                                 int8_t *, const conv_params_t *,
                                 const quant_data_t *, void *);

typedef struct {
    conv_s8_pie_fn_t fn;
    data_dims_t input_dims;
    const int8_t *input;
    const data_dims_t *filter_dims;
    const int8_t *filter_data;
    const int32_t *bias;
    data_dims_t output_dims;
    int8_t *out_data;
    conv_params_t conv_params;
    const quant_data_t *quant_data;
    void *scratch;
} conv_pie_rows_mt_job_t;

static void conv_pie_rows_mt_worker(void *p)
{
    const conv_pie_rows_mt_job_t *j = (const conv_pie_rows_mt_job_t *)p;
    j->fn(&j->input_dims, j->input, j->filter_dims, j->filter_data, j->bias,
          &j->output_dims, j->out_data, &j->conv_params, j->quant_data,
          j->scratch);
}

/* Output-row split for the pixel-independent conv paths. The top slice
 * keeps the top padding; the bottom slice enters the input at an interior
 * row with pad_ht = 0, so both slices evaluate exactly the taps the single
 * call would, on disjoint output rows (bit-identical results). Returns
 * false when the split is not applicable or the worker is unavailable. */
static bool conv_pie_rows_split(conv_s8_pie_fn_t fn,
                                const data_dims_t *input_dims,
                                const int8_t *input,
                                const data_dims_t *filter_dims,
                                const int8_t *filter_data,
                                const int32_t *bias,
                                const data_dims_t *output_dims,
                                int8_t *out_data,
                                const conv_params_t *conv_params,
                                const quant_data_t *quant_data,
                                void *scratch)
{
    const uint16_t out_ht = output_dims->height;
    const uint16_t h0 = out_ht / 2;
    const int32_t in_row_off =
            (int32_t)h0 * conv_params->stride.height - conv_params->padding.height;
    if (!esp_nn_dual_core_active() || out_ht < 4 || in_row_off < 0) {
        return false;
    }
    const int scratch_size = esp_nn_get_conv_scratch_size_riscv_pie(
            input_dims, filter_dims, output_dims, conv_params);
    void *wscr = esp_nn_dual_core_scratch(scratch_size + 16);
    if (wscr == NULL) {
        return false;
    }
    conv_pie_rows_mt_job_t job = {
        .fn = fn, .input_dims = *input_dims, .input = input,
        .filter_dims = filter_dims, .filter_data = filter_data, .bias = bias,
        .output_dims = *output_dims, .out_data = out_data,
        .conv_params = *conv_params, .quant_data = quant_data, .scratch = wscr,
    };
    job.output_dims.height = h0;
    if (!esp_nn_dual_core_run(conv_pie_rows_mt_worker, &job)) {
        return false;
    }
    {
        data_dims_t in1 = *input_dims;
        data_dims_t outd1 = *output_dims;
        conv_params_t params1 = *conv_params;
        in1.height = input_dims->height - in_row_off;
        outd1.height = out_ht - h0;
        params1.padding.height = 0;
        fn(&in1,
           input + in_row_off * input_dims->width * input_dims->channels,
           filter_dims, filter_data, bias, &outd1,
           out_data + (int32_t)h0 * output_dims->width * output_dims->channels,
           &params1, quant_data, scratch);
    }
    esp_nn_dual_core_wait();
    return true;
}

typedef struct {
    data_dims_t input_dims;
    const int8_t *input;
    const int8_t *filter_data;
    const int32_t *bias;
    data_dims_t output_dims;
    int8_t *out_data;
    const conv_params_t *conv_params;
    const quant_data_t *quant_data;
    void *scratch;
    const int32_t *row_addend;
} conv1x1_pie_mt_job_t;

static void conv1x1_pie_mt_worker(void *p)
{
    const conv1x1_pie_mt_job_t *j = (const conv1x1_pie_mt_job_t *)p;
    esp_nn_conv_s8_1x1(&j->input_dims, j->input, j->filter_data, j->bias,
                       &j->output_dims, j->out_data, j->conv_params,
                       j->quant_data, j->scratch, j->row_addend);
}

void esp_nn_conv_s8_riscv_pie(const data_dims_t *input_dims,
                            const int8_t *input,
                            const data_dims_t *filter_dims,
                            const int8_t *filter_data,
                            const int32_t *bias,
                            const data_dims_t *output_dims,
                            int8_t *out_data,
                            const conv_params_t *conv_params,
                            const quant_data_t *quant_data)
{
    ESP_NN_PIE_ENABLE();

    if (scratch_buffer == NULL) {
        printf("esp_nn_conv error! scratch_buffer not set!\n");
        return;
    }

    /* Grouped conv (filter_ch < input_ch) must be caught before any fast
     * path: they all assume full-depth filters. Run each group through the
     * optimized path via repacked slices (recursing into this dispatcher
     * with matching channel counts), or fall back to the reference with a
     * dual-core row split. Lives here, not in a sub-kernel, because every
     * sub-kernel is only reached with matching channels. */
    if (input_dims->channels != filter_dims->channels) {
        data_dims_t in_g = *input_dims;
        data_dims_t out_g = *output_dims;
        const int32_t groups = filter_dims->channels
                ? input_dims->channels / filter_dims->channels : 0;
        if (groups > 1 && output_dims->channels % groups == 0) {
            in_g.channels = filter_dims->channels;
            out_g.channels = output_dims->channels / groups;
            const int inner = esp_nn_get_conv_scratch_size_riscv_pie(
                    &in_g, filter_dims, &out_g, conv_params);
            const int total = esp_nn_get_conv_scratch_size_riscv_pie(
                    input_dims, filter_dims, output_dims, conv_params);
            if (esp_nn_conv_s8_grouped_repack(
                    esp_nn_conv_s8_riscv_pie, input_dims, input,
                    filter_dims, filter_data, bias, output_dims, out_data,
                    conv_params, quant_data, scratch_buffer, total, inner)) {
                return;
            }
        }
        esp_nn_conv_s8_ansi_mt_split(input_dims, input, filter_dims, filter_data,
                                     bias, output_dims, out_data, conv_params, quant_data);
        return;
    }

    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;

    if (filter_wd == 1 && filter_ht == 1 && pad_wd == 0 && pad_ht == 0 &&
            stride_wd == 1 && stride_ht == 1) {
        /* Rows are independent for a stride-1 unpadded 1x1: split them
         * across both cores (bit-identical, disjoint outputs). */
        if (esp_nn_dual_core_active() && output_dims->height >= 4) {
            const int scratch_size = esp_nn_get_conv_scratch_size_riscv_pie(
                    input_dims, filter_dims, output_dims, conv_params);
            void *wscr = esp_nn_dual_core_scratch(scratch_size + 16);
            const uint16_t h0 = output_dims->height / 2;
            if (wscr != NULL && h0 > 0) {
                conv1x1_pie_mt_job_t job = {
                    .input_dims = *input_dims, .input = input,
                    .filter_data = filter_data, .bias = bias,
                    .output_dims = *output_dims, .out_data = out_data,
                    .conv_params = conv_params, .quant_data = quant_data,
                    .scratch = wscr,
                    .row_addend = fc_row_addend,
                };
                job.input_dims.height = h0;
                job.output_dims.height = h0;
                if (esp_nn_dual_core_run(conv1x1_pie_mt_worker, &job)) {
                    data_dims_t in1 = *input_dims;
                    data_dims_t outd1 = *output_dims;
                    in1.height = input_dims->height - h0;
                    outd1.height = output_dims->height - h0;
                    const int32_t off = (int32_t)h0 * input_dims->width;
                    const int32_t *row_rest = fc_row_addend == NULL
                            ? NULL : fc_row_addend + off;
                    esp_nn_conv_s8_1x1(&in1, input + off * input_dims->channels,
                                       filter_data, bias, &outd1,
                                       out_data + off * output_dims->channels,
                                       conv_params, quant_data, scratch_buffer,
                                       row_rest);
                    esp_nn_dual_core_wait();
                    return;
                }
            }
        }
        esp_nn_conv_s8_1x1(input_dims, input, filter_data, bias,
                           output_dims, out_data, conv_params, quant_data,
                           scratch_buffer, fc_row_addend);
    } else if (pad_wd == 0 && pad_ht == 0 &&
               (int32_t)filter_wd * filter_ht * input_dims->channels >= 128 &&
               (int32_t)filter_wd * filter_ht * input_dims->channels
                       <= ESP_NN_IM2COL_BATCH_MAX_WINDOW &&
               output_dims->channels >= 16) {
        /* Dense no-pad conv whose window fits the 16-pixel QACC batch: the
         * per-element XACC path below pays a dot setup and a requant per
         * output element, while the batch stages 16 windows once and
         * amortizes both across them (and weaves the requant into the MAC
         * stream). Same accumulation order per output: bit-identical. */
        esp_nn_conv_s8_im2col(input_dims, input, filter_dims, filter_data, bias,
                              output_dims, out_data, conv_params, quant_data,
                              scratch_buffer);
    } else if (pad_wd == 0 && pad_ht == 0 &&
               filter_wd * input_dims->channels >= 16) {
        /* No-pad, channels large enough for PIE: use direct padded path */
        if (!conv_pie_rows_split(esp_nn_conv_s8_padded, input_dims, input,
                                 filter_dims, filter_data, bias, output_dims,
                                 out_data, conv_params, quant_data,
                                 scratch_buffer)) {
            esp_nn_conv_s8_padded(input_dims, input, filter_dims, filter_data, bias,
                                  output_dims, out_data, conv_params, quant_data,
                                  scratch_buffer);
        }
    } else if ((pad_wd != 0 || pad_ht != 0) &&
               filter_wd * input_dims->channels >= 16 &&
               (int32_t)filter_wd * filter_ht * input_dims->channels *
                       output_dims->channels > 96 * 1024) {
        /* Padded conv whose filter exceeds the L2 cache: the pixel-outer
         * im2col loop re-streams the whole filter per pixel, and once it
         * no longer fits L2 that traffic comes from flash (yolo11n's
         * 147 KB 3x3 pad-1 head ran 6.4 c/MAC). Stage the padding once and
         * run the dense OC-panel kernel per tile. Filters that fit L2 stay
         * on im2col, whose contiguous window dot has the higher peak
         * throughput (32 B/iteration, double-buffered). */
        esp_nn_conv_s8_tiled(input_dims, input, filter_dims, filter_data, bias,
                             output_dims, out_data, conv_params, quant_data,
                             scratch_buffer);
    } else if (filter_wd * filter_ht * input_dims->channels >= 16) {
        /* Small in_ch but window_len >= 16: use im2col for zero-waste PIE.
         * Also handles padded cases naturally. */
        if (!conv_pie_rows_split(esp_nn_conv_s8_im2col, input_dims, input,
                                 filter_dims, filter_data, bias, output_dims,
                                 out_data, conv_params, quant_data,
                                 scratch_buffer)) {
            esp_nn_conv_s8_im2col(input_dims, input, filter_dims, filter_data, bias,
                                  output_dims, out_data, conv_params, quant_data,
                                  scratch_buffer);
        }
    } else if (pad_wd != 0 || pad_ht != 0) {
        /* Padded case with very small window: use tiled path */
        esp_nn_conv_s8_tiled(input_dims, input, filter_dims, filter_data, bias,
                             output_dims, out_data, conv_params, quant_data,
                             scratch_buffer);
    } else {
        /* Tiny output: fall back to generic opt */
        esp_nn_conv_s8_opt(input_dims, input, filter_dims, filter_data, bias,
                           output_dims, out_data, conv_params, quant_data);
    }
}
