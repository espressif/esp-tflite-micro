/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * 1x1 convolution for ESP32-S3 using transpose + parallel MAC.
 * Processes 8 spatial positions simultaneously via QACC lanes.
 */

#include <stdint.h>
#include <string.h>
#include <esp_nn_defs.h>
#include <common_functions.h>

int esp_nn_conv_s8_1x1_scratch_size(int in_channels)
{
    /* 128 bytes per input channel group, and all groups are held at once.
     * Depends on in_channels, not out_channels. */
    const int ch8 = in_channels / 8;
    return ch8 * 128 + 64;      /* transpose + alignment */
}

/*
 * Transpose 8 spatial positions × 8 channels from int8 to int16 with offset.
 * C fallback for when input address is not 8-byte aligned.
 */
static inline void transpose_8x8_s16_c(const int8_t *input, int stride,
                                         int32_t input_offset, int16_t *out_buf)
{
    for (int ch = 0; ch < 8; ch++) {
        for (int pos = 0; pos < 8; pos++) {
            out_buf[ch * 8 + pos] = (int16_t)(input[pos * stride + ch] + input_offset);
        }
    }
}

/*
 * SIMD transpose: 8 positions × 8 channels → channel-major int16 with offset.
 * Uses vzip.8/16/32 chain (same as original .S transpose, verified correct).
 *
 * Input: 8 consecutive spatial positions, each `stride` bytes apart.
 *        Input address MUST be 8-byte aligned.
 * Output: int16 buffer [ch0: pos0..pos7, ch1: pos0..pos7, ...] (16-byte aligned)
 */
static inline void transpose_8x8_s16_simd(const int8_t *input, int stride,
                                            int16_t offset16, int16_t *out_buf)
{
    const int8_t *p = input;
    int16_t *out = out_buf;
    int16_t *off_ptr = &offset16;

    __asm__ volatile(
        /* Load input_offset broadcast to all 8 int16 lanes */
        "ee.vldbc.16 q5, %[off]\n"
        /* Zero register for sign extension comparisons */
        "ee.zero.q q7\n"

        /* Load 8 positions × 8 channels into q0-q3 using paired l/h loads.
         * Each vld.l.64.xp loads 8 bytes (1 position) into low half, advances by stride.
         * Each vld.h.64.xp loads 8 bytes into high half, advances by stride.
         * Result: q0=[pos0|pos2], q1=[pos1|pos3], q2=[pos4|pos6], q3=[pos5|pos7] */
        "ee.vld.l.64.xp q0, %[p], %[s]\n"
        "ee.vld.l.64.xp q1, %[p], %[s]\n"
        "ee.vld.h.64.xp q0, %[p], %[s]\n"
        "ee.vld.h.64.xp q1, %[p], %[s]\n"
        "ee.vld.l.64.xp q2, %[p], %[s]\n"
        "ee.vzip.8 q0, q1\n"
        "ee.vld.l.64.xp q3, %[p], %[s]\n"
        "ee.vld.h.64.xp q2, %[p], %[s]\n"
        "ee.vld.h.64.ip q3, %[p], 0\n"
        "ee.vzip.16 q0, q1\n"
        "ee.vzip.8 q2, q3\n"
        "ee.vzip.16 q2, q3\n"
        "ee.vzip.32 q0, q2\n"

        /* First 4 channels: sign-extend q0→(q0,q6), q2→(q2,q4), add offset, store */
        "ee.vcmp.lt.s8 q4, q2, q7\n"
        "ee.vzip.8 q2, q4\n"
        "ee.vcmp.lt.s8 q6, q0, q7\n"
        "ee.vzip.8 q0, q6\n"
        "ee.vadds.s16 q0, q0, q5\n"
        "ee.vst.128.ip q0, %[out], 16\n"
        "ee.vadds.s16 q6, q6, q5\n"
        "ee.vst.128.ip q6, %[out], 16\n"
        "ee.vadds.s16 q2, q2, q5\n"
        "ee.vst.128.ip q2, %[out], 16\n"
        "ee.vadds.s16 q4, q4, q5\n"
        "ee.vst.128.ip q4, %[out], 16\n"

        /* Last 4 channels: sign-extend q1→(q1,q6), q3→(q3,q4), add offset, store */
        "ee.vzip.32 q1, q3\n"
        "ee.vcmp.lt.s8 q4, q3, q7\n"
        "ee.vzip.8 q3, q4\n"
        "ee.vcmp.lt.s8 q6, q1, q7\n"
        "ee.vzip.8 q1, q6\n"
        "ee.vadds.s16 q1, q1, q5\n"
        "ee.vst.128.ip q1, %[out], 16\n"
        "ee.vadds.s16 q6, q6, q5\n"
        "ee.vst.128.ip q6, %[out], 16\n"
        "ee.vadds.s16 q3, q3, q5\n"
        "ee.vst.128.ip q3, %[out], 16\n"
        "ee.vadds.s16 q4, q4, q5\n"
        "ee.vst.128.ip q4, %[out], 16\n"

        : [p] "+r" (p), [out] "+r" (out), [off] "+r" (off_ptr)
        : [s] "r" (stride)
        : "memory"
    );
}

/*
 * MAC 8 filter channels against 8 positions using QACC.
 * data_buf: [ch0: 8 int16, ch1: 8 int16, ...] = 128 bytes, 16-byte aligned
 * filter: 8 int8 values, sign-extended to int16 internally
 * Accumulates into QACC lanes 0-7 (must be zeroed before first call per oc)
 *
 * NOTE: filter pointer may not be 8-byte aligned, so we copy to an aligned
 * local buffer before using ee.vld.l.64.ip (which ignores unaligned address bits).
 */
static inline void mac_8pos_8ch_simd(const int16_t *data_buf, const int8_t *filter)
{
    /* Copy filter to aligned buffer — ee.vld.l.64.ip requires 8-byte alignment */
    int8_t __attribute__((aligned(16))) f_aligned[16];
    memcpy(f_aligned, filter, 8);

    const int16_t *dp = data_buf;
    const int8_t *fp = f_aligned;
    __asm__ volatile(
        /* Sign-extend filter: load 8 int8 → 8 int16 in q7 */
        "ee.zero.q q5\n"
        "ee.vld.l.64.ip q7, %[f], 0\n"
        /* Pre-load first two data chunks during sign extension */
        "ee.vld.128.ip q0, %[d], 16\n"
        "ee.vld.128.ip q1, %[d], 16\n"
        "ee.vcmp.lt.s8 q6, q7, q5\n"
        "ee.vzip.8 q7, q6\n"

        /* Pipelined: MAC current + load next in one instruction */
        "ee.vsmulas.s16.qacc.ld.incp q2, %[d], q0, q7, 0\n"
        "ee.vsmulas.s16.qacc.ld.incp q3, %[d], q1, q7, 1\n"
        "ee.vsmulas.s16.qacc.ld.incp q0, %[d], q2, q7, 2\n"
        "ee.vsmulas.s16.qacc.ld.incp q1, %[d], q3, q7, 3\n"
        "ee.vsmulas.s16.qacc.ld.incp q2, %[d], q0, q7, 4\n"
        "ee.vsmulas.s16.qacc.ld.incp q3, %[d], q1, q7, 5\n"
        /* Last two: plain MAC, no more data to load */
        "ee.vsmulas.s16.qacc q2, q7, 6\n"
        "ee.vsmulas.s16.qacc q3, q7, 7\n"
        : [d] "+r" (dp), [f] "+r" (fp)
        :
        : "memory"
    );
}

void esp_nn_conv_s8_1x1(const int8_t *input,
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
                         const int32_t *row_addend)
{
    const int size = input_wd * input_ht;
    const int ch8 = in_channels / 8;

    /* SIMD transpose requires 8-byte aligned input; check once */
    const int use_simd_transpose = (in_channels % 8 == 0) &&
                                    (((uintptr_t)input & 7) == 0);
    const int16_t offset16 = (int16_t)input_offset;

    /* Use scratch buffer for transpose data — holds ALL channel groups at once.
     * Layout: [cg0: 8 int16 × 8 pos, cg1: 8 int16 × 8 pos, ...] = ch8 × 128 bytes.
     * Aligned to 16 bytes for SIMD loads. */
    int16_t *tbuf = (int16_t *)((uintptr_t)((int8_t *)scratch + 15) & ~15);

    int pos = 0;
    for (; pos + 7 < size; pos += 8) {
        const int8_t *in_base = input + pos * in_channels;

        /* Transpose ALL channel groups ONCE per position batch.
         * This is the key optimization — reuse transposed data across all out_channels. */
        for (int cg = 0; cg < ch8; cg++) {
            int16_t *cg_buf = tbuf + cg * 64; /* 64 int16 per channel group */
            if (use_simd_transpose) {
                transpose_8x8_s16_simd(in_base + cg * 8, in_channels,
                                        offset16, cg_buf);
            } else {
                transpose_8x8_s16_c(in_base + cg * 8, in_channels,
                                     input_offset, cg_buf);
            }
        }
        __asm__ volatile("" ::: "memory");

        for (int oc = 0; oc < out_channels; oc++) {
            const int8_t *filt = filter_data + oc * in_channels;

            /* MAC across all channel groups using pre-transposed data */
            __asm__ volatile("ee.zero.qacc");

            for (int cg = 0; cg < ch8; cg++) {
                mac_8pos_8ch_simd(tbuf + cg * 64, filt + cg * 8);
            }

            /* Extract QACC → 8 int32 values */
            int32_t qacc[8];
            {
                int8_t __attribute__((aligned(16))) qraw[24];
                int8_t *qp = qraw;

                __asm__ volatile(
                    "ee.st.qacc_l.l.128.ip %[p], 16\n"
                    "ee.st.qacc_l.h.32.ip  %[p], -16\n"
                    : [p] "+r" (qp) : : "memory"
                );
                qacc[0] = *(int32_t *)(qraw + 0);
                qacc[1] = *(int32_t *)(qraw + 5);
                qacc[2] = *(int32_t *)(qraw + 10);
                qacc[3] = *(int32_t *)(qraw + 15);

                qp = qraw;
                __asm__ volatile(
                    "ee.st.qacc_h.l.128.ip %[p], 16\n"
                    "ee.st.qacc_h.h.32.ip  %[p], -16\n"
                    : [p] "+r" (qp) : : "memory"
                );
                qacc[4] = *(int32_t *)(qraw + 0);
                qacc[5] = *(int32_t *)(qraw + 5);
                qacc[6] = *(int32_t *)(qraw + 10);
                qacc[7] = *(int32_t *)(qraw + 15);
            }

            /* Remainder channels (scalar) */
            for (int c = ch8 * 8; c < in_channels; c++) {
                int16_t f = (int16_t)filt[c];
                for (int p = 0; p < 8; p++) {
                    qacc[p] += ((int32_t)in_base[p * in_channels + c] + input_offset) * f;
                }
            }

            /* Bias + requant + store for 8 positions */
            for (int p = 0; p < 8; p++) {
                int32_t acc = qacc[p];
                if (bias) acc += bias[oc];
                if (row_addend != NULL) {
                    acc += row_addend[pos + p];
                }
                acc = esp_nn_multiply_by_quantized_mult(acc, out_mult[oc], out_shift[oc]);
                acc += out_offset;
                acc = max(acc, activation_min);
                acc = min(acc, activation_max);
                out_data[(pos + p) * out_channels + oc] = (int8_t)acc;
            }
        }
    }

    /* Leftover positions (< 8 remaining) */
    for (; pos < size; pos++) {
        const int8_t *in_ptr = input + pos * in_channels;
        for (int oc = 0; oc < out_channels; oc++) {
            const int8_t *filt = filter_data + oc * in_channels;
            int32_t acc = 0;
            int c = 0;
            for (; c + 2 < in_channels; c += 3) {
                acc += ((int32_t)in_ptr[c]     + input_offset) * (int32_t)filt[c];
                acc += ((int32_t)in_ptr[c + 1] + input_offset) * (int32_t)filt[c + 1];
                acc += ((int32_t)in_ptr[c + 2] + input_offset) * (int32_t)filt[c + 2];
            }
            for (; c < in_channels; c++) {
                acc += ((int32_t)in_ptr[c] + input_offset) * (int32_t)filt[c];
            }
            if (bias) acc += bias[oc];
            if (row_addend != NULL) {
                acc += row_addend[pos];
            }
            acc = esp_nn_multiply_by_quantized_mult(acc, out_mult[oc], out_shift[oc]);
            acc += out_offset;
            acc = max(acc, activation_min);
            acc = min(acc, activation_max);
            out_data[pos * out_channels + oc] = (int8_t)acc;
        }
    }
}
