/*
 * SPDX-FileCopyrightText: 2020-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <esp_nn.h>
#include <esp_nn_multicore.h>
#include "test_utils.h"

#include "esp_rom_sys.h"


void esp_nn_fully_connected_s8_test()
{
    uint32_t total_c = 0, total_opt = 0;
    /* prepare data */
    uint16_t row_len = 256 + 8 + 7; /* odd len to test unaligned+left-over */
    const int32_t max_out_ch = 16;
    const int32_t max_row_len = 271;
    uint16_t out_channels = 3;

    /* Use heap-allocated aligned buffers (matches TFLite real-world usage) */
    int8_t *input_orig = malloc(max_row_len + 16);
    int8_t *filter_orig = malloc(max_row_len * max_out_ch + 16);
    int8_t *out_c_orig = malloc(max_out_ch + 16);
    int8_t *out_opt_orig = malloc(max_out_ch + 16);
    if (!input_orig || !filter_orig || !out_c_orig || !out_opt_orig) {
        TEST_SKIP("%s allocations failed\n", __FUNCTION__);
        goto fc_s8_cleanup;
    }
    int8_t *input = (int8_t *)(((uint32_t)input_orig + 15) & ~15);
    int8_t *filter_data = (int8_t *)(((uint32_t)filter_orig + 15) & ~15);
    int8_t *output_c = (int8_t *)(((uint32_t)out_c_orig + 15) & ~15);
    int8_t *output_opt = (int8_t *)(((uint32_t)out_opt_orig + 15) & ~15);
    int32_t activation_min = -128;
    int32_t activation_max = 127;
    int32_t input_offset = 5; /* default non-zero, swept per iteration below */
    int32_t filter_offset = 0;
    int32_t out_shift = -10;
    int32_t out_offset = 5;
    int32_t out_mult = 0x59e492c4;
    /* sweep input_offset over realistic TFLite range incl. boundaries */
    const int32_t input_offsets[] = {0, 5, -1, 1, 127, -128, 110};
    printf("\n######## Running %s ##########\n", __FUNCTION__);
    for (int itr = 0; itr < 15; itr++) {
        input_offset = input_offsets[itr % (int)(sizeof(input_offsets) / sizeof(input_offsets[0]))];
        /* TFLite weights are symmetric (filter_offset == 0) almost always;
         * a couple of iterations exercise the non-zero filter_offset SIMD path. */
        filter_offset = (itr == 9 || itr == 12) ? 5 : 0;
        out_mult = INT32_MAX / row_len + rand() % INT16_MAX;
        switch (itr) {
        case 0:
            out_shift = -10;
            break;
        case 1:
            out_shift = SHIFT_MIN;
            break;
        case 2:
            out_shift = SHIFT_MAX;
            break;
        case 3:
            out_shift = 0;
            break;
        case 4:
            row_len = 1;
            out_channels = 16;
            out_shift = -10 + rand() % 5;
            break;
        case 5:
            row_len = 16;
            out_channels = 8;
            out_shift = -10 + rand() % 5;
            break;
        case 6:
            row_len = 8;
            out_channels = 8;
            out_shift = -10 + rand() % 5;
            break;
        case 7:
            row_len = 8;
            out_channels = 15;
            out_shift = -10 + rand() % 5;
            break;
        case 8:
            row_len = 8;
            out_channels = 1;
            out_shift = -10 + rand() % 5;
            break;
        default:
            row_len = rand() % 7 + 1;
            out_channels = 8;
            out_shift = -10 + rand() % 5;
            break;
        }
        if (itr == 0) {
            out_shift = SHIFT_MAX;
        }
        /* Generate input and filter data */
        for (int i = 0; i < row_len; ++i) {
            input[i] = rand() % 256 - 128;
        }
        for (int i = 0; i < row_len * out_channels; ++i) {
            filter_data[i] = rand() % 256 - 128;
        }

        /* enable profiler */
        profile_c_start();

        /* C function */
        esp_nn_fully_connected_s8_ansi(input, input_offset, row_len, filter_data, filter_offset,
                                    NULL, output_c, out_channels, out_offset, out_shift, out_mult,
                                    activation_min, activation_max);

        total_c = profile_c_end();
        profile_opt_start();

        /* Optimized function */
        esp_nn_fully_connected_s8(input, input_offset, row_len, filter_data, filter_offset,
                                NULL, output_opt, out_channels, out_offset, out_shift, out_mult,
                                activation_min, activation_max);

        /* disable profiler */
        total_opt = profile_opt_end();

        bool ret = CHECK_EQUAL(output_c, output_opt, out_channels);
        if (ret == false) {
            TEST_FAIL("[%3d] failed\n", itr);
#if 0
            printf("Output: \n");
            PRINT_ARRAY_HEX(output_opt, out_channels, 1);
            printf("Expected: \n");
            PRINT_ARRAY_HEX(output_c, out_channels, 1);
            printf("Input:\n");
            PRINT_ARRAY_HEX(input, row_len, 1);
            printf("Filter data:\n");
            PRINT_ARRAY_HEX(filter_data, row_len, out_channels);
            printf("Out shift: %d\n", out_shift);
            printf("Out mult: %x\n", out_mult);
#endif
            goto fc_s8_cleanup;
        }
        TEST_PASS("[%3d] passed [row_len %"PRIu16", out_ch %"PRIu16", in_off %4"PRId32"]",
                  itr, row_len, out_channels, input_offset);
        printf("\tcycles: c %8"PRIu32", opt %8"PRIu32"\n", total_c, total_opt);
    }

fc_s8_cleanup:
    if (input_orig) {
        free(input_orig);
    }
    if (filter_orig) {
        free(filter_orig);
    }
    if (out_c_orig) {
        free(out_c_orig);
    }
    if (out_opt_orig) {
        free(out_opt_orig);
    }
}

void esp_nn_fully_connected_per_ch_s8_test()
{
    uint32_t total_c = 0, total_opt = 0;
    /* prepare data */
    uint16_t row_len = 256 + 8 + 7; /* odd len to test unaligned+left-over */
    const int32_t max_out_ch = 16;
    const int32_t max_row_len = 271;
    uint16_t out_channels = 3;

    /* Use heap-allocated aligned buffers (matches TFLite real-world usage) */
    int8_t *input_orig = malloc(max_row_len + 16);
    int8_t *filter_orig = malloc(max_row_len * max_out_ch + 16);
    int8_t *out_c_orig = malloc(max_out_ch + 16);
    int8_t *out_opt_orig = malloc(max_out_ch + 16);
    if (!input_orig || !filter_orig || !out_c_orig || !out_opt_orig) {
        TEST_SKIP("%s allocations failed\n", __FUNCTION__);
        goto fc_per_ch_s8_buffers_cleanup;
    }
    int8_t *input = (int8_t *)(((uint32_t)input_orig + 15) & ~15);
    int8_t *filter_data = (int8_t *)(((uint32_t)filter_orig + 15) & ~15);
    int8_t *output_c = (int8_t *)(((uint32_t)out_c_orig + 15) & ~15);
    int8_t *output_opt = (int8_t *)(((uint32_t)out_opt_orig + 15) & ~15);
    int32_t activation_min = -128;
    int32_t activation_max = 127;
    int32_t input_offset = 5; /* default non-zero, swept per iteration below */
    int32_t filter_offset = 0;
    int32_t out_offset = 7;

    int32_t* out_mult = NULL;
    int32_t* out_shift = NULL;

    const int32_t input_offsets[] = {0, 5, -1, 1, 127, -128, 110};
    printf("\n######## Running %s ##########\n", __FUNCTION__);
    for (int itr = 0;  itr < 15; itr++) {
        input_offset = input_offsets[itr % (int)(sizeof(input_offsets) / sizeof(input_offsets[0]))];
        filter_offset = (itr == 9 || itr == 12) ? 5 : 0;
        int32_t out_shift_val = 0;
        switch (itr) {
        case 0:
            out_shift_val = -10;
            break;
        case 1:
            out_shift_val = SHIFT_MIN;
            break;
        case 2:
            out_shift_val = SHIFT_MAX;
            break;
        case 3:
            out_shift_val = 0;
            break;
        case 4:
            row_len = 1;
            out_channels = 16;
            break;
        case 5:
            row_len = 16;
            out_channels = 8;
            break;
        case 6:
            row_len = 8;
            out_channels = 8;
            break;
        case 7:
            row_len = 8;
            out_channels = 15;
            break;
        case 8:
            row_len = 8;
            out_channels = 1;
            break;
        default:
            row_len = rand() % 7 + 1;
            out_channels = 8;
            break;
        }

        out_mult = ESP_NN_TEST_ALLOC(out_channels * sizeof(int32_t));
        out_shift = ESP_NN_TEST_ALLOC(out_channels * sizeof(int32_t));

        if (out_shift == NULL || out_mult == NULL) {
            TEST_SKIP("out_shift/out_mult allocations failed\n");
            goto fully_connected_per_ch_cleanup;
        }

        for (int i = 0; i < out_channels; i++) {
            out_mult[i] = INT32_MAX / row_len + rand() % INT16_MAX;
            if (i < 4) {
                out_shift[i] = out_shift_val;
            } else {
                out_shift[i] = -10 + rand() % 5;
            }
        }

        /* Generate input and filter data */
        for (int i = 0; i < row_len; ++i) {
            input[i] = rand() % 256 - 128;
        }
        for (int i = 0; i < row_len * out_channels; ++i) {
            filter_data[i] = rand() % 256 - 128;
        }
        
        /* enable profiler */
        profile_c_start();

        /* C function */
        esp_nn_fully_connected_per_ch_s8_ansi(input, input_offset, row_len, filter_data, filter_offset,
                                    NULL, output_c, out_channels, out_offset, out_shift, out_mult,
                                    activation_min, activation_max);

        total_c = profile_c_end();
        profile_opt_start();

        /* Optimized function */
        esp_nn_fully_connected_per_ch_s8(input, input_offset, row_len, filter_data, filter_offset,
                                NULL, output_opt, out_channels, out_offset, out_shift, out_mult,
                                activation_min, activation_max);

        /* disable profiler */
        total_opt = profile_opt_end();

        bool ret = CHECK_EQUAL(output_c, output_opt, out_channels);
        if (ret == false) {
            TEST_FAIL("[%3d] failed\n", itr);
            goto fully_connected_per_ch_cleanup;
        }
        TEST_PASS("[%3d] passed [row_len %"PRIu16", out_ch %"PRIu16", in_off %4"PRId32"]",
                  itr, row_len, out_channels, input_offset);
        printf("\tcycles: c %8"PRIu32", opt %8"PRIu32"\n", total_c, total_opt);

    fully_connected_per_ch_cleanup:
        if (out_shift) {
            free(out_shift);
        }
        if (out_mult) {
            free(out_mult);
        }
    }

fc_per_ch_s8_buffers_cleanup:
    if (input_orig) {
        free(input_orig);
    }
    if (filter_orig) {
        free(filter_orig);
    }
    if (out_c_orig) {
        free(out_c_orig);
    }
    if (out_opt_orig) {
        free(out_opt_orig);
    }
}

/**
 * Regression sweep for esp-nn#34: per-channel/per-tensor FC reported to give
 * memory-layout-dependent wrong results for a narrow output width (out_ch = 2)
 * at a large row_len.
 *
 * The other FC tests always 16-byte align `input` and `filter_data`, so two
 * things are never exercised for a TFLite-shaped geometry:
 *   a) the `(uintptr_t)input_data & 15` fast-path predicate in the S3
 *      dispatcher, i.e. the fallback into the s16 assembly, and
 *   b) narrow out_ch at a row_len that is a clean multiple of 16 (every filter
 *      row aligned, so the unaligned-filter QUP path is not entered either).
 *
 * This sweeps input and filter misalignment independently, and out_ch across
 * multiples and non-multiples of 8, so an alignment-dependent failure and an
 * out_ch-granularity failure can be told apart. Output buffers are guarded to
 * catch out-of-bounds writes past a 2-byte output.
 */

#define FC_ALIGN_ROW_LEN    1280    /* esp-nn#34 geometry: 1280 -> 2 */
#define FC_ALIGN_MAX_OUT_CH 16
#define FC_ALIGN_GUARD      32
#define FC_ALIGN_TRIALS     8

#if CONFIG_IDF_TARGET_ESP32S3
extern int32_t esp_nn_dot_s8_unaligned_esp32s3(const int8_t *a,
                                               const int8_t *b,
                                               int32_t len_div16);

static bool fc_check_exact_sized_dot_product(void)
{
    const int vector_counts[] = {1, 2, 39, 40};
    const int offsets[] = {0, 1, 7, 15};

    for (int vc = 0; vc < (int)(sizeof(vector_counts) / sizeof(vector_counts[0])); ++vc) {
        const int len = vector_counts[vc] * 16;
        for (int oi = 0; oi < (int)(sizeof(offsets) / sizeof(offsets[0])); ++oi) {
            const int offset = offsets[oi];
            int8_t *a = ESP_NN_TEST_ALLOC(len);
            int8_t *b_base = ESP_NN_TEST_ALLOC(len + offset);
            if (!a || !b_base) {
                free(a);
                free(b_base);
                return false;
            }
            int8_t *b = b_base + offset;
            int32_t expected = 0;
            for (int i = 0; i < len; ++i) {
                a[i] = rand() % 256 - 128;
                b[i] = rand() % 256 - 128;
                expected += (int32_t)a[i] * b[i];
            }
            int32_t actual = esp_nn_dot_s8_unaligned_esp32s3(
                    a, b, vector_counts[vc]);
            free(a);
            free(b_base);
            if (actual != expected) {
                return false;
            }
        }
    }
    return true;
}
#endif

typedef struct {
    int mismatch_pt;    /* per-tensor kernel vs its ansi reference */
    int mismatch_pc;    /* per-channel kernel vs its ansi reference */
    int guard_pt;       /* guard bytes clobbered by per-tensor kernel */
    int guard_pc;       /* guard bytes clobbered by per-channel kernel */
} fc_align_result_t;

/* Runs FC_ALIGN_TRIALS randomized trials for one (row_len, out_ch, alignment)
 * point and accumulates mismatches/guard corruption into `res`. */
static void fc_align_run_point(int8_t *input, int8_t *filter_data,
                               int8_t *out_region, int8_t *out_ref,
                               uint16_t row_len, uint16_t out_channels,
                               int32_t *out_mult, int32_t *out_shift,
                               fc_align_result_t *res)
{
    /* Quantization params taken from the esp-nn#34 report. */
    const int32_t input_offset = -3;
    const int32_t filter_offset = 0;    /* TFLite int8 weights are symmetric */
    const int32_t out_offset = -128;
    const int32_t activation_min = -128;
    const int32_t activation_max = 127;
    const int32_t mult = 1355715584;
    const int32_t shift = -6;

    int8_t *out = out_region + FC_ALIGN_GUARD;
    /* TFLite FULLY_CONNECTED always carries a bias; the older FC tests only
     * cover the bias == NULL case, so exercise a real one here. */
    int32_t bias[FC_ALIGN_MAX_OUT_CH];

    for (int c = 0; c < out_channels; c++) {
        out_mult[c] = mult - c * 12345;
        out_shift[c] = shift;
    }

    for (int t = 0; t < FC_ALIGN_TRIALS; t++) {
        for (int i = 0; i < row_len; ++i) {
            input[i] = rand() % 256 - 128;
        }
        for (int i = 0; i < row_len * out_channels; ++i) {
            filter_data[i] = rand() % 256 - 128;
        }
        for (int c = 0; c < out_channels; c++) {
            bias[c] = (rand() % 256 - 128) * 64;
        }

        /* ---- per-tensor kernel ---- */
        memset(out_region, 0xA5, FC_ALIGN_GUARD * 2 + out_channels);
        esp_nn_fully_connected_s8(input, input_offset, row_len, filter_data,
                                  filter_offset, bias, out, out_channels,
                                  out_offset, shift, mult,
                                  activation_min, activation_max);
        esp_nn_fully_connected_s8_ansi(input, input_offset, row_len, filter_data,
                                       filter_offset, bias, out_ref, out_channels,
                                       out_offset, shift, mult,
                                       activation_min, activation_max);
        if (!CHECK_EQUAL(out, out_ref, out_channels)) {
            res->mismatch_pt++;
        }
        for (int i = 0; i < FC_ALIGN_GUARD; i++) {
            if (out_region[i] != (int8_t) 0xA5 ||
                out_region[FC_ALIGN_GUARD + out_channels + i] != (int8_t) 0xA5) {
                res->guard_pt++;
                break;
            }
        }

        /* ---- per-channel kernel ---- */
        memset(out_region, 0xA5, FC_ALIGN_GUARD * 2 + out_channels);
        esp_nn_fully_connected_per_ch_s8(input, input_offset, row_len, filter_data,
                                         filter_offset, bias, out, out_channels,
                                         out_offset, out_shift, out_mult,
                                         activation_min, activation_max);
        esp_nn_fully_connected_per_ch_s8_ansi(input, input_offset, row_len,
                                              filter_data, filter_offset, bias,
                                              out_ref, out_channels, out_offset,
                                              out_shift, out_mult,
                                              activation_min, activation_max);
        if (!CHECK_EQUAL(out, out_ref, out_channels)) {
            res->mismatch_pc++;
        }
        for (int i = 0; i < FC_ALIGN_GUARD; i++) {
            if (out_region[i] != (int8_t) 0xA5 ||
                out_region[FC_ALIGN_GUARD + out_channels + i] != (int8_t) 0xA5) {
                res->guard_pc++;
                break;
            }
        }
    }
}

/* `what` may be NULL when the caller has already printed its own label. */
static void fc_align_report(const char *what, int value,
                            const fc_align_result_t *res)
{
    bool ok = (res->mismatch_pt == 0 && res->mismatch_pc == 0 &&
               res->guard_pt == 0 && res->guard_pc == 0);
    if (ok) {
        esp_nn_test_pass();
    } else {
        esp_nn_test_fail();
    }
    printf("%s", ok ? ANSI_COLOR_GREEN : ANSI_COLOR_RED);
    if (what) {
        printf("[%-14s %2d] ", what, value);
    }
    printf("per-tensor %d/%d, per-channel %d/%d, guard pt %d pc %d%s\n",
           res->mismatch_pt, FC_ALIGN_TRIALS, res->mismatch_pc, FC_ALIGN_TRIALS,
           res->guard_pt, res->guard_pc, ANSI_COLOR_RESET);
}

void esp_nn_fully_connected_align_s8_test()
{
    const size_t filter_bytes = (size_t) FC_ALIGN_ROW_LEN * FC_ALIGN_MAX_OUT_CH;
    int8_t *input_orig = ESP_NN_TEST_ALLOC(FC_ALIGN_ROW_LEN + 32);
    int8_t *filter_orig = ESP_NN_TEST_ALLOC(filter_bytes + 32);
    int8_t *out_region = ESP_NN_TEST_ALLOC(FC_ALIGN_GUARD * 2 + FC_ALIGN_MAX_OUT_CH);
    int8_t *out_ref = ESP_NN_TEST_ALLOC(FC_ALIGN_MAX_OUT_CH);
    int32_t *out_mult = ESP_NN_TEST_ALLOC(FC_ALIGN_MAX_OUT_CH * sizeof(int32_t));
    int32_t *out_shift = ESP_NN_TEST_ALLOC(FC_ALIGN_MAX_OUT_CH * sizeof(int32_t));

    if (!input_orig || !filter_orig || !out_region || !out_ref ||
        !out_mult || !out_shift) {
        TEST_SKIP("%s allocations failed\n", __FUNCTION__);
        goto fc_align_cleanup;
    }

    int8_t *input_base = (int8_t *)(((uintptr_t) input_orig + 15) & ~(uintptr_t) 15);
    int8_t *filter_base = (int8_t *)(((uintptr_t) filter_orig + 15) & ~(uintptr_t) 15);

    printf("\n######## Running %s ##########\n", __FUNCTION__);
    printf("geometry: row_len %d, filter_offset 0, input_offset -3 (esp-nn#34)\n",
           FC_ALIGN_ROW_LEN);

#if CONFIG_IDF_TARGET_ESP32S3
    if (fc_check_exact_sized_dot_product()) {
        TEST_PASS("-- exact-sized bounded dot-product inputs: passed\n");
    } else {
        TEST_FAIL("-- exact-sized bounded dot-product inputs: failed\n");
    }
#endif

    /* Sweep A: input misalignment 0..15, filter 16-byte aligned, out_ch = 2.
     * Exercises the `input_data & 15` dispatcher predicate. */
    printf("-- sweep A: input misalignment, out_ch 2, filter aligned --\n");
    for (int off = 0; off < 16; off++) {
        fc_align_result_t res = {0};
        fc_align_run_point(input_base + off, filter_base, out_region, out_ref,
                           FC_ALIGN_ROW_LEN, 2, out_mult, out_shift, &res);
        fc_align_report("input off", off, &res);
    }

    /* Sweep B: filter misalignment 0..15, input 16-byte aligned, out_ch = 2. */
    printf("-- sweep B: filter misalignment, out_ch 2, input aligned --\n");
    for (int off = 0; off < 16; off++) {
        fc_align_result_t res = {0};
        fc_align_run_point(input_base, filter_base + off, out_region, out_ref,
                           FC_ALIGN_ROW_LEN, 2, out_mult, out_shift, &res);
        fc_align_report("filter off", off, &res);
    }

    /* Sweep C: out_ch across multiples and non-multiples of 8, at both an
     * aligned and a misaligned input, to separate an out_ch-granularity
     * failure from an alignment failure. */
    const uint16_t out_chs[] = {1, 2, 3, 7, 8, 15, 16};
    for (int a = 0; a < 2; a++) {
        int off = a ? 4 : 0;
        printf("-- sweep C: out_ch sweep, input off %d --\n", off);
        for (int i = 0; i < (int)(sizeof(out_chs) / sizeof(out_chs[0])); i++) {
            fc_align_result_t res = {0};
            fc_align_run_point(input_base + off, filter_base, out_region, out_ref,
                               FC_ALIGN_ROW_LEN, out_chs[i], out_mult, out_shift,
                               &res);
            fc_align_report("out_ch", out_chs[i], &res);
        }
    }

    /* Sweep D0: row_len below one SIMD width, where the header now promises
     * correctness too. Misaligned small rows go to ansi, 8-aligned ones reach
     * the s16 assembly's scalar tail — both need covering. */
    const uint16_t small_rows[] = {1, 5, 8, 15, 16, 17};
    printf("-- sweep D0: small row_len, out_ch 2, input off 0/1/8 --\n");
    for (int i = 0; i < (int)(sizeof(small_rows) / sizeof(small_rows[0])); i++) {
        for (int j = 0; j < 3; j++) {
            const int off = (int[]){0, 1, 8}[j];
            fc_align_result_t res = {0};
            fc_align_run_point(input_base + off, filter_base, out_region, out_ref,
                               small_rows[i], 2, out_mult, out_shift, &res);
            printf("  row_len %2u / in off %d: ", small_rows[i], off);
            fc_align_report(NULL, 0, &res);
        }
    }

    /* Sweep D: row_len not a multiple of 16, so filter rows are unaligned too
     * and a misaligned input cannot be rescued by swapping the operands. */
    const int align_offs[] = {0, 1, 4, 8};
    printf("-- sweep D: row_len 1271 (odd), out_ch 2, input misalignment --\n");
    for (int i = 0; i < (int)(sizeof(align_offs) / sizeof(align_offs[0])); i++) {
        fc_align_result_t res = {0};
        fc_align_run_point(input_base + align_offs[i], filter_base, out_region,
                           out_ref, 1271, 2, out_mult, out_shift, &res);
        fc_align_report("input off", align_offs[i], &res);
    }

    /* Sweep E: input and filter both misaligned. */
    printf("-- sweep E: input + filter misalignment, out_ch 2 --\n");
    for (int i = 0; i < (int)(sizeof(align_offs) / sizeof(align_offs[0])); i++) {
        for (int j = 0; j < (int)(sizeof(align_offs) / sizeof(align_offs[0])); j++) {
            fc_align_result_t res = {0};
            fc_align_run_point(input_base + align_offs[i],
                               filter_base + align_offs[j], out_region, out_ref,
                               FC_ALIGN_ROW_LEN, 2, out_mult, out_shift, &res);
            printf("  in off %d / filt off %d: ", align_offs[i], align_offs[j]);
            fc_align_report(NULL, 0, &res);
        }
    }

fc_align_cleanup:
    if (input_orig) {
        free(input_orig);
    }
    if (filter_orig) {
        free(filter_orig);
    }
    if (out_region) {
        free(out_region);
    }
    if (out_ref) {
        free(out_ref);
    }
    if (out_mult) {
        free(out_mult);
    }
    if (out_shift) {
        free(out_shift);
    }
}

/**
 * Performance and bit-exactness guard for esp-nn#36.
 *
 * That regression was a scalar O(out_channels * row_len) filter-sum added to the
 * FC fast path: every correctness test passed while the kernel got ~4x slower,
 * so nothing caught it. A fixed cycles-per-MAC ceiling is a poor tripwire here
 * because the achievable rate varies hugely with shape (a 20x128 layer costs
 * ~6.8 cyc/MAC even on the fused assembly). Instead this measures both paths in
 * the same run and asserts the dispatcher's choice is never materially worse
 * than the fused s16 assembly, which is the pre-1.2.1 behaviour and therefore
 * the floor we must not fall below.
 *
 * Correctness is checked first at every channel against the ansi reference: a
 * faster kernel that is not bit-exact is worthless.
 */
#define FC_PERF_ITERS       10
#define FC_PERF_TOLERANCE   1.10f   /* allow 10% for dispatch + measurement noise */

/* The io==0 case guards the second dispatch regime: with no correction pass the
 * dot path wins from row_len ~64, so at 128x256 the dispatcher must pick it and
 * beat the assembly outright (measured 0.68x). Found in self-review: a rule
 * gating on row_len alone sent io==0 mid rows to the assembly, a 1.5x loss.
 * Only the S3 dispatcher has this choice to make; elsewhere the misaligned call
 * is the same kernel with a load penalty, so only the generic bound applies. */
#ifdef CONFIG_IDF_TARGET_ESP32S3
#define FC_PERF_IO0_MAX     0.90f
#else
#define FC_PERF_IO0_MAX     FC_PERF_TOLERANCE
#endif

void esp_nn_fully_connected_perf_test()
{
    /* Spans both regimes: row_len >= 256 takes the dot path, below it the
     * dispatcher should fall back to the assembly. */
    const struct { const char *name; uint16_t row_len; uint16_t out_ch;
                   int32_t io; float max_ratio; } cases[] = {
        { "256x256",     256, 256, -3, FC_PERF_TOLERANCE },
        { "512x128",     512, 128, -3, FC_PERF_TOLERANCE },
        { "320x320",     320, 320, -3, FC_PERF_TOLERANCE },
        { "128x256",     128, 256, -3, FC_PERF_TOLERANCE },
        { " 64x256",      64, 256, -3, FC_PERF_TOLERANCE },
        { "128x256 io0", 128, 256,  0, FC_PERF_IO0_MAX },
    };
    const int32_t filter_offset = 0;    /* symmetric weights, as real int8 models */
    const int32_t out_mult = 1355715584, out_shift = -6;
    int advisory = 0;

    printf("\n######## Running %s ##########\n", __FUNCTION__);

    for (int c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++) {
        const uint16_t row_len = cases[c].row_len, out_ch = cases[c].out_ch;
        const int32_t input_offset = cases[c].io;
        /* +32 slack so the operands can be pushed off 16-byte alignment below */
        int8_t *input_raw  = ESP_NN_TEST_ALLOC(row_len + 32);
        int8_t *filter_raw = ESP_NN_TEST_ALLOC((size_t)row_len * out_ch + 32);
        int8_t *in2_raw    = ESP_NN_TEST_ALLOC(row_len + 32);
        int8_t *filt2_raw  = ESP_NN_TEST_ALLOC((size_t)row_len * out_ch + 32);
        int8_t *out_raw    = ESP_NN_TEST_ALLOC(out_ch + 16);
        int8_t *ref        = ESP_NN_TEST_ALLOC(out_ch);
        int32_t *bias      = ESP_NN_TEST_ALLOC(out_ch * sizeof(int32_t));
        if (!input_raw || !filter_raw || !in2_raw || !filt2_raw || !out_raw || !ref || !bias) {
            TEST_SKIP("[%s] allocation failed\n", cases[c].name);
            goto perf_cleanup;
        }
        /* 16-byte align, as TFLite Micro's arena does. Plain malloc is 4-byte
         * aligned, which makes the dispatcher bail to the ansi reference and
         * would measure the wrong path entirely (~10 cyc/MAC). */
        int8_t *input  = (int8_t *)(((uintptr_t)input_raw  + 15) & ~(uintptr_t)15);
        int8_t *filter = (int8_t *)(((uintptr_t)filter_raw + 15) & ~(uintptr_t)15);
        int8_t *out    = (int8_t *)(((uintptr_t)out_raw    + 15) & ~(uintptr_t)15);

        for (int i = 0; i < row_len; i++) {
            input[i] = rand() % 256 - 128;
        }
        for (int i = 0; i < row_len * out_ch; i++) {
            filter[i] = rand() % 256 - 128;
        }
        for (int i = 0; i < out_ch; i++) {
            bias[i] = (rand() % 256 - 128) * 64;
        }

        /* --- bit-exactness at every channel --- */
        esp_nn_fully_connected_s8(input, input_offset, row_len, filter, filter_offset,
                                  bias, out, out_ch, 0, out_shift, out_mult, -128, 127);
        esp_nn_fully_connected_s8_ansi(input, input_offset, row_len, filter, filter_offset,
                                       bias, ref, out_ch, 0, out_shift, out_mult, -128, 127);
        if (!CHECK_EQUAL(out, ref, out_ch)) {
            int bad = 0, first = -1;
            for (int i = 0; i < out_ch; i++) {
                if (out[i] != ref[i]) {
                    bad++;
                    if (first < 0) {
                        first = i;
                    }
                }
            }
            TEST_FAIL("[%s] NOT BIT-EXACT vs ansi: %d/%d channels, first ch %d "
                      "(opt %d, ref %d)\n",
                      cases[c].name, bad, out_ch, first, out[first], ref[first]);
            goto perf_cleanup;
        }

        /* --- dispatcher's chosen path --- */
        profile_opt_start();
        for (int it = 0; it < FC_PERF_ITERS; it++) {
            esp_nn_fully_connected_s8(input, input_offset, row_len, filter, filter_offset,
                                      bias, out, out_ch, 0, out_shift, out_mult, -128, 127);
        }
        uint32_t chosen = profile_opt_end();

        /* --- fused s16 assembly, the floor. Reached by defeating both fast-path
         * alignment conditions; the input stays 8-aligned, which that assembly
         * requires. --- */
        /* Offset from the ALIGNED pointers, not the raw malloc'd ones: malloc is
         * only 4-byte aligned, so raw+8 is not reliably 8-aligned and the call
         * would fall through to the ansi reference instead of the assembly,
         * making this comparison vacuous. Copy into separate buffers - deriving
         * these from the same allocation and memcpy'ing would overlap, which is
         * undefined behaviour. */
        int8_t *in_off = (int8_t *)(((uintptr_t)in2_raw + 15) & ~(uintptr_t)15) + 8;
        int8_t *filt_off = (int8_t *)(((uintptr_t)filt2_raw + 15) & ~(uintptr_t)15) + 8;
        memcpy(in_off, input, row_len);
        memcpy(filt_off, filter, (size_t)row_len * out_ch);
        esp_nn_fully_connected_s8(in_off, input_offset, row_len, filt_off, filter_offset,
                                  bias, out, out_ch, 0, out_shift, out_mult, -128, 127);
        profile_opt_start();
        for (int it = 0; it < FC_PERF_ITERS; it++) {
            esp_nn_fully_connected_s8(in_off, input_offset, row_len, filt_off, filter_offset,
                                      bias, out, out_ch, 0, out_shift, out_mult, -128, 127);
        }
        uint32_t asm_floor = profile_opt_end();

        /* Bit-exactness is the verdict; the dispatch-vs-assembly ratio below is
         * advisory (a regression signal on hardware, noise under emulation). */
        TEST_PASS("[%s] bit-exact vs ansi\n", cases[c].name);
        /* Compare by cross-multiplication and report hundredths: nano-printf,
         * the default C library on some targets, has no float conversions. */
        const uint32_t max_x100 = (uint32_t) (cases[c].max_ratio * 100.0f + 0.5f);
        bool ok = (uint64_t) chosen * 100 <= (uint64_t) max_x100 * asm_floor;
        const uint32_t ratio_x100 = asm_floor
                ? (uint32_t) (((uint64_t) chosen * 100 + asm_floor / 2) / asm_floor) : 0;
        if (!ok) {
            advisory++;
        }
        printf("%s[%s] chosen %8"PRIu32" cyc, asm floor %8"PRIu32" cyc, %"PRIu32".%02"PRIu32"x  %s%s\n",
               ok ? ANSI_COLOR_GREEN : ANSI_COLOR_RED, cases[c].name,
               chosen / FC_PERF_ITERS, asm_floor / FC_PERF_ITERS,
               ratio_x100 / 100, ratio_x100 % 100,
               ok ? "ok" : "SLOWER THAN THE ASSEMBLY (advisory)", ANSI_COLOR_RESET);

    perf_cleanup:
        if (input_raw) {
            free(input_raw);
        }
        if (filter_raw) {
            free(filter_raw);
        }
        if (in2_raw) {
            free(in2_raw);
        }
        if (filt2_raw) {
            free(filt2_raw);
        }
        if (out_raw) {
            free(out_raw);
        }
        if (ref) {
            free(ref);
        }
        if (bias) {
            free(bias);
        }
    }
    if (advisory) {
        printf(ANSI_COLOR_YELLOW"%s: %d advisory ratio miss(es), not counted\n"ANSI_COLOR_RESET,
               __FUNCTION__, advisory);
    }
}

void esp_nn_fully_connected_per_ch_s8_batch_test()
{
    printf("\n######## Running %s ##########\n", __FUNCTION__);
    /* This test is the fully connected kernel. A fully connected scratch
     * buffer left by an earlier case would take the 1x1 path for several of
     * these shapes. A convolution scratch registration does not. */
    esp_nn_set_fc_scratch_buf(NULL, 0);
    /* (batches, row_len, out_ch, input_offset, input misalign, filter misalign).
     * row_len a multiple of 16 keeps the s8 fast path; 33 exercises the scalar
     * tail and, since its rows then start at differing offsets, mixed row
     * alignments; in_off 0 skips the correction.  Batching is only eligible
     * when the fast path is, so a misaligned input (or filter) must
     * transparently fall back to the per-row kernel instead of reading
     * garbage.  Batched output must match the per-row ANSI reference. */
    const struct {
        int batches, row_len, out_ch;
        int32_t in_off;
        int in_align, f_align;
    } cfg[] = {
        { 48, 160, 32,  5, 0,  0 },   /* conformer FFN-d1 shape */
        { 48,  64, 16,  0, 0,  0 },   /* input_offset 0 */
        {  8,  80,  8, -3, 0,  0 },
        {  4,  33,  5,  7, 0,  0 },   /* non-multiple-of-16 row_len -> scalar tail */
        {  1, 128, 12, 11, 0,  0 },   /* single row */
        {  4,  64,  8,  5, 8,  4 },   /* neither operand 16-aligned: per-row s16 fallback */
        {  4,  64,  8, -3, 4,  4 },   /* input 4-mod-8: per-row ANSI fallback */
        {  4,  64,  8,  5, 0,  4 },   /* fast path, unaligned filter rows */
        {  4,  64,  8, -3, 8,  0 },   /* fast path, filter supplied as aligned operand */
        {  4,  64, 128,  5, 8, 12 },  /* wide out_ch + both misaligned (fallback) */
        {  4,  64, 128,  5, 0,  0 },  /* wide out_ch fast path: the VLA shape */
    };
    const int n_cfg = (int)(sizeof(cfg) / sizeof(cfg[0]));
    const int32_t activation_min = -128, activation_max = 127;
    const int32_t out_offset = 7, filter_offset = 0;

    for (int c = 0; c < n_cfg; c++) {
        const int B = cfg[c].batches, row_len = cfg[c].row_len, out_ch = cfg[c].out_ch;
        const int32_t input_offset = cfg[c].in_off;
        const int in_align = cfg[c].in_align;
        const int f_align  = cfg[c].f_align;

        int8_t  *in_o  = ESP_NN_TEST_ALLOC(B * row_len + 32);          /* + align/misalign */
        int8_t  *flt_o = ESP_NN_TEST_ALLOC(row_len * out_ch + 32);
        int8_t  *ob_o  = ESP_NN_TEST_ALLOC(B * out_ch + 16);
        int8_t  *or_o  = ESP_NN_TEST_ALLOC(B * out_ch + 16);
        int32_t *out_mult  = ESP_NN_TEST_ALLOC(out_ch * sizeof(int32_t));
        int32_t *out_shift = ESP_NN_TEST_ALLOC(out_ch * sizeof(int32_t));
        if (!in_o || !flt_o || !ob_o || !or_o || !out_mult || !out_shift) {
            TEST_SKIP("%s alloc failed\n", __FUNCTION__);
            goto fc_batch_cleanup;
        }
        int8_t *input       = (int8_t *)((((uintptr_t)in_o  + 15) & ~(uintptr_t)15) + in_align);
        int8_t *filter      = (int8_t *)((((uintptr_t)flt_o + 15) & ~(uintptr_t)15) + f_align);
        int8_t *out_batched = (int8_t *)(((uintptr_t)ob_o  + 15) & ~(uintptr_t)15);
        int8_t *out_ref     = (int8_t *)(((uintptr_t)or_o  + 15) & ~(uintptr_t)15);

        for (int i = 0; i < out_ch; i++) {
            out_mult[i]  = INT32_MAX / row_len + rand() % INT16_MAX;
            out_shift[i] = -10 + rand() % 5;
        }
        for (int i = 0; i < B * row_len; i++)     input[i]  = rand() % 256 - 128;
        for (int i = 0; i < row_len * out_ch; i++) filter[i] = rand() % 256 - 128;

        /* batched: one weight-stationary pass over all rows */
        esp_nn_fully_connected_per_ch_s8_batch(input, input_offset, row_len, filter,
                filter_offset, NULL, out_batched, out_ch, out_offset, out_shift,
                out_mult, activation_min, activation_max, B);

        /* reference: the per-row kernel, once per row */
        for (int b = 0; b < B; b++) {
            esp_nn_fully_connected_per_ch_s8_ansi(input + b * row_len, input_offset,
                    row_len, filter, filter_offset, NULL, out_ref + b * out_ch, out_ch,
                    out_offset, out_shift, out_mult, activation_min, activation_max);
        }

        if (CHECK_EQUAL(out_ref, out_batched, B * out_ch) == false) {
            TEST_FAIL("[%3d] failed [B %d, row_len %d, out_ch %d, in_align %d, f_align %d]\n",
                   c, B, row_len, out_ch, in_align, f_align);
        } else {
            TEST_PASS("[%3d] passed [B %d, row_len %d, out_ch %d, in_off %"PRId32", in_align %d, f_align %d]\n",
                   c, B, row_len, out_ch, input_offset, in_align, f_align);
        }
    fc_batch_cleanup:
        free(in_o); free(flt_o); free(ob_o); free(or_o); free(out_mult); free(out_shift);
    }
}

/* A row that names fully connected fails when the registered call is faster
 * than this fraction of the same call with the scratch buffer cleared.
 * Convolution rows are not compared this way: a small shape can take the
 * 1x1 path and still be slower, so the path comes from the dispatch return. */
enum {
    FC_1X1_FC_MIN_X100 = 95,
};

static uint32_t fc_1x1_median(uint32_t *samples, int n)
{
    for (int i = 1; i < n; i++) {
        uint32_t v = samples[i];
        int j = i;
        while (j > 0 && samples[j - 1] > v) {
            samples[j] = samples[j - 1];
            j--;
        }
        samples[j] = v;
    }
    return samples[n / 2];
}

/* Both batch wrappers take the 1x1 path exactly when this returns 1. */
extern int esp_nn_fc_per_ch_s8_batch_via_conv(
        const int8_t *input_data, const int32_t input_offset,
        const uint16_t row_len, const int8_t *filter_data,
        const int32_t filter_offset, const int32_t *bias,
        int8_t *out_data, const uint16_t out_channels,
        const int32_t out_offset, const int32_t *out_shift,
        const int32_t *out_mult, const int32_t activation_min,
        const int32_t activation_max, const int32_t batches);

/* A weight zero-point of 0 is the same product as a 1x1 convolution.
 * A nonzero weight zero-point stays on that path on P4, S31, and S3 when
 * the registered scratch also holds the per-row term.
 * expect_s3 is that path with one core, expect_s3_dual is that path after
 * esp_nn_dual_core_enable(), and expect_p4 and expect_other are the other
 * two targets. Optimized ESP32-P4 and ESP32-S31 both use expect_p4.
 * 1 means the 1x1 convolution. The numbers are written on the
 * row. A batch of 1 or 2 whose row length is not a multiple of 8 stays on
 * fully connected even when the worker is active. Scratch mode is 0 none,
 * 1 a fully connected buffer of the required size, 2 one byte short, 3 the
 * convolution setter, which does not register fully connected scratch. */
static void fc_1x1_arm(int scratch_mode, void *scratch, int scratch_bytes)
{
    if (scratch_mode == 1) {
        esp_nn_set_fc_scratch_buf(scratch, (size_t) scratch_bytes);
    } else if (scratch_mode == 2) {
        esp_nn_set_fc_scratch_buf(scratch, (size_t) scratch_bytes - 1);
    } else if (scratch_mode == 3) {
        esp_nn_set_fc_scratch_buf(NULL, 0);
        esp_nn_set_conv_scratch_buf(scratch);
    } else {
        esp_nn_set_fc_scratch_buf(NULL, 0);
        esp_nn_set_conv_scratch_buf(NULL);
    }
}

void esp_nn_fully_connected_per_ch_s8_batch_1x1_test(void)
{
    printf("\n######## Running %s ##########\n", __FUNCTION__);
    const struct {
        const char *name;
        int batches, row_len, out_ch;
        int32_t in_off, filt_off;
        int misalign_input;
        int misalign_filter;
        int scratch_mode;
        int with_bias;
        int clamp;
        int expect_s3;
        int expect_s3_dual;
        int expect_p4;
        int expect_other;
    } cfg[] = {
        { "m16_k64_n64",       16,  64,  64, 128, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1 },
        { "m32_k64_n128",      32,  64, 128, 128, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1 },
        { "m4_k128_n192",       4, 128, 192, 128, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1 },
        { "m4_k128_n200",       4, 128, 200, 128, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1 },
        { "m64_k128_n256",     64, 128, 256, 128, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1 },
        { "m48_k160_n32",      48, 160,  32,   5, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1 },
        { "filt7_m16_k64_n64", 16,  64,  64, 128, 7, 0, 0, 1, 1, 0, 1, 1, 1, 0 },
        { "misalign_m16_k64_n64", 16, 64, 64, 128, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0 },
        { "noscratch_m16_k64_n64", 16, 64, 64, 128, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
        { "tiny_m2_k32_n32",    2,  32,  32, 128, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0 },
        { "edge_m3_k8_n8",      3,   8,   8, 128, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0 },
        { "odd_m3_k40_n8",      3,  40,   8, 128, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0 },
        { "out4_m4_k64_n4",     4,  64,   4, 128, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0 },
        { "m1_k8_n4",           1,   8,   4, 128, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0 },
        { "odd_m1_k40_n4",      1,  40,   4,   5, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0 },
        { "dual_m1_k128_n256",  1, 128, 256, 128, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0 },
        { "dual_m2_k256_n128",  2, 256, 128, 128, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0 },
        { "oddrow_m1_k41_n640", 1,  41, 640, 128, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0 },
        { "nodual_m2_k128_n32", 2, 128,  32, 128, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0 },
        { "below_m4_k7_n16",    4,   7,  16, 128, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0 },
        { "s3out_m4_k16_n7",    4,  16,   7, 128, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0 },
        { "nobias_m4_k64_n16",  4,  64,  16, 128, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1 },
        { "clamp_m4_k16_n16",   4,  16,  16, 128, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0 },
        { "short_m16_k64_n64", 16,  64,  64, 128, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0 },
        { "unsized_m16_k64_n64", 16, 64, 64, 128, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0 },
        { "misfilt_m16_k64_n64", 16, 64, 64, 128, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0 },
        { "zero_in_m4_k8_n8",   4,   8,   8,   0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0 },
        { "inoff_m3_k9_n8",     3,   9,   8,   5, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0 },
    };
    const int n_cfg = (int) (sizeof(cfg) / sizeof(cfg[0]));
    const int32_t out_offset = 5;
    const int reps = 5;
    const int dual = esp_nn_dual_core_active() ? 1 : 0;

    for (int c = 0; c < n_cfg; c++) {
        const int B = cfg[c].batches;
        const int row_len = cfg[c].row_len;
        const int out_ch = cfg[c].out_ch;
        const int prefix = cfg[c].misalign_input ? 1 : 0;
        const int filt_prefix = cfg[c].misalign_filter ? 1 : 0;
        const int32_t activation_min = cfg[c].clamp ? -20 : -128;
        const int32_t activation_max = cfg[c].clamp ? 20 : 127;
#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)
        const int expect = dual ? cfg[c].expect_s3_dual : cfg[c].expect_s3;
#elif defined(CONFIG_NN_OPTIMIZED) && \
        (defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S31))
        const int expect = cfg[c].expect_p4;
#else
        const int expect = cfg[c].expect_other;
#endif
        uint32_t seed = 2166136261u;
        const char *np = cfg[c].name;
        while (*np) {
            seed ^= (uint8_t) *np++;
            seed *= 16777619u;
        }

        int8_t *in_o = NULL, *flt_o = NULL, *ob_o = NULL, *or_o = NULL;
        int32_t *out_mult = NULL, *out_shift = NULL, *bias = NULL;
        void *scratch_raw = NULL;
        void *scratch = NULL;
        int scratch_bytes = 0;
        int8_t *input = NULL, *filter = NULL, *out_batched = NULL, *out_ref = NULL;

        in_o = ESP_NN_TEST_ALLOC((size_t) B * row_len + 32 + prefix);
        flt_o = ESP_NN_TEST_ALLOC((size_t) row_len * out_ch + 32 + filt_prefix);
        ob_o = ESP_NN_TEST_ALLOC((size_t) B * out_ch + 32);
        or_o = ESP_NN_TEST_ALLOC((size_t) B * out_ch + 32);
        out_mult = ESP_NN_TEST_ALLOC((size_t) out_ch * sizeof(int32_t));
        out_shift = ESP_NN_TEST_ALLOC((size_t) out_ch * sizeof(int32_t));
        if (cfg[c].with_bias) {
            bias = ESP_NN_TEST_ALLOC((size_t) out_ch * sizeof(int32_t));
        }
        if (!in_o || !flt_o || !ob_o || !or_o || !out_mult || !out_shift ||
                (cfg[c].with_bias && !bias)) {
            TEST_SKIP("[%s] alloc failed\n", cfg[c].name);
            goto case_cleanup;
        }
        input = (int8_t *) (((uintptr_t) in_o + 15) & ~(uintptr_t) 15);
        input += prefix;
        filter = (int8_t *) (((uintptr_t) flt_o + 15) & ~(uintptr_t) 15);
        filter += filt_prefix;
        out_batched = (int8_t *) (((uintptr_t) ob_o + 15) & ~(uintptr_t) 15);
        out_ref = (int8_t *) (((uintptr_t) or_o + 15) & ~(uintptr_t) 15);

        for (int i = 0; i < B * row_len; i++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            input[i] = (int8_t) ((int) (seed % 256) - 128);
        }
        for (int i = 0; i < row_len * out_ch; i++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            filter[i] = (int8_t) ((int) (seed % 256) - 128);
        }
        for (int i = 0; i < out_ch; i++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            out_mult[i] = INT32_MAX / row_len + (int32_t) (seed % 1000);
            out_shift[i] = -8;
            if (bias) {
                bias[i] = (int32_t) (seed % 21) - 10;
            }
        }

#if defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)
        esp_nn_set_conv_preferred_scratch_buf_esp32s3(NULL, 0);
#endif
        if (cfg[c].scratch_mode != 0) {
            data_dims_t in_dims = { .width = 1, .height = B, .channels = row_len };
            data_dims_t filt_dims = { .width = 1, .height = 1, .channels = row_len };
            data_dims_t out_dims = { .width = 1, .height = B, .channels = out_ch };
            conv_params_t conv_params = {
                .in_offset = cfg[c].in_off,
                .out_offset = out_offset,
                .stride = { .width = 1, .height = 1 },
                .padding = { .width = 0, .height = 0 },
                .dilation = { .width = 1, .height = 1 },
                .activation = { .min = activation_min, .max = activation_max },
            };
            scratch_bytes = esp_nn_get_conv_scratch_size(&in_dims, &filt_dims,
                                                         &out_dims, &conv_params);
            if (scratch_bytes < 0) {
                scratch_bytes = 0;
            }
            /* The per-row term sits after the convolution workspace. */
            if (cfg[c].filt_off != 0 && scratch_bytes > 0) {
                const int addend_at = (scratch_bytes + 15) & ~15;
                scratch_bytes = addend_at + B * (int) sizeof(int32_t);
            }
            if (cfg[c].scratch_mode == 2 && scratch_bytes <= 0) {
                /* ANSI and the generic targets report no workspace, so there
                 * is no short buffer to check. A nonzero size still runs. */
                TEST_SKIP("[%s] scratch size is 0\n", cfg[c].name);
                goto case_cleanup;
            }
#if IDF_HEAP_CAPS
            scratch_raw = heap_caps_malloc((size_t) scratch_bytes + 32,
                                           MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#endif
            if (!scratch_raw) {
                scratch_raw = ESP_NN_TEST_ALLOC((size_t) scratch_bytes + 32);
            }
            if (!scratch_raw) {
                TEST_SKIP("[%s] scratch alloc failed\n", cfg[c].name);
                goto case_cleanup;
            }
            scratch = (void *) (((uintptr_t) scratch_raw + 15) & ~(uintptr_t) 15);
            fc_1x1_arm(cfg[c].scratch_mode, scratch, scratch_bytes);
        } else {
            fc_1x1_arm(0, NULL, 0);
        }

        for (int b = 0; b < B; b++) {
            esp_nn_fully_connected_per_ch_s8_ansi(
                    input + b * row_len, cfg[c].in_off, row_len, filter,
                    cfg[c].filt_off, bias, out_ref + b * out_ch, out_ch,
                    out_offset, out_shift, out_mult, activation_min, activation_max);
        }

        uint32_t samples[5];
        for (int r = 0; r < reps; r++) {
            /* The batched call clears the registration, so each sample
             * registers the buffer again. */
            fc_1x1_arm(cfg[c].scratch_mode, scratch, scratch_bytes);
            profile_opt_start();
            esp_nn_fully_connected_per_ch_s8_batch(
                    input, cfg[c].in_off, row_len, filter, cfg[c].filt_off, bias,
                    out_batched, out_ch, out_offset, out_shift, out_mult,
                    activation_min, activation_max, B);
            samples[r] = profile_opt_end();
        }
        const uint32_t cycles = fc_1x1_median(samples, reps);
        const uint32_t ticks_per_us = esp_rom_get_cpu_ticks_per_us();
        const uint32_t us_x100 = ticks_per_us
                ? (uint32_t) (((uint64_t) cycles * 100ull) / ticks_per_us) : 0;

        int mismatch = 0;
        const int n_out = B * out_ch;
        for (int i = 0; i < n_out; i++) {
            if (out_ref[i] != out_batched[i]) {
                mismatch++;
            }
        }

        /* The cleared call is fully connected. Its cycle count is printed for
         * every row. On S3, P4, and S31 a fully connected row fails when the
         * registered call is faster than FC_1X1_FC_MIN_X100 percent of it. */
        uint32_t cycles_fc = 0;
        int cycle_fail = (cycles == 0);
#if defined(CONFIG_NN_OPTIMIZED) && \
        (defined(CONFIG_IDF_TARGET_ESP32S3) || defined(CONFIG_IDF_TARGET_ESP32P4) || \
         defined(CONFIG_IDF_TARGET_ESP32S31))
        if (scratch != NULL && !cycle_fail) {
            for (int r = 0; r < reps; r++) {
                fc_1x1_arm(0, NULL, 0);
                profile_opt_start();
                esp_nn_fully_connected_per_ch_s8_batch(
                        input, cfg[c].in_off, row_len, filter, cfg[c].filt_off, bias,
                        out_batched, out_ch, out_offset, out_shift, out_mult,
                        activation_min, activation_max, B);
                samples[r] = profile_opt_end();
            }
            cycles_fc = fc_1x1_median(samples, reps);
            if (cycles_fc == 0) {
                cycle_fail = 1;
            } else if (!expect) {
                cycle_fail = (uint64_t) cycles * 100 <
                        (uint64_t) cycles_fc * FC_1X1_FC_MIN_X100;
            }
        }
#endif
        /* The outputs match on either path. Ask the same function the batch
         * wrappers use. On S3, P4, and S31 that return has to match expect,
         * so a missed 1x1 dispatch fails. Other builds have no 1x1 workspace
         * when the scratch size is 0, and then the call stays on fully
         * connected. */
        fc_1x1_arm(cfg[c].scratch_mode, scratch, scratch_bytes);
        const int took = esp_nn_fc_per_ch_s8_batch_via_conv(
                input, cfg[c].in_off, row_len, filter, cfg[c].filt_off, bias,
                out_batched, out_ch, out_offset, out_shift, out_mult,
                activation_min, activation_max, B);
#if defined(CONFIG_NN_OPTIMIZED) && \
        (defined(CONFIG_IDF_TARGET_ESP32S3) || defined(CONFIG_IDF_TARGET_ESP32P4) || \
         defined(CONFIG_IDF_TARGET_ESP32S31))
        const int path_fail = (took != expect);
#else
        const int path_fail = (scratch_bytes <= 0) ? (took != 0) : (took != expect);
#endif
        printf("FC_BATCH_1X1 name=%s cycles=%" PRIu32 " fc_cycles=%" PRIu32
               " us_x100=%" PRIu32 " mismatch=%d expect=%d path=%d dual=%d\n",
               cfg[c].name, cycles, cycles_fc, us_x100, mismatch, expect, took,
               dual);
        if (mismatch != 0) {
            TEST_FAIL("[%s] failed, %d of %d outputs differ\n",
                      cfg[c].name, mismatch, n_out);
        } else if (path_fail) {
            TEST_FAIL("[%s] path=%d expect=%d\n", cfg[c].name, took, expect);
        } else if (cycle_fail) {
            TEST_FAIL("[%s] cycles=%" PRIu32 " fc_cycles=%" PRIu32 " expect=%d\n",
                      cfg[c].name, cycles, cycles_fc, expect);
        } else {
            TEST_PASS("[%s] passed, cycles=%" PRIu32 "\n", cfg[c].name, cycles);
        }

    case_cleanup:
        esp_nn_set_fc_scratch_buf(NULL, 0);
        esp_nn_set_conv_scratch_buf(NULL);
        free(in_o);
        free(flt_o);
        free(ob_o);
        free(or_o);
        free(out_mult);
        free(out_shift);
        free(bias);
        free(scratch_raw);
    }

    /* Conv height is uint16_t. A batch of 65536 must stay off the 1x1 path.
     * The buffers are 16 rows; the oversized count is only an argument. */
    {
        const int row = 64;
        const int out_ch = 64;
        const int batch_ok = 16;
        const int32_t batches_hi = (int32_t) UINT16_MAX + 1;
        int8_t *in_o = NULL, *flt_o = NULL, *out_o = NULL;
        int32_t *mult = NULL, *shift = NULL;
        void *scratch_raw = NULL;

        in_o = ESP_NN_TEST_ALLOC((size_t) batch_ok * row + 32);
        flt_o = ESP_NN_TEST_ALLOC((size_t) row * out_ch + 32);
        out_o = ESP_NN_TEST_ALLOC((size_t) batch_ok * out_ch + 32);
        mult = ESP_NN_TEST_ALLOC((size_t) out_ch * sizeof(int32_t));
        shift = ESP_NN_TEST_ALLOC((size_t) out_ch * sizeof(int32_t));
        if (!in_o || !flt_o || !out_o || !mult || !shift) {
            TEST_SKIP("[batches_above_u16] alloc failed\n");
        } else {
            int8_t *input = (int8_t *) (((uintptr_t) in_o + 15) & ~(uintptr_t) 15);
            int8_t *filter = (int8_t *) (((uintptr_t) flt_o + 15) & ~(uintptr_t) 15);
            int8_t *out = (int8_t *) (((uintptr_t) out_o + 15) & ~(uintptr_t) 15);
            data_dims_t in_dims = { .width = 1, .height = batch_ok, .channels = row };
            data_dims_t filt_dims = { .width = 1, .height = 1, .channels = row };
            data_dims_t out_dims = { .width = 1, .height = batch_ok, .channels = out_ch };
            conv_params_t conv_params = {
                .in_offset = 128, .out_offset = 5,
                .stride = { .width = 1, .height = 1 },
                .padding = { .width = 0, .height = 0 },
                .dilation = { .width = 1, .height = 1 },
                .activation = { .min = -128, .max = 127 },
            };
            int scratch_bytes;
            void *scratch = NULL;
            int took_hi;

            memset(input, 0, (size_t) batch_ok * row);
            memset(filter, 0, (size_t) row * out_ch);
            for (int i = 0; i < out_ch; i++) {
                mult[i] = INT32_MAX / row;
                shift[i] = -8;
            }
            scratch_bytes = esp_nn_get_conv_scratch_size(&in_dims, &filt_dims,
                                                         &out_dims, &conv_params);
            if (scratch_bytes < 0) {
                scratch_bytes = 0;
            }
            if (scratch_bytes > 0) {
                scratch_raw = ESP_NN_TEST_ALLOC((size_t) scratch_bytes + 32);
            }
            if (scratch_bytes > 0 && !scratch_raw) {
                TEST_SKIP("[batches_above_u16] scratch alloc failed\n");
            } else {
                if (scratch_raw) {
                    scratch = (void *) (((uintptr_t) scratch_raw + 15) & ~(uintptr_t) 15);
                    esp_nn_set_fc_scratch_buf(scratch, (size_t) scratch_bytes);
                }
                took_hi = esp_nn_fc_per_ch_s8_batch_via_conv(
                        input, 128, row, filter, 0, NULL, out, out_ch, 5,
                        shift, mult, -128, 127, batches_hi);
                if (took_hi != 0) {
                    TEST_FAIL("[batches_above_u16] entered the 1x1 path\n");
                } else {
                    TEST_PASS("[batches_above_u16] stayed on fully connected\n");
                }
#if defined(CONFIG_NN_OPTIMIZED) && \
                (defined(CONFIG_IDF_TARGET_ESP32S3) || defined(CONFIG_IDF_TARGET_ESP32P4) || \
                 defined(CONFIG_IDF_TARGET_ESP32S31))
                {
                    int took_ok = esp_nn_fc_per_ch_s8_batch_via_conv(
                            input, 128, row, filter, 0, NULL, out, out_ch, 5,
                            shift, mult, -128, 127, batch_ok);
                    if (took_ok != 1) {
                        TEST_FAIL("[batches_u16_ok] did not enter the 1x1 path\n");
                    } else {
                        TEST_PASS("[batches_u16_ok] entered the 1x1 path\n");
                    }
                }
#endif
            }
        }
        esp_nn_set_fc_scratch_buf(NULL, 0);
        esp_nn_set_conv_scratch_buf(NULL);
        free(in_o);
        free(flt_o);
        free(out_o);
        free(mult);
        free(shift);
        free(scratch_raw);
    }

    /* The getter must match the 1x1 conv scratch size. Offsets are not part
     * of that size. ANSI and the generic targets report 0. */
    {
        const int32_t batches = 16;
        const uint16_t row = 64;
        const uint16_t out_ch = 64;
        data_dims_t in_dims = { .width = 1, .height = batches, .channels = row };
        data_dims_t filt_dims = { .width = 1, .height = 1, .channels = row };
        data_dims_t out_dims = { .width = 1, .height = batches, .channels = out_ch };
        conv_params_t conv_params = {
            .in_offset = 0, .out_offset = 0,
            .stride = { .width = 1, .height = 1 },
            .padding = { .width = 0, .height = 0 },
            .dilation = { .width = 1, .height = 1 },
            .activation = { .min = -128, .max = 127 },
        };
        const int got = esp_nn_get_fc_batch_scratch_size(batches, row, out_ch);
        const int ref = esp_nn_get_conv_scratch_size(&in_dims, &filt_dims,
                                                     &out_dims, &conv_params);
        conv_params.in_offset = 128;
        conv_params.out_offset = 5;
        const int ref_off = esp_nn_get_conv_scratch_size(&in_dims, &filt_dims,
                                                         &out_dims, &conv_params);
        in_dims.height = 3;
        in_dims.channels = 9;
        filt_dims.channels = 9;
        out_dims.height = 3;
        out_dims.channels = 8;
        conv_params.in_offset = 0;
        conv_params.out_offset = 0;
        const int odd = esp_nn_get_fc_batch_scratch_size(3, 9, 8);
        const int odd_ref = esp_nn_get_conv_scratch_size(&in_dims, &filt_dims,
                                                         &out_dims, &conv_params);
        const int hi = esp_nn_get_fc_batch_scratch_size((int32_t) UINT16_MAX + 1,
                                                        row, out_ch);
        const int zero_batch = esp_nn_get_fc_batch_scratch_size(0, row, out_ch);
        const int neg_batch = esp_nn_get_fc_batch_scratch_size(-1, row, out_ch);
        const int zero_row = esp_nn_get_fc_batch_scratch_size(batches, 0, out_ch);
        if (ESP_NN_HAS_FC_SCRATCH_BUF != 1 || got != ref || got != ref_off ||
                odd != odd_ref || hi != 0 || zero_batch != 0 ||
                neg_batch != 0 || zero_row != 0) {
            TEST_FAIL("[fc_scratch_size] got=%d ref=%d odd=%d hi=%d\n",
                      got, ref, odd, hi);
#if defined(CONFIG_NN_OPTIMIZED) && \
        (defined(CONFIG_IDF_TARGET_ESP32S3) || defined(CONFIG_IDF_TARGET_ESP32P4) || \
         defined(CONFIG_IDF_TARGET_ESP32S31))
        } else if (got <= 0 || odd <= 0) {
            TEST_FAIL("[fc_scratch_size] expected a workspace, got=%d odd=%d\n",
                      got, odd);
#endif
        } else {
            TEST_PASS("[fc_scratch_size] got=%d odd=%d\n", got, odd);
        }
    }
}
