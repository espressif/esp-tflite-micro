/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Private to the fully connected sources. Not part of esp_nn.h. */
#pragma once

#include <stddef.h>
#include <stdint.h>

/* Install buf as the convolution scratch pointer for one call, and remember
 * the pointer that was already there. The preferred buffer, where the
 * target has one, is suspended so the 1x1 kernel writes buf. */
#if defined(CONFIG_NN_OPTIMIZED) && \
        (defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S31))
void esp_nn_conv_scratch_borrow_riscv_pie(void *buf, void **scratch,
                                           void **preferred, size_t *preferred_size);
void esp_nn_conv_scratch_restore_riscv_pie(void *scratch, void *preferred,
                                            size_t preferred_size);
#define esp_nn_conv_scratch_borrow esp_nn_conv_scratch_borrow_riscv_pie
#define esp_nn_conv_scratch_restore esp_nn_conv_scratch_restore_riscv_pie
#elif defined(CONFIG_NN_OPTIMIZED) && defined(CONFIG_IDF_TARGET_ESP32S3)
void esp_nn_conv_scratch_borrow_esp32s3(void *buf, void **scratch,
                                        void **preferred, size_t *preferred_size);
void esp_nn_conv_scratch_restore_esp32s3(void *scratch, void *preferred,
                                         size_t preferred_size);
#define esp_nn_conv_scratch_borrow esp_nn_conv_scratch_borrow_esp32s3
#define esp_nn_conv_scratch_restore esp_nn_conv_scratch_restore_esp32s3
#else
void esp_nn_conv_scratch_borrow_ansi(void *buf, void **scratch,
                                     void **preferred, size_t *preferred_size);
void esp_nn_conv_scratch_restore_ansi(void *scratch, void *preferred,
                                      size_t preferred_size);
#define esp_nn_conv_scratch_borrow esp_nn_conv_scratch_borrow_ansi
#define esp_nn_conv_scratch_restore esp_nn_conv_scratch_restore_ansi
#endif

/* Drop the fully connected scratch registration. The batched call does this
 * before it returns. */
void esp_nn_fc_scratch_clear(void);

/* Per-pixel accumulator term for a nonzero fully connected weight
 * zero-point. P4 only. NULL keeps the weight-zero-point-0 kernel. */
#if defined(CONFIG_NN_OPTIMIZED) && \
        (defined(CONFIG_IDF_TARGET_ESP32P4) || defined(CONFIG_IDF_TARGET_ESP32S31))
void esp_nn_conv_set_row_addend_riscv_pie(const int32_t *row_addend);
#endif

/* Returns 1 when the batch was evaluated as a 1x1 convolution.
 * On P4 a nonzero weight zero-point is applied as a per-row addend and
 * still uses the 1x1 kernel. A shape below the per-chip performance floor,
 * batches above UINT16_MAX, an unaligned input or filter, or a fully
 * connected scratch buffer that is missing or smaller than this 1x1 shape
 * returns 0 and leaves out_data untouched. The convolution scratch pointer
 * is not consulted. */
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
                                    const int32_t batches);
