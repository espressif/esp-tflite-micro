/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

#ifdef ESP_PLATFORM
#include "sdkconfig.h"
#endif

/* The resident filter panel is sized against the data cache the image is
 * actually built with (64 KB, 8-way, 64-byte lines by default on ESP32-S3):
 * at most half of it, so the staged windows, the transpose scratch and the
 * output staging that stream past the panel still have ways to live in. */
#if defined(CONFIG_ESP32S3_DATA_CACHE_SIZE)
#define ESP_NN_S3_DCACHE_BYTES      CONFIG_ESP32S3_DATA_CACHE_SIZE
#else
#define ESP_NN_S3_DCACHE_BYTES      (64 * 1024)
#endif

#define ESP_NN_S3_PANEL_BYTES       (ESP_NN_S3_DCACHE_BYTES / 2)

/*
 * OC-panel driver for mult8 1x1 convolutions: a panel of filter rows stays
 * L1-resident while the batched assembly sweeps the spatial positions.
 * Also used by the 3x3 path, whose staged im2col windows are exactly a 1x1
 * problem with in_channels = 9 * in_channels.
 *
 * `scratch` needs esp_nn_conv_1x1_panel_scratch_size(in_channels) bytes.
 */
void esp_nn_conv_s8_mult8_1x1_oc_panel(
        const int8_t *input, int spatial_size, uint16_t in_channels,
        int32_t input_offset, const int8_t *filter_data, const int32_t *bias,
        int8_t *out_data, uint16_t out_channels, uint16_t out_stride,
        int32_t out_offset, const int32_t *out_shift, const int32_t *out_mult,
        int32_t activation_min, int32_t activation_max, void *scratch,
        const int32_t *row_addend);

/* Panel work area: transpose + tails (24 * in_channels), the staging buffer
 * the driver scatters from (16 KB), and alignment slack. */
static inline int esp_nn_conv_1x1_panel_scratch_size(int in_channels)
{
    return 24 * in_channels + 16 * 1024 + 64;
}
