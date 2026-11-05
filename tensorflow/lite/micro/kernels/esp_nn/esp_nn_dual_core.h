/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Claim of the ESP-NN dual-core worker for the wrappers that split an op
// across both cores themselves.
//
// From ESP_NN_HAS_DUAL_CORE_CLAIM, esp_nn_dual_core_run() needs the caller
// to hold the claim. Older ESP-NN has no claim, and only checks that the
// worker runs. Pair every true EspNnDualCoreBegin() with
// EspNnDualCoreEnd(). On false, run the op on this core.

#pragma once

#if ESP_NN
#if defined(__has_include)
#if __has_include(<esp_nn_multicore.h>)
#include <esp_nn_multicore.h>
#define ESP_NN_HAVE_MULTICORE 1
#endif
#endif
#endif

#ifdef ESP_NN_HAVE_MULTICORE
static inline bool EspNnDualCoreBegin() {
#if ESP_NN_HAS_DUAL_CORE_CLAIM
  return esp_nn_dual_core_begin();
#else
  return esp_nn_dual_core_active();
#endif
}

static inline void EspNnDualCoreEnd() {
#if ESP_NN_HAS_DUAL_CORE_CLAIM
  esp_nn_dual_core_end();
#endif
}
#endif  // ESP_NN_HAVE_MULTICORE
