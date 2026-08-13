/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: ESPRESSIF MIT
 */

#pragma once

#include "esp_err.h"
#include "esp_err.h"
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Ensure CAM_WIDTH and CAM_HEIGHT are set properly to resolution of sensor
#define CAM_HEIGHT 800
#define CAM_WIDTH  800

/**
 * @brief Frame buffer structure
 */
typedef struct {
    uint8_t *buf;               /*!< Pointer to the frame data */
    size_t len;                 /*!< Length of the buffer in bytes */
    size_t width;               /*!< Width of the image frame in pixels */
    size_t height;              /*!< Height of the image frame in pixels */
    struct timeval timestamp;   /*!< Timestamp since boot of the frame */
} video_fb_t;

/**
 * @brief Initialize the video interface
 *
 * @return esp_err_t ESP_OK on success, otherwise an error code
 */
esp_err_t esp_video_if_init(void);

/**
 * @brief Stop the video interface
 *
 * @return esp_err_t ESP_OK on success, otherwise an error code
 */
esp_err_t esp_video_if_stop(void);

/**
 * @brief Start the video interface
 *
 * @return esp_err_t ESP_OK on success, otherwise an error code
 */
esp_err_t esp_video_if_start(void);

/**
 * @brief Get a raw video frame
 *
 * @return video_fb_t* Pointer to the raw frame, or NULL if no frame is available
 */
video_fb_t *esp_video_if_get_frame(void);

/**
 * @brief Release a video frame when done with it
 *
 * @param fb Pointer to the frame to release
 */
void esp_video_if_release_frame(video_fb_t *fb);

#ifdef __cplusplus
}
#endif
