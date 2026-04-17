/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*/

/*
 * Modified by Shyam Jha
 *
 * Changes:
 * - Replaced legacy I2S driver with I2S standard driver (for ESP-IDF version > 5.0.0)
 * - Kept legacy I2S driver as it is for ESP-IDF version < 5.0 (Note: Project doesn't support version below 4.4)
 *
 * Tested with:
 * - ICS43434 MEMS microphone on idf v5.5.4
*/

#include "i2s_setup.h"

#define NO_I2S_SUPPORT CONFIG_IDF_TARGET_ESP32C2 || \
                          (CONFIG_IDF_TARGET_ESP32C3 \
                          && (ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(4, 4, 0)))

/**
 * @brief  Constructor for Microphone class
 * @param  sclk   GPIO pin for I2S serial clock (BCLK)
 * @param  lrclk  GPIO pin for I2S word select (LRCLK / WS)
 * @param  sdo    GPIO pin for I2S serial data input (SD)
 * @retval None
 * @note   Initializes GPIO pins mentioned by user.
 */
Microphone::Microphone(gpio_num_t sclk, gpio_num_t lrclk, gpio_num_t sdo)
{
#if !NO_I2S_SUPPORT

    sclk_pin = sclk;
    lrclk_pin = lrclk;
    sdo_pin = sdo;

#endif  //!NO_I2S_SUPPORT   
}

/**
 * @brief  Constructor for Microphone class
 * @param  None
 * @retval None
 * @note   Initializes GPIO pins with default values.
 *         esp32 mic pinouts - GPIO26:SCLK, GPIO32:LRCLK, GPIO33:SDOUT
 *         esp32s3 mic pinouts - GPIO41:SCLK, GPIO42:LRCLK, GPIO2:SDOUT
 *         Call above constructor in case default pins are not needed.
 */
Microphone::Microphone(void)
{
#if !NO_I2S_SUPPORT   

#if CONFIG_IDF_TARGET_ESP32S3
    sclk_pin = GPIO_NUM_41;
    lrclk_pin = GPIO_NUM_42;
    sdo_pin = GPIO_NUM_2;
#else
    sclk_pin = GPIO_NUM_26;
    lrclk_pin = GPIO_NUM_32;
    sdo_pin = GPIO_NUM_33;
#endif

#endif  //!NO_I2S_SUPPORT   
}

/**
 * @brief  Destructor for Microphone class
 * @param  None
 * @retval None
 */
Microphone::~Microphone(void)
{
#if !NO_I2S_SUPPORT   

#ifdef MICRO_SPEECH_USE_I2S_STD
    i2s_channel_disable(rx_handle);
    i2s_del_channel(rx_handle);
#else
    i2s_driver_uninstall(i2s_port);
#endif

#endif  //!NO_I2S_SUPPORT   
}

/**
 * @brief  init function for i2s peripheral
 * @param  None
 * @retval None.
 * @note   Supports both standard (ESP-IDF >= 5.x) and legacy I2S drivers.
 *         The appropriate driver configuration is selected at compile time.
*/
void Microphone::i2s_setup(void)
{
#if !NO_I2S_SUPPORT   

#ifdef MICRO_SPEECH_USE_I2S_STD
    /* Setting I2S configurations */
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(sample_rate),
        .slot_cfg = {
#if CONFIG_IDF_TARGET_ESP32S3
            .data_bit_width = I2S_DATA_BIT_WIDTH_32BIT,
            .slot_bit_width = I2S_SLOT_BIT_WIDTH_32BIT,
#else
            .data_bit_width = I2S_DATA_BIT_WIDTH_16BIT,
            .slot_bit_width = I2S_SLOT_BIT_WIDTH_16BIT,
#endif
            .slot_mode = I2S_SLOT_MODE_MONO,           // mono
            .slot_mask = I2S_STD_SLOT_LEFT,            // capture left channel
            .ws_width = I2S_SLOT_BIT_WIDTH_16BIT,
            .ws_pol = false,                           // WS = 0 → left channel
            .bit_shift = true,                         // Philips mode
#if CONFIG_IDF_TARGET_ESP32S3 | CONFIG_IDF_TARGET_ESP32P4
            .bit_order_lsb = false,
#else
            .msb_right = false,                        // MSB first
#endif
        },
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = sclk_pin,
            .ws = lrclk_pin,
            .dout = I2S_GPIO_UNUSED,
            .din = sdo_pin,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    i2s_chan_config_t chan_cfg = {
        .id = i2s_port,
        .role = I2S_ROLE_MASTER,
        .dma_desc_num = 8,
        .dma_frame_num = 240,
    };

    /* Allocate a new RX channel and get the handle of this channel */
    i2s_new_channel(&chan_cfg, NULL, &rx_handle);

    /* Initialize the channel */
    i2s_channel_init_std_mode(rx_handle, &std_cfg);

    /* Before reading data, start the RX channel first */
    i2s_channel_enable(rx_handle);

#else
    // Start listening for audio: MONO @ 16KHz
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = sample_rate,
#if CONFIG_IDF_TARGET_ESP32S3
        .bits_per_sample = (i2s_bits_per_sample_t) 32,
#else
        .bits_per_sample = (i2s_bits_per_sample_t) 16,  
#endif
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = 0,
        .dma_buf_count = 3,
        .dma_buf_len = 300,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = -1,
    };


    i2s_pin_config_t pin_config = {
        .bck_io_num = sclk_pin,   // IIS_SCLK
        .ws_io_num = lrclk_pin,   // IIS_LCLK
        .data_out_num = -1,       // IIS_DSIN
        .data_in_num = sdo_pin,   // IIS_DOUT
    };

    static const char *TAG = "I2S_LEGACY";
    esp_err_t ret = 0;
    ret = i2s_driver_install(i2s_port, &i2s_config, 0, NULL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Error in i2s_driver_install");
    }
    ret = i2s_set_pin(i2s_port, &pin_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Error in i2s_set_pin");
    }

    ret = i2s_zero_dma_buffer(i2s_port);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Error in initializing dma buffer with 0");
    }

#endif

#endif  //!NO_I2S_SUPPORT   
}

/**
 * @brief  i2s sample read function
 * @param  dest - pointer of receiving application buffer
 * @param  bytes_read - variable to store number of bytes read
 * @retval None.
 * @note   Supports both standard (ESP-IDF >= 5.x) and legacy I2S read methods.
 *         The appropriate function is selected at compile time.
*/
void Microphone::i2s_readsamples(void *dest, size_t *bytes_read)
{
#if !NO_I2S_SUPPORT   

#ifdef MICRO_SPEECH_USE_I2S_STD
    i2s_channel_read(rx_handle, dest, sizeof(int16_t) * (sample_buffer_size), bytes_read, portMAX_DELAY);
#else
    i2s_read(i2s_port, dest, sizeof(int16_t) * (sample_buffer_size), bytes_read, pdMS_TO_TICKS(100));
#endif

#endif //!NO_I2S_SUPPORT   
}

