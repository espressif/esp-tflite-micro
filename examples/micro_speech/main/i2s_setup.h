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

#ifndef _MICRO_SPEECH_I2S_SETUP_H_ 
#define _MICRO_SPEECH_I2S_SETUP_H_ 

#include <driver/gpio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_idf_version.h"


#pragma message ("ESP IDF " IDF_VER)

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5,0,0)
#define MICRO_SPEECH_USE_I2S_STD
#include <driver/i2s_std.h>
#pragma message ("using I2S_STD")
#else
#define MICRO_SPEECH_USE_I2S_LEGACY
#include "driver/i2s.h"
#include "esp_log.h"
#pragma message ("using I2S_LEGACY")
#endif

//Microphone Class
class Microphone {

public:
    Microphone(gpio_num_t sclk, gpio_num_t lrclk, gpio_num_t sdo);
    Microphone(void);
    ~Microphone(void);

    //function declarations
    void i2s_setup(void);
    void i2s_readsamples(void *dest, size_t *bytes_read);

private:
    //variables
    static constexpr int sample_buffer_size = 1600;  //3200bytes to read / 2bytes per sample (100ms of data - 1600 samples)
    static constexpr int sample_rate = 16000;        //16000 samples per second
    gpio_num_t sclk_pin;                             //bit clock line
    gpio_num_t lrclk_pin;                            //left right channel clock / WS
    gpio_num_t sdo_pin;                              //serial data line

#ifdef MICRO_SPEECH_USE_I2S_STD
    i2s_chan_handle_t rx_handle;        // I2S rx channel handler
#endif

#if CONFIG_IDF_TARGET_ESP32
    i2s_port_t i2s_port = I2S_NUM_1; // for esp32-eye
#else
    i2s_port_t i2s_port = I2S_NUM_0; // for esp32-s3-eye
#endif

};

#endif  //_MICRO_SPEECH_I2S_SETUP_H_ 

