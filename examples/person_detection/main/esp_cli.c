// Copyright 2020-2021 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <string.h>
#include <esp_log.h>
#include <esp_console.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "esp_main.h"
#include "esp_cli.h"
#include "esp_timer.h"

#define IMAGE_COUNT 10
static uint8_t *image_database[IMAGE_COUNT];


extern const uint8_t image0_start[]   asm("_binary_image0_start");
extern const uint8_t image1_start[]   asm("_binary_image1_start");
extern const uint8_t image2_start[]   asm("_binary_image2_start");
extern const uint8_t image3_start[]   asm("_binary_image3_start");
extern const uint8_t image4_start[]   asm("_binary_image4_start");
extern const uint8_t image5_start[]   asm("_binary_image5_start");
extern const uint8_t image6_start[]   asm("_binary_image6_start");
extern const uint8_t image7_start[]   asm("_binary_image7_start");
extern const uint8_t image8_start[]   asm("_binary_image8_start");
extern const uint8_t image9_start[]   asm("_binary_image9_start");

static const char *TAG = "[esp_cli]";

static int task_dump_cli_handler(int argc, char *argv[])
{
    int num_of_tasks = uxTaskGetNumberOfTasks();
    TaskStatus_t *task_array = calloc(1, num_of_tasks * sizeof(TaskStatus_t));
    /* Just to go to the next line */
    printf("\n");
    if (!task_array) {
        ESP_LOGE(TAG, "Memory not allocated for task list.");
        return 0;
    }
    num_of_tasks = uxTaskGetSystemState(task_array, num_of_tasks, NULL);
    printf("\tName\tNumber\tPriority\tStackWaterMark\n");
    for (int i = 0; i < num_of_tasks; i++) {
        printf("%16s\t%u\t%u\t%u\n",
               task_array[i].pcTaskName,
               (unsigned) task_array[i].xTaskNumber,
               (unsigned) task_array[i].uxCurrentPriority,
               (unsigned) task_array[i].usStackHighWaterMark);
    }
    free(task_array);
    return 0;
}

static int cpu_dump_cli_handler(int argc, char *argv[])
{
    /* Just to go to the next line */
    printf("\n");
#ifndef CONFIG_FREERTOS_GENERATE_RUN_TIME_STATS
    printf("%s: To use this utility enable: Component config --> FreeRTOS --> Enable FreeRTOS to collect run time stats\n", TAG);
#else
    char *buf = calloc(1, 2 * 1024);
    vTaskGetRunTimeStats(buf);
    printf("%s: Run Time Stats:\n%s\n", TAG, buf);
    free(buf);
#endif
    return 0;
}

static int mem_dump_cli_handler(int argc, char *argv[])
{
    /* Just to go to the next line */
    printf("\n");
    printf("\tDescription\tInternal\tSPIRAM\n");
    printf("Current Free Memory\t%d\t\t%d\n",
           heap_caps_get_free_size(MALLOC_CAP_8BIT) - heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    printf("Largest Free Block\t%d\t\t%d\n",
           heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL),
           heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
    printf("Min. Ever Free Size\t%d\t\t%d\n",
           heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL),
           heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
    return 0;
}

static int inference_cli_handler(int argc, char *argv[])
{
    /* Just to go to the next line */
    printf("\n");
    if (argc != 2) {
        printf("%s: Incorrect arguments\n", TAG);
        return 0;
    }
    int image_number = atoi(argv[1]);

    if((image_number < 0) || (image_number >= IMAGE_COUNT)) {
        ESP_LOGE(TAG, "Please Enter a valid Number ( 0 - %d)", IMAGE_COUNT-1);
        return -1;
    }
   // char file_name[30];
   // sprintf(file_name, "image%d.raw", image_number);
    unsigned detect_time;
    detect_time = esp_timer_get_time();
    run_inference((void *)image_database[image_number]);
    detect_time = (esp_timer_get_time() - detect_time)/1000;
    ESP_LOGI(TAG,"Time required for the inference is %u ms", detect_time);

    return 0;
}

static esp_console_cmd_t diag_cmds[] = {
    {
        .command = "mem-dump",
        .help = "",
        .func = mem_dump_cli_handler,
    },
    {
        .command = "task-dump",
        .help = "",
        .func = task_dump_cli_handler,
    },
    {
        .command = "cpu-dump",
        .help = "",
        .func = cpu_dump_cli_handler,
    },
    {
        .command = "detect_image",
        .help = "detect_image <image_number>"
                "Note: image numbers ranging from 0 - 9 only are valid",
        .func = inference_cli_handler,
    },
};

int esp_cli_register_cmds()
{
    int cmds_num = sizeof(diag_cmds) / sizeof(esp_console_cmd_t);
    int i;
    for (i = 0; i < cmds_num; i++) {
        ESP_LOGI(TAG, "Registering command: %s", diag_cmds[i].command);
        esp_console_cmd_register(&diag_cmds[i]);
    }
    return 0;
}

static void image_database_init()
{
    image_database[0] = (uint8_t *) image0_start;
    image_database[1] = (uint8_t *) image1_start;
    image_database[2] = (uint8_t *) image2_start;
    image_database[3] = (uint8_t *) image3_start;
    image_database[4] = (uint8_t *) image4_start;
    image_database[5] = (uint8_t *) image5_start;
    image_database[6] = (uint8_t *) image6_start;
    image_database[7] = (uint8_t *) image7_start;
    image_database[8] = (uint8_t *) image8_start;
    image_database[9] = (uint8_t *) image9_start;

}

int esp_cli_start()
{
    image_database_init();
    static int cli_started;
    if (cli_started) {
        return 0;
    }

    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();

    esp_console_register_help_command();
    esp_cli_register_cmds();
#if defined(CONFIG_ESP_CONSOLE_UART_DEFAULT) || defined(CONFIG_ESP_CONSOLE_UART_CUSTOM)
    esp_console_dev_uart_config_t hw_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&hw_config, &repl_config, &repl));

#elif defined(CONFIG_ESP_CONSOLE_USB_CDC)
    esp_console_dev_usb_cdc_config_t hw_config = ESP_CONSOLE_DEV_CDC_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_usb_cdc(&hw_config, &repl_config, &repl));

#elif defined(CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG)
    esp_console_dev_usb_serial_jtag_config_t hw_config = ESP_CONSOLE_DEV_USB_SERIAL_JTAG_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_usb_serial_jtag(&hw_config, &repl_config, &repl));

#else
#error Unsupported console type
#endif
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
    cli_started = 1;
    return 0;
}
