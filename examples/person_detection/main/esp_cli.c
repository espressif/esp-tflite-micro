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
#include <freertos/queue.h>
#include <driver/uart.h>

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

static int stop;
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
        printf("%16s\t%d\t%d\t%d\n",
               task_array[i].pcTaskName,
               task_array[i].xTaskNumber,
               task_array[i].uxCurrentPriority,
               task_array[i].usStackHighWaterMark);
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
    uint32_t detect_time;
    detect_time = esp_timer_get_time();
    run_inference((void *)image_database[image_number]);
    detect_time = (esp_timer_get_time() - detect_time)/1000;
    ESP_LOGI(TAG,"Time required for the inference is %d ms", detect_time);

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

static void esp_cli_task(void *arg)
{
    int uart_num = (int) arg;
    uint8_t linebuf[256];
    int i, cmd_ret;
    esp_err_t ret;
    QueueHandle_t uart_queue;
    uart_event_t event;

    ESP_LOGI(TAG, "Initialising UART on port %d", uart_num);
    uart_driver_install(uart_num, 256, 0, 8, &uart_queue, 0);
    /* Initialize the console */
    esp_console_config_t console_config = {
            .max_cmdline_args = 8,
            .max_cmdline_length = 256,
    };

    esp_console_init(&console_config);
    esp_console_register_help_command();

    while (!stop) {
        uart_write_bytes(uart_num, "\n>> ", 4);
        memset(linebuf, 0, sizeof(linebuf));
        i = 0;
        do {
            ret = xQueueReceive(uart_queue, (void * )&event, (portTickType)portMAX_DELAY);
            if (ret != pdPASS) {
                if(stop == 1) {
                    break;
                } else {
                    continue;
                }
            }
            if (event.type == UART_DATA) {
                while (uart_read_bytes(uart_num, (uint8_t *) &linebuf[i], 1, 0)) {
                    if (linebuf[i] == '\r') {
                        uart_write_bytes(uart_num, "\r\n", 2);
                    } else {
                        uart_write_bytes(uart_num, (char *) &linebuf[i], 1);
                    }
                    i++;
                }
            }
        } while ((i < 255) && linebuf[i-1] != '\r');
        if (stop) {
            break;
        }
        /* Remove the truncating \r\n */
        linebuf[strlen((char *)linebuf) - 1] = '\0';
        ret = esp_console_run((char *) linebuf, &cmd_ret);
        if (ret < 0) {
            printf("%s: Console dispatcher error\n", TAG);
            break;
        }
    }
    ESP_LOGE(TAG, "Stopped CLI");
    vTaskDelete(NULL);
}

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
    image_database[7] = (uint8_t *) image8_start;
    image_database[8] = (uint8_t *) image8_start;
    image_database[9] = (uint8_t *) image9_start;

}

int esp_cli_init()
{
    image_database_init();
    static int cli_started;
    if (cli_started) {
        return 0;
    }
#define ESP_CLI_STACK_SIZE (4 * 1024)
    //StackType_t *task_stack = (StackType_t *) calloc(1, ESP_CLI_STACK_SIZE);
    //static StaticTask_t task_buf;
    if(pdPASS != xTaskCreate(&esp_cli_task, "cli_task", ESP_CLI_STACK_SIZE, NULL, 4,NULL)) {
        ESP_LOGE(TAG, "Couldn't create task");
        return -1;
    }
    cli_started = 1;
    return 0;
}
