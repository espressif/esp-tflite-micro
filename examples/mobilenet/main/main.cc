
#include <freertos/FreeRTOS.h>
#include <esp_log.h>
#include <esp_timer.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_profiler.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"
#include "image.h"

static const char* TAG = "mobilenet";

namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;

    constexpr int kTensorArenaSize = 1.5 * 1024 * 1024;
    static uint8_t* tensor_arena;

    TfLiteTensor* input;
    TfLiteTensor* output;
}

int8_t quantize(float val) {
    auto zero_point = input->params.zero_point;
    auto scale = input->params.scale;
    return (val / scale) + zero_point;
}

float dequantize(int8_t val) {
    auto zero_point = output->params.zero_point;
    auto scale = output->params.scale;
    return (val - zero_point) * scale;
}

extern "C" void app_main(void)
{
    model = tflite::GetModel(esp_mobile_net_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model provided is schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    }

    if (tensor_arena == NULL) {
        tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }

    if (tensor_arena == NULL) {
        printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
        return;
    }

    ESP_LOGI(TAG, "Allocated memory for Tensor Arena");

    static tflite::MicroMutableOpResolver<7> micro_op_resolver;
    micro_op_resolver.AddRelu6();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();

    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    ESP_LOGI(TAG, "Allocated Tensors");

    input = interpreter->input(0);
    output = interpreter->output(0);

    for (int i = 0; i < image_raw_len; i++) {
        input->data.int8[i] = quantize(((uint8_t)image_raw[i] / 127.5) - 1);
    }

    long long start_time = esp_timer_get_time();

    if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke() failed");
    }

    long long total_time = esp_timer_get_time() - start_time;
    ESP_LOGI(TAG, "Invoke was successful");
    printf("Invoke: Total time = %lld\n", total_time / 1000);

    int maxLabel = 0;
    float maxConf = 0.0;

    for (int i = 0; i < 1000; i++) {
        float conf = dequantize(output->data.int8[i]);
        if (conf > maxConf) {
            maxLabel = i;
            maxConf = conf;
        }
    }

    printf("\nLabel: %d, Confidence: %f\n", maxLabel, maxConf);
}
