
#include <freertos/FreeRTOS.h>
#include <esp_log.h>
#include <esp_timer.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"
#include "image.h"

static const char* TAG = "mobilenet_custom";

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

void print_output() {
    printf("{ ");
    for (int i = 0; i < output->dims->data[1]; i++) {
        printf("%f, ", dequantize(output->data.int8[i]));
    }
    printf("}\n");
}

extern "C" void app_main(void)
{
    model = tflite::GetModel(mobilenet_custom_model);
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

    static tflite::MicroMutableOpResolver<8> micro_op_resolver;
    micro_op_resolver.AddRelu6();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddPad();

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

    for (int i = 0; i < 224 * 224 * 3; i++) {
        input->data.int8[i] = quantize((image1[i] / 127.5) - 1);
    }

    long long start_time = esp_timer_get_time();
    if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke() failed");
    }

    printf("Invoke Time: %lld\n", (esp_timer_get_time() - start_time) / 1000);
    
    print_output();

    for (int i = 0; i < 224 * 224 * 3; i++) {
        input->data.int8[i] = quantize((image2[i] / 127.5) - 1);
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke() failed");
    }
    
    print_output();
}
