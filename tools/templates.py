"""

SPDX-FileCopyrightText: 2023 Espressif Systems (Shanghai) CO LTD

SPDX-License-Identifier: Apache-2.0

"""

from string import Template

# main_functions.cc
cppTemplate = Template(
    """ 
/* 

SPDX-FileCopyrightText: ${current_year} Espressif Systems (Shanghai) CO LTD
SPDX-License-Identifier: Apache-2.0

Copyright ${current_year} The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "gen_micro_mutable_op_resolver.h"
#include "${model_name_header}_model_data.h"
#include "constants.h"
#include "output_handler.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// update the value for kTensorArenaSize as per your problem's use case
constexpr int kTensorArenaSize = 50000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  
  // X: variable =  g_hello_world_model_data
  model = tflite::GetModel($model_name);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  // X: variable = Pull all the operations from op_resolver.h file
  static tflite::MicroMutableOpResolver<$num_of_operations> $resolver;
  $resolver = get_resolver();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, $resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  float position = static_cast<float>(inference_count) /
                   static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  // Quantize the input from floating-point to integer
  int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  // Place the quantized input in the model's input tensor
  input->data.int8[0] = x_quantized;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed on x: ", static_cast<double>(x));
    return;
  }

  // Obtain the quantized output from model's output tensor
  int8_t y_quantized = output->data.int8[0];
  // Dequantize the output from integer to floating-point
  float y = (y_quantized - output->params.zero_point) * output->params.scale;

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(x, y);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}

"""
)

# main_functions.h
main_functions = Template(
    """
/* 
SPDX-FileCopyrightText: ${current_year} Espressif Systems (Shanghai) CO LTD
SPDX-License-Identifier: Apache-2.0


Copyright ${current_year} The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

// Initializes all data needed for the example. The name is important, and needs
// to be setup() for Arduino compatibility.
void setup();

// Runs one iteration of data gathering and inference. This should be called
// repeatedly from the application code. The name needs to be loop() for Arduino
// compatibility.
void loop();

#ifdef __cplusplus
}
#endif
"""
)

# main.cc
main_cc = Template(
    """
/*

SPDX-FileCopyrightText: ${current_year} Espressif Systems (Shanghai) CO LTD
SPDX-License-Identifier: Apache-2.0

*/

#include "main_functions.h"

extern "C" void app_main(void) {
  setup();
  while (true) {
    loop();
  }
}
"""
)

# output_handler.cc
output_handler_cc = Template(
    """
/*

SPDX-FileCopyrightText: ${current_year} Espressif Systems (Shanghai) CO LTD
SPDX-License-Identifier: Apache-2.0


Copyright ${current_year} The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "output_handler.h"
#include "tensorflow/lite/micro/micro_log.h"

void HandleOutput(float x_value, float y_value) {
  // Log the current X and Y values
  MicroPrintf("x_value: ", static_cast<double>(x_value));
  MicroPrintf("y_value: ", static_cast<double>(y_value));
}
"""
)

# output_handler.h
output_handler_h = Template(
"""
/*

SPDX-FileCopyrightText: ${current_year} Espressif Systems (Shanghai) CO LTD
SPDX-License-Identifier: Apache-2.0


Copyright ${current_year} The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "tensorflow/lite/c/common.h"

// Called by the main loop to produce some output based on the x and y values
void HandleOutput(float x_value, float y_value);
"""
)

# constants.h
constants_h = Template(
    """
/* 

SPDX-FileCopyrightText: ${current_year} Espressif Systems (Shanghai) CO LTD
SPDX-License-Identifier: Apache-2.0

Copyright ${current_year} The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

// This constant represents the range of x values our model was trained on,
// which is from 0 to (2 * Pi). We approximate Pi to avoid requiring additional
// libraries.
const float kXrange = 2.f * 3.14159265359f;

// This constant determines the number of inferences to perform across the range
// of x values defined above. Since each inference takes time, the higher this
// number, the more time it will take to run through the entire range. The value
// of this constant can be tuned so that one full cycle takes a desired amount
// of time. Since different devices take different amounts of time to perform
// inference, this value should be defined per-device.
extern const int kInferencesPerCycle;

"""
)

# constants.cc
constants_cc = Template(
    """
/* 

SPDX-FileCopyrightText: ${current_year} Espressif Systems (Shanghai) CO LTD
SPDX-License-Identifier: Apache-2.0


Copyright ${current_year} The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "constants.h"

// This is a small number so that it's easy to read the logs
const int kInferencesPerCycle = 20;
"""
)

# CMakeLists.txt
CMakeLists_txt = Template(
    """
idf_component_register(SRCS main_functions.cc main.cc ${model_name_header}_model_data.cc output_handler.cc constants.cc
                       INCLUDE_DIRS "")
"""
)

topLevelCMake = Template(
    r"""
# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.5)
set(EXTRA_COMPONENT_DIRS ../../components)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(test)
"""
)
