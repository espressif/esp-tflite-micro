#!/usr/bin/env bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Creates the project file distributions for the TensorFlow Lite Micro test and
# example targets aimed at embedded platforms.

set -e -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
TFLITE_LIB_DIR="${ROOT_DIR}/"
cd "${TFLITE_LIB_DIR}"

TEMP_DIR=$(mktemp -d)
cd "${TEMP_DIR}"

echo Cloning tflite-micro repo to "${TEMP_DIR}"
git clone --depth 1 --single-branch "https://github.com/tensorflow/tflite-micro.git"
cd tflite-micro


# Create the TFLM base tree
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  -e hello_world -e micro_speech -e person_detection "${TEMP_DIR}/tflm-out"

# Backup `micro/kernels/esp_nn` directory to new tree
/bin/cp -r "${TFLITE_LIB_DIR}"/tensorflow/lite/micro/kernels/esp_nn \
  "${TEMP_DIR}"/tflm-out/tensorflow/lite/micro/kernels/

# Backup `micro/esp` directory to new tree
/bin/cp -r "${TFLITE_LIB_DIR}"/tensorflow/lite/micro/esp \
  "${TEMP_DIR}"/tflm-out/tensorflow/lite/micro/

cd "${TFLITE_LIB_DIR}"
rm -rf tensorflow
rm -rf third_party
rm -rf signal
mv "${TEMP_DIR}/tflm-out/tensorflow" tensorflow

# For this repo we are forking both the models and the examples.
rm -rf tensorflow/lite/micro/models
mkdir -p third_party/
/bin/cp -r "${TEMP_DIR}"/tflm-out/third_party/* third_party/
mkdir -p signal/
/bin/cp -r "${TEMP_DIR}"/tflm-out/signal/* signal/

rm -rf "${TEMP_DIR}"
