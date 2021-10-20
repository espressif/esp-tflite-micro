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
TFLITE_LIB_DIR="${ROOT_DIR}/components/tflite-lib/"
cd "${TFLITE_LIB_DIR}"

TEMP_DIR=$(mktemp -d)
cd "${TEMP_DIR}"

echo Cloning tflite-micro repo to "${TEMP_DIR}"
git clone --depth 1 --single-branch "https://github.com/tensorflow/tflite-micro.git"
cd tflite-micro

TARGET=esp
OPTIMIZED_KERNEL_DIR=esp_nn
TARGET_ARCH=project_generation


# Create the TFLM base tree
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  -e hello_world -e magic_wand -e micro_speech -e person_detection \
  --makefile_options="TARGET=${TARGET} TARGET_ARCH=${TARGET_ARCH}" \
  "${TEMP_DIR}/tflm-out"

cd "${TFLITE_LIB_DIR}"
rm -rf tensorflow
mv "${TEMP_DIR}/tflm-out/tensorflow" tensorflow
rm -rf tensorflow/lite/micro/cortex_m_generic
rm -rf tensorflow/lite/micro/system_setup.cc

# For this repo we are forking both the models and the examples.
rm -rf tensorflow/lite/micro/models
mkdir -p third_party/
/bin/cp -r "${TEMP_DIR}"/tflm-out/third_party/* third_party/

rm -rf "${TEMP_DIR}"
