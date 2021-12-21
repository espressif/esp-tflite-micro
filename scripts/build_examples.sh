#!/usr/bin/env bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# Tests the microcontroller code for esp32 platform

set -e

TARGET=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script Dir is set to ${SCRIPT_DIR}"
ROOT_DIR=${SCRIPT_DIR}/../
echo "Root Dir is set to ${ROOT_DIR}"

# build examples
echo "------------------------ Started: Building hello_world example ------------------------"
cd "${ROOT_DIR}"/examples/hello_world
# set target
idf.py set-target $TARGET
# build app
idf.py build
echo "------------------------ Done: Building hello_world example ------------------------"

echo "------------------------ Started: Building person_detection example ------------------------"
cd "${ROOT_DIR}"/examples/person_detection
# set target
idf.py set-target $TARGET
# build app
idf.py build
echo "------------------------ Done: Building person_detection example ------------------------"

echo "------------------------ Started: Building micro_speech example ------------------------"
cd "${ROOT_DIR}"/examples/micro_speech
# set target
idf.py set-target $TARGET
# build app
idf.py build
echo "------------------------ Done: Building micro_speech example ------------------------"

