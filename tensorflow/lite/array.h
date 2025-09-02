/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_ARRAY_H_
#define TENSORFLOW_LITE_ARRAY_H_

#include <cstring>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <vector>

#include "tensorflow/lite/core/c/common.h"

namespace tflite {

/// TfLite*Array helpers

namespace array_internal {

// Function object used as a deleter for unique_ptr holding TFLite*Array
// objects.
struct TfLiteArrayDeleter {
  void operator()(TfLiteIntArray* a);
  void operator()(TfLiteFloatArray* a);
};

// Maps T to the corresponding TfLiteArray type.
template <class T>
struct TfLiteArrayInfo;

template <>
struct TfLiteArrayInfo<int> {
  using Type = TfLiteIntArray;
};

template <>
struct TfLiteArrayInfo<float> {
  using Type = TfLiteFloatArray;
};

}  // namespace array_internal

template <class T>
using TfLiteArrayUniquePtr =
    std::unique_ptr<typename array_internal::TfLiteArrayInfo<T>::Type,
                    array_internal::TfLiteArrayDeleter>;

// `unique_ptr` wrapper for `TfLiteIntArray`s.
using IntArrayUniquePtr = TfLiteArrayUniquePtr<int>;

// `unique_ptr` wrapper for `TfLiteFloatArray`s.
using FloatArrayUniquePtr = TfLiteArrayUniquePtr<float>;

// Allocates a TfLiteArray of given size using malloc.
//
// This builds an int array by default as this is the overwhelming part of the
// use cases.
template <class T = int>
TfLiteArrayUniquePtr<T> BuildTfLiteArray(int size);

// Allocates a TfLiteIntArray of given size using malloc.
template <>
inline IntArrayUniquePtr BuildTfLiteArray<int>(const int size) {
  return IntArrayUniquePtr(TfLiteIntArrayCreate(size));
}

// Allocates a TfLiteFloatArray of given size using malloc.
template <>
inline FloatArrayUniquePtr BuildTfLiteArray<float>(const int size) {
  return FloatArrayUniquePtr(TfLiteFloatArrayCreate(size));
}

// Allocates a TFLiteArray of given size and initializes it.
//
// `values` is expected to holds `size` elements.
template <class T>
TfLiteArrayUniquePtr<T> BuildTfLiteArray(const int size,
                                         const T* const values) {
  auto array = BuildTfLiteArray<T>(size);
  if (array) {
    memcpy(array->data, values, size * sizeof(T));
  }
  return array;
}

// Allocates a TFLiteArray and initializes it with the given values.
template <class T>
TfLiteArrayUniquePtr<T> BuildTfLiteArray(const std::vector<T>& values) {
  return BuildTfLiteArray(static_cast<int>(values.size()), values.data());
}

// Allocates a TFLiteArray and initializes it with the given values.
template <class T>
TfLiteArrayUniquePtr<T> BuildTfLiteArray(
    const std::initializer_list<T>& values) {
  return BuildTfLiteArray(static_cast<int>(values.size()), values.begin());
}

// Allocates a TFLiteArray and initializes it with the given array.
inline IntArrayUniquePtr BuildTfLiteArray(const TfLiteIntArray& other) {
  return BuildTfLiteArray(other.size, other.data);
}

// Allocates a TFLiteArray and initializes it with the given array.
inline FloatArrayUniquePtr BuildTfLiteArray(const TfLiteFloatArray& other) {
  return BuildTfLiteArray(other.size, other.data);
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_ARRAY_H_
