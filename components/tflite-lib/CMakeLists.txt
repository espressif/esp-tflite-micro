
set(tflite_dir "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/lite")
set(tfmicro_dir "${tflite_dir}/micro")
set(tfmicro_frontend_dir "${tflite_dir}/experimental/microfrontend/lib")
set(tfmicro_kernels_dir "${tfmicro_dir}/kernels")

set(srcs_micro
          "${tfmicro_dir}/micro_graph.cc"
          "${tfmicro_dir}/flatbuffer_utils.cc"
          "${tfmicro_dir}/micro_interpreter.cc"
          "${tfmicro_dir}/micro_utils.cc"
          "${tfmicro_dir}/all_ops_resolver.cc"
          "${tfmicro_dir}/micro_allocator.cc"
          "${tfmicro_dir}/debug_log.cc"
          "${tfmicro_dir}/recording_simple_memory_allocator.cc"
          "${tfmicro_dir}/micro_profiler.cc"
          "${tfmicro_dir}/micro_string.cc"
          "${tfmicro_dir}/micro_time.cc"
          "${tfmicro_dir}/recording_micro_allocator.cc"
          "${tfmicro_dir}/test_helpers.cc"
          "${tfmicro_dir}/memory_helpers.cc"
          "${tfmicro_dir}/micro_error_reporter.cc"
          "${tfmicro_dir}/simple_memory_allocator.cc")

file(GLOB src_micro_frontend
          "${tfmicro_frontend_dir}/*.c"
          "${tfmicro_frontend_dir}/*.cc")

set(srcs_kernels
          "${tfmicro_kernels_dir}/activations.cc"
          "${tfmicro_kernels_dir}/add.cc"
          "${tfmicro_kernels_dir}/arg_min_max.cc"
          "${tfmicro_kernels_dir}/batch_to_space_nd.cc"
          "${tfmicro_kernels_dir}/cast.cc"
          "${tfmicro_kernels_dir}/ceil.cc"
          "${tfmicro_kernels_dir}/circular_buffer.cc"
          "${tfmicro_kernels_dir}/comparisons.cc"
          "${tfmicro_kernels_dir}/concatenation.cc"
          "${tfmicro_kernels_dir}/conv.cc"
          "${tfmicro_kernels_dir}/conv_common.cc"
          "${tfmicro_kernels_dir}/depthwise_conv.cc"
          "${tfmicro_kernels_dir}/depthwise_conv_common.cc"
          "${tfmicro_kernels_dir}/dequantize.cc"
          "${tfmicro_kernels_dir}/detection_postprocess.cc"
          "${tfmicro_kernels_dir}/elementwise.cc"
          "${tfmicro_kernels_dir}/elu.cc"
          "${tfmicro_kernels_dir}/ethosu.cc"
          "${tfmicro_kernels_dir}/exp.cc"
          "${tfmicro_kernels_dir}/fill.cc"
          "${tfmicro_kernels_dir}/floor.cc"
          "${tfmicro_kernels_dir}/fully_connected.cc"
          "${tfmicro_kernels_dir}/fully_connected_common.cc"
          "${tfmicro_kernels_dir}/hard_swish.cc"
          "${tfmicro_kernels_dir}/kernel_runner.cc"
          "${tfmicro_kernels_dir}/kernel_util.cc"
          "${tfmicro_kernels_dir}/l2norm.cc"
          "${tfmicro_kernels_dir}/leaky_relu.cc"
          "${tfmicro_kernels_dir}/logical.cc"
          "${tfmicro_kernels_dir}/logistic.cc"
          "${tfmicro_kernels_dir}/maximum_minimum.cc"
          "${tfmicro_kernels_dir}/mul.cc"
          "${tfmicro_kernels_dir}/neg.cc"
          "${tfmicro_kernels_dir}/pack.cc"
          "${tfmicro_kernels_dir}/pad.cc"
          "${tfmicro_kernels_dir}/pooling.cc"
          "${tfmicro_kernels_dir}/pooling_common.cc"
          "${tfmicro_kernels_dir}/prelu.cc"
          "${tfmicro_kernels_dir}/quantize.cc"
          "${tfmicro_kernels_dir}/quantize_common.cc"
          "${tfmicro_kernels_dir}/reduce.cc"
          "${tfmicro_kernels_dir}/reshape.cc"
          "${tfmicro_kernels_dir}/resize_nearest_neighbor.cc"
          "${tfmicro_kernels_dir}/round.cc"
          "${tfmicro_kernels_dir}/shape.cc"
          "${tfmicro_kernels_dir}/softmax.cc"
          "${tfmicro_kernels_dir}/softmax_common.cc"
          "${tfmicro_kernels_dir}/space_to_batch_nd.cc"
          "${tfmicro_kernels_dir}/split.cc"
          "${tfmicro_kernels_dir}/split_v.cc"
          "${tfmicro_kernels_dir}/squeeze.cc"
          "${tfmicro_kernels_dir}/strided_slice.cc"
          "${tfmicro_kernels_dir}/sub.cc"
          "${tfmicro_kernels_dir}/svdf.cc"
          "${tfmicro_kernels_dir}/svdf_common.cc"
          "${tfmicro_kernels_dir}/tanh.cc"
          "${tfmicro_kernels_dir}/transpose_conv.cc"
          "${tfmicro_kernels_dir}/unpack.cc"
          "${tfmicro_kernels_dir}/zeros_like.cc")

set(lib_srcs
          "${srcs_micro}"
          "${srcs_kernels}"
          "${src_micro_frontend}"
          "${tflite_dir}/kernels/kernel_util.cc"
          "${tflite_dir}/micro/memory_planner/greedy_memory_planner.cc"
          "${tflite_dir}/micro/memory_planner/linear_memory_planner.cc"
          "${tflite_dir}/c/common.c"
          "${tflite_dir}/core/api/error_reporter.cc"
          "${tflite_dir}/core/api/flatbuffer_conversions.cc"
          "${tflite_dir}/core/api/op_resolver.cc"
          "${tflite_dir}/core/api/tensor_utils.cc"
          "${tflite_dir}/kernels/internal/quantization_util.cc"
          "${tflite_dir}/schema/schema_utils.cc")

idf_component_register(
            SRCS "${lib_srcs}"
            INCLUDE_DIRS "." "third_party/gemmlowp"
                         "third_party/flatbuffers/include"
                         "third_party/ruy"
                         "third_party/kissfft")

# Reduce the level of paranoia to be able to compile TF sources
target_compile_options(${COMPONENT_LIB} PRIVATE
  -Wno-maybe-uninitialized
  -Wno-missing-field-initializers
  -Wno-type-limits)

target_compile_options(${COMPONENT_LIB} PRIVATE -fno-unwind-tables -ffunction-sections -fdata-sections -fmessage-length=0 -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -O3 -Wsign-compare -Wdouble-promotion -Wshadow -Wunused-variable -Wmissing-field-initializers -Wunused-function -Wswitch -Wvla -Wall -Wextra -Wstrict-aliasing -Wno-unused-parameter -DESP -DESP_NN -Wno-nonnull -Wno-nonnull -Wno-nonnull)
target_compile_options(${COMPONENT_LIB} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++11 -fno-rtti -fno-exceptions -fno-threadsafe-statics -fno-unwind-tables -ffunction-sections -fdata-sections -fmessage-length=0 -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -O3 -Werror -Wsign-compare -Wdouble-promotion -Wshadow -Wunused-variable -Wmissing-field-initializers -Wunused-function -Wswitch -Wvla -Wall -Wextra -Wstrict-aliasing -Wno-unused-parameter -DESP -DESP_NN -Wno-return-type -Wno-strict-aliasing -std=gnu++14 -Wno-return-type -Wno-strict-aliasing -std=gnu++14 -Wno-return-type -Wno-strict-aliasing -std=gnu++14 >)
target_compile_options(${COMPONENT_LIB} INTERFACE $<$<IN_LIST:-DTF_LITE_STATIC_MEMORY,$<TARGET_PROPERTY:${COMPONENT_LIB},COMPILE_OPTIONS>>:-DTF_LITE_STATIC_MEMORY>)
target_link_libraries(${COMPONENT_LIB} PRIVATE -lm)
