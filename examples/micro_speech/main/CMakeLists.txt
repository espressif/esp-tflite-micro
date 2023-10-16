
#
# Main component of TF Micro project 'micro_speech'.
#

idf_component_register(
    SRCS main.cc main_functions.cc audio_provider.cc feature_provider.cc
         no_micro_features_data.cc yes_micro_features_data.cc
         model.cc recognize_commands.cc command_responder.cc
         micro_features_generator.cc ringbuf.c
    PRIV_REQUIRES spi_flash driver esp_timer test_data
    INCLUDE_DIRS "")

    # Reduce the level of paranoia to be able to compile sources
target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-maybe-uninitialized
    -Wno-missing-field-initializers
    -Wno-error=sign-compare
    -Wno-error=double-promotion
    -Wno-type-limits)
