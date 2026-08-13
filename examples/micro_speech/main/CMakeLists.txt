
#
# Main component of TF Micro project 'micro_speech'.
#

set(priv_reqs spi_flash driver esp_timer test_data)
# gpio/i2s drivers live in separate components since IDF v5.3 (required on v6+)
if("${IDF_VERSION_MAJOR}.${IDF_VERSION_MINOR}" VERSION_GREATER_EQUAL "5.3")
    list(APPEND priv_reqs esp_driver_gpio esp_driver_i2s)
endif()

idf_component_register(
    SRCS main.cc main_functions.cc audio_provider.cc feature_provider.cc
         no_micro_features_data.cc yes_micro_features_data.cc
         model.cc recognize_commands.cc command_responder.cc
         micro_features_generator.cc ringbuf.c i2s_setup.cc
    PRIV_REQUIRES ${priv_reqs}
    INCLUDE_DIRS "")

    # Reduce the level of paranoia to be able to compile sources
target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-maybe-uninitialized
    -Wno-missing-field-initializers
    -Wno-error=sign-compare
    -Wno-error=double-promotion
    -Wno-type-limits)
