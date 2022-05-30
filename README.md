# TensorFlow Lite Micro for Espressif Chipsets

- As per TFLite Micro guidelines for vendor support, this repository has the examples needed to use Tensorflow Lite Micro on Espressif Chipsets (e.g., ESP32) using ESP-IDF platform.
- The base repo on which this is based can be found [here.](https://github.com/tensorflow/tflite-micro)

## Build Status

|   Build Type  |  Status    |
| -----------   |  --------- |
| Examples Build | [![CI](https://github.com/espressif/tflite-micro-esp-examples/actions/workflows/ci.yml/badge.svg)](https://github.com/espressif/tflite-micro-esp-examples/actions/workflows/ci.yml)

## How to Install

### Install the ESP IDF

Follow the instructions of the
[ESP-IDF get started guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html)
to setup the toolchain and the ESP-IDF itself.

The next steps assume that this installation is successful and the
[IDF environment variables are set](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#step-4-set-up-the-environment-variables). Specifically,
* the `IDF_PATH` environment variable is set
* the `idf.py` and Xtensa-esp32 tools (e.g., `xtensa-esp32-elf-gcc`) are in `$PATH`

## Build the example

Go to example directory (`examples/<example_name>`) and build the example.

Set the IDF_TARGET (For ESP32-S3 target, IDF version `release/v4.4` is needed)

```
idf.py set-target esp32s3
```

To build this, run:

```
idf.py build
```

### Load and run the example

To flash (replace `/dev/ttyUSB0` with the device serial port):
```
idf.py --port /dev/ttyUSB0 flash
```

Monitor the serial output:
```
idf.py --port /dev/ttyUSB0 monitor
```

Use `Ctrl+]` to exit.

The previous two commands can be combined:
```
idf.py --port /dev/ttyUSB0 flash monitor
```

  - Please follow example READMEs for more details.

## ESP-NN Integration
[ESP-NN](https://github.com/espressif/esp-nn) contains optimized kernel implementations for kernels used in TFLite Micro. The library is integrated with this repo and gets compiled as a part of every example. Additional information along with performance numbers can be found [here](https://github.com/espressif/esp-nn#performance).

### Performance Comparison

A quick summary of ESP-NN optimisations, measured on various chipsets:

|   Target  |   TFLite Micro Example  | without ESP-NN  | with ESP-NN | CPU Freq  |
| --------- | ----------------------- | --------------- | ----------- |-----------|
| ESP32-S3  |   Person Detection      |     2300ms      |     54ms    |  240MHz   |
| ESP32     |   Person Detection      |     4084ms      |    380ms    |  240MHz   |
| ESP32-C3  |   Person Detection      |     3355ms      |    426ms    |  160MHz   |

Note:
  - The above is time taken for execution of the `invoke()` call
  - Internal memory used

Detailed kernelwise performance can be found [here](https://github.com/espressif/esp-nn).

## Sync to latest TFLite Micro upstream

As per the upstream repository policy, the tflite-lib is copied into the components directory in this repository. We keep updating this to the latest upstream version from time to time. Should you, in any case, wish to update it locally, you may run the `scripts/sync_from_tflite_micro.sh` script.

## Contributing
- If you find an issue in these examples, or wish to submit an enhancement request, please use the Issues section on Github.
- For ESP-IDF related issues please use [esp-idf](https://github.com/espressif/esp-idf) repo.
- For TensorFlow related information use [tflite-micro](https://github.com/tensorflow/tflite-micro) repo.

## License

These examples are covered under Apache2 License.

TensorFlow library code and third_party code contains their own license specified under respective [repos](https://github.com/tensorflow/tflite-micro).
