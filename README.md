# TensorFlow Lite Micro Examples for ESP-IDF Platforms

This repository has the  examples needed to use Tensorflow Lite Micro on ESP-IDF platforms.

## How to Install

### Install the ESP IDF

Follow the instructions of the
[ESP-IDF get started guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html)
to setup the toolchain and the ESP-IDF itself.

The next steps assume that the
[IDF environment variables are set](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#step-4-set-up-the-environment-variables) :
* The `IDF_PATH` environment variable is set. * `idf.py` and Xtensa-esp32 tools
(e.g., `xtensa-esp32-elf-gcc`) are in `$PATH`.

## Build the example

Go to example directory (`examples/<example_name>`) and build the example.

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

## Sync to latest sources

- `components/tflite-lib` directory contains the tflite micro sources.
- If you need the latest head, just run `sync_from_tflite_micro.sh`.
## License

These examples are covered under Apache2 License.
TensorFlow library code and third_party code contains their own license specified under respective [repos](https://github.com/tensorflow/tflite-micro).
