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

## Build the example (Person Detection)

As the `person_detection` example requires an external component `esp32-camera`
for functioning hence we will have to manually clone it in `components/`
directory of the example with following commands.

 * `esp32-camera` should be
downloaded in `components/` dir of example as explained in `Build the
example`(below)

```
 git clone https://github.com/espressif/esp32-camera.git components/esp32-camera
 cd components/esp32-camera/
 git checkout eacd640b8d379883bff1251a1005ebf3cf1ed95c
 cd ../../
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

## License

These examples are covered under Apache2 License.
TensorFlow library code and third_party code contains their own license specified under respective [repos](https://github.com/tensorflow/tflite-micro).
