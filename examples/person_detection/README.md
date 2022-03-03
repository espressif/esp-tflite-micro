# Person detection example

This example shows how you can use Tensorflow Lite to run a 250 kilobyte neural
network to recognize people in images captured by a camera.  It is designed to
run on systems with small amounts of memory such as microcontrollers and DSPs.
This uses the experimental int8 quantized version of the person detection model.

## Deploy to ESP32

The following instructions will help you build and deploy this sample
to [ESP32](https://www.espressif.com/en/products/hardware/esp32/overview)
devices using the [ESP IDF](https://github.com/espressif/esp-idf).

The sample has been tested on ESP-IDF version `release/v4.2` and `release/v4.4` with the following devices:
- [ESP32-DevKitC](http://esp-idf.readthedocs.io/en/latest/get-started/get-started-devkitc.html)
- [ESP32-S3-DevKitC](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/hw-reference/esp32s3/user-guide-devkitc-1.html)
- [ESP-EYE](https://github.com/espressif/esp-who/blob/master/docs/en/get-started/ESP-EYE_Getting_Started_Guide.md)

### Install the ESP IDF

Follow the instructions of the
[ESP-IDF get started guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html)
to setup the toolchain and the ESP-IDF itself.

The next steps assume that the
[IDF environment variables are set](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#step-4-set-up-the-environment-variables) :

 * The `IDF_PATH` environment variable is set
 * `idf.py` and Xtensa-esp32 tools (e.g. `xtensa-esp32-elf-gcc`) are in `$PATH`

### Dependencies

This example requires an external component `esp32-camera` which is the submodule of the example.
If you have not cloned current repo with `--recursive` option, please use `git submodule update` to get it cloned.

### Building the example

Set the chip target (For esp32s3 target, IDF version `release/v4.4` is needed):

```
idf.py set-target esp32s3
```

Then build with `idf.py`
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

### Using CLI for inferencing

Not all dev boards come with camera and you may wish to do inferencing on static images.
This example hence, by default uses static inferencing.
There are 10 [images](static_images/sample_images/README.md) embedded into the application.

  * To run an inferencing you need to type follwing on `idf.py monitor` window:

```
detect_image <image_number>
```
where `<image_number>` is in [0, 9].

The output is person and no_person score printed on the log screen.

  * To switch to camera mode just uncomment following line from [esp_main.h](main/esp_main.h):

  ```
  #define CLI_ONLY_INFERENCE 1
  ```

### Using Display

If you want to use display or your dev board supports it. (ESP-S3-EYE), you can enable it by disabling `CLI_ONLY_INFERENCE` and enabling following macro from `esp_main.h`
```
#define DISPLAY_SUPPORT 1
```

You will need to select appropriate drivers via menuconfig:

`idf.py menuconfig` > `Component Config` > `LCD drivers`

  * When display is enabled, you will see camera feed and `green` color strip. The strip color will change to `red` when a person is detected.
