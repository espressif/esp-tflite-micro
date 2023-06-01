# TFlite-Micro Tools for Espressif Chipsets

This repository offers a code generation tool designed to facilitate the creation of essential C++ code files necessary for deploying TensorFlow Lite models on Espressif micro-controllers. It caters specifically to the requirements of TensorFlow Lite Micro, enabling seamless integration of machine learning models into embedded systems.

By utilizing this tool, developers can generate C++ code that efficiently executes TensorFlow Lite models. This tool serves as a valuable asset for developers seeking to leverage the power of TensorFlow Lite for developing IoT applications through Espressif Chipsets.


# Features

- Converts TensorFlow Lite model files (.tflite) into C++ unsigned integer arrays storing hex values of .tflite models for efficient deployment on Espressif Chipsets.

- Generates a micro mutable op resolver based on the model, which extracts all the operations associated with the model.

- Creates template code for the main function, simplifying the integration of TensorFlow Lite models into Espressif based real time microcontroller applications.

# Setup Environment
- Clone the [tflite-micro](https://github.com/tensorflow/tflite-micro/) repository locally on your machine.
- Set paths for TFLITE_PATH and PYTHONPATH necessary for running the python scripts associated with generating array for model and micro mutable op resolver which extracts the model's operations.
```
export TFLITE_PATH=path/to/cloned/repository
export PYTHONPATH=path/to/directory/of/cloned/repository
```
- Once the paths have been set, create a python virtual environment by following the steps below.
```
# create virtual environment - env
python3 -m venv env

# activate the environment
source env/bin/activate

# install requirements.txt
pip install -r requirements.txt
```

# Usage

This script can convert any tflite model to example template which then can be integrated with any project.

```
python main.py model.tflite
```
This command performs the following:

- Convert the tflite model to a C++ unsigned integer array representation using generate_cc_arrays.py.

- Generate a micro mutable operation resolver based on the model using generate_micro_mutable_op_resolver_from_model.py.

- Generate templates for main functions using generate_main_templates.py.

- Extract relevant information from the generated files and create C++ files required for the application.


# Building the project

The command idf.py build is typically associated with the ESP-IDF (Espressif IoT Development Framework), which is the official development framework for the Espressif microcontrollers.

```
idf.py build
```

After executing `idf.py build`, it initiates the build process for the current project. This command compiles the source code files, resolves dependencies, and generates the firmware binary that can be flashed onto the Espressif microcontroller. Developers can further use this image for real time applications and use cases.

# Customization

You can modify the templates in the templates.py file to customize the generated code according to your project requirements.

Adjust the code and templates as needed to suit your specific use case.


# Resources

- [Tensorflow Lite for Microcontrollers](https://github.com/tensorflow/tflite-micro)

- [TensorFlow Lite Micro for Espressif Chipsets](https://github.com/espressif/tflite-micro-esp-examples)

- [Espressif IoT Development Framework](https://github.com/espressif/esp-idf)

- [CMake Documentation](https://cmake.org/documentation/)

 