"""

SPDX-FileCopyrightText: 2023 Espressif Systems (Shanghai) CO LTD

SPDX-License-Identifier: Apache-2.0

"""

# import necessary modules
import subprocess
import os
import re
import argparse
from templates import *
import shutil
import datetime


current_year = datetime.date.today().year

# subprocess to run generate_cc_array.py
"""
RUN from tools
> python generate_cc_arrays.py main hello_world.tflite 
"""

parser = argparse.ArgumentParser(
    description="Description: Accept .tflite or .bmp file as input from the user"
)
parser.add_argument("input_file", help=".tflite or .bmp format")
args = parser.parse_args()
tflite_file = args.input_file

# TFLITE_PATH
tflite_folder_path = os.environ.get('TFLITE_PATH')
print("tflite_folder_path: ", tflite_folder_path)

# extract model name from .tflite file
x = tflite_file.split(".")[0]

generate_cc_array = [
    "python",
    os.path.join(tflite_folder_path, "tensorflow/lite/micro/tools/generate_cc_arrays.py"),
    f"{x}/main",
    tflite_file,
]
subprocess.run(generate_cc_array, check=True)

# subprocess to run generate_micromutable_op_resolver.py
"""
RUN from tools
> python gen_micro_mutable_op_resolver/generate_micro_mutable_op_resolver_from_model.py --common_tflite_path=. --input_tflite_files=hello_world.tflite --output_dir=main
"""
generate_micromutable_op_resolver = [
    "python",
    os.path.join(tflite_folder_path, "tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver/generate_micro_mutable_op_resolver_from_model.py"),
    f"--common_tflite_path=.",
    f"--input_tflite_files={tflite_file}",
    f"--output_dir={x}/main",
]
subprocess.run(generate_micromutable_op_resolver, check=True)

# Define the file path
file_path = f"{x}/main/{x}_model_data.cc"

# Create a temporary file path
temp_file_path = file_path + ".temp"

# Open the input file for reading
with open(file_path, "r") as input_file:
    lines = input_file.readlines()

# Modify the include statement
modified_lines = [
    f'#include "{x}_model_data.h"\n'
    if line.startswith(f'#include "{x}/main/{x}_model_data.h"')
    else line
    for line in lines
]

# Open the temporary file for writing
with open(temp_file_path, "w") as temp_file:
    temp_file.writelines(modified_lines)

# Replace the original file with the temporary file
shutil.move(temp_file_path, file_path)

# subprocess to generate .cc and .h templates
# generate_main_templates = ["python", "generate_main_templates.py"]
# subprocess.run(generate_main_templates, check=True)

# extract operations from gen_micro_mutable_op_resolver.h
with open(f"{x}/main/gen_micro_mutable_op_resolver.h") as cppfile:
    operations = []
    for line in cppfile:
        if "micro_op_resolver." in line:
            # print(line)
            line = "".join(line.split())
            operations.append(line)
# print(operations)

# extract the name of the unsigned integer array
with open(f"{x}/main/{x}_model_data.cc", "r") as file:
    cpp_content = file.read()

pattern = r"const\s+unsigned\s+char\s+(\w+)\[\]"
match = re.search(pattern, cpp_content)

if match:
    array_name = match.group(1)
    print("/*")
    print("Array name:", array_name)
else:
    print("Array name not found.")


"""
Format the unsigned int array from model_data.cc file
"""
with open(f"{x}/main/{x}_model_data.cc", "r+") as file:
    cpp_code = file.read()

    start_index = cpp_code.find("alignas(16) const unsigned char")
    end_index = cpp_code.find("};", start_index)

    array_string = cpp_code[start_index:end_index]

    elements = [element.strip() for element in array_string.split(",")]

    formatted_elements = []
    for i, element in enumerate(elements):
        if len(element) == 3:
            if element == "0x0":
                element = "0x00"
            else:
                element = element[:2] + "0" + element[2:]
        formatted_elements.append(element)

    formatted_array_lines = []
    for i in range(0, len(formatted_elements), 12):
        line_elements = formatted_elements[i : i + 12]
        line = ", ".join(line_elements)
        formatted_array_lines.append(line)

    formatted_array_string = ",\n".join(formatted_array_lines)

    formatted_cpp_code = (
        cpp_code[:start_index] + formatted_array_string + cpp_code[end_index:]
    )

    file.seek(0)
    file.truncate()
    file.write(formatted_cpp_code)

# add tabs and close the array on a new line
with open(f"{x}/main/{x}_model_data.cc", "r+") as file:    
    lines = file.readlines()

    formatted_lines = []
    for i in lines[:5]:
        formatted_lines.append(i)
    lines = lines[5:]

    last_line = lines[-1]

    last_line = last_line.split("};")
    last_line = [last_line[0], "};"]

    last_element = last_line[-2]
    # last_element = '0x04, 0x00, 0x00, 0x00'

    lines[-1] = last_element
    # last_line = ['0x04, 0x00, 0x00, 0x00', '};']

    closing_bracket = last_line[-1]

    for line in lines:
        if "alignas(16) const unsigned char" in line:
            formatted_lines.append(line.split("{")[0] + "{\n")
            data = line.split("{")[1].split("}")[0].strip().split(", ")
            for i in range(0, len(data), 12):
                formatted_lines.append("    " + ", ".join(data[i:i+12]) + "\n")
        else:
            formatted_lines.append("    "+line)

    formatted_lines.append('\n' + closing_bracket)

    file.seek(0)
    file.writelines(formatted_lines)
    file.truncate()

# VARIABLES
model_name = array_name
num_of_operations = len(operations)
resolver = "micro_op_resolver"
model_name_header = x
print("Name of the Model:", model_name)
print("Number of Operations:", num_of_operations)
print("Model Name Header:", model_name_header)
print("Description of Operations: ")
for i in operations:
    print(i)
print("*/")

results = cppTemplate.safe_substitute(
    model_name=array_name,
    num_of_operations=num_of_operations,
    resolver=resolver,
    model_name_header=x,
    current_year=current_year
)
cmake_template = CMakeLists_txt.safe_substitute(model_name_header=x)

# store the results in main_functions.cc inside main folder
folder_path = f"{x}/main"
file_path = os.path.join(folder_path, "main_functions.cc")

with open(file_path, "w") as file:
    file.write(results)

# x is passed to ${model_header_name} and written inside CMakeLists.txt
file_path_cmake = os.path.join(folder_path, "CMakeLists.txt")
with open(file_path_cmake, "w") as file:
    file.write(cmake_template)

"""
The following code generates:
- main.cc
- main_functions.h
- output_handler.cc
- output_handler.h
- constants.h
- constants.cc
- Top Level: CMakeLists.txt
"""

main_cc = main_cc
main_functions_h = main_functions
output_handler_cc = output_handler_cc
output_handler_h = output_handler_h
constants_h = constants_h
constants_cc = constants_cc
CMakeLists_txt = CMakeLists_txt
topLevelCMakeList = topLevelCMake

result_a = main_cc.safe_substitute(current_year=current_year)
result_b = main_functions_h.safe_substitute(current_year=current_year)
result_c = output_handler_cc.safe_substitute(current_year=current_year)
result_d = output_handler_h.safe_substitute(current_year=current_year)
result_e = constants_h.safe_substitute(current_year=current_year)
result_f = constants_cc.safe_substitute(current_year=current_year)
result_g = topLevelCMakeList.safe_substitute(current_year=current_year)

files = {
    "main.cc": result_a,
    "main_functions.h": result_b,
    "output_handler.cc": result_c,
    "output_handler.h": result_d,
    "constants.h": result_e,
    "constants.cc": result_f,
}

# top level CMakeLists.txt
directory = f"{x}/main"
for filename, result in files.items():
    file_path = os.path.join(directory, filename)
    with open(file_path, "w") as file:
        print(filename)
        file.write(result)

file_path_topCMake = os.path.join(f"{x}", "CMakeLists.txt")
with open(file_path_topCMake, "w") as file_g:
    file_g.write(result_g)
