
This example shows how to run a MobileNet V2 based model with a custom classifier head using `esp-tflite-micro`.
The model is fully integer quantised (int8) with alpha=0.35

For the purpose of demonstrating how its done, `model.cc` contains a model we've trained to detect humans. It produces two labels:-
1. `0` indicates confidence score for `not person`
2. `1` indicates confidence score for `person`

**Note:-** To see how to use the pre-trained MobileNet V2 base model, check the `mobilenet` example.

## How to build your own model?
We have provided the `generate_model.ipynb` jupyter notebook to show how a custom model can be built. Follow the instructions in that notebook.

After generating the model in `mobilenet_custom_model.tflite` do the following
```bash
xxd -i mobilenet_custom_model.tflite > main/model.cc
```

include the `model.h` header file in `model.cc` and rename variables accordingly.

## How to test with your own images?
This example uses raw image data stored in `image1.cc` and `image2.cc` files. You can easily run it with your own images by doing the following:-

1. Convert image to raw format
```bash
python img_convert.py <img_path>
```

2. Convert raw file to .cc file
```bash
xxd -i <raw_img_path> >> main/image1.cc
```

3. Rename variables and add `image.h` header include in `image1.cc`