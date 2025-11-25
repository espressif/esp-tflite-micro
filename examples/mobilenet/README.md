
This example shows how to run a MobileNet V2 model using `esp-tflite-micro`.
The model is int8 quantised with `alpha=0.35`.

**Note:-** MobileNet V2's classifier head can be easily modified to detect a custom number of labels after some re-training and fine-tuning. We have created another example to show how to do this along with a jupyter notebook to generate your own custom models. Check it out in the `mobilenet_custom` example.

## How to test with your own images?
This example uses raw image data stored in `image.cc` file. You can easily run it with your own images by doing the following:-

1. Convert image to raw format
```bash
python img_convert.py <img_path>
```

2. Convert raw file to .cc file
```bash
xxd -i <raw_img_path> >> main/image.cc
```

3. Rename variables and add `image.h` header include in `image.cc`