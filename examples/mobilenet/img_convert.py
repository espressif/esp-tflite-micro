
import sys
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from array import array

def convert_and_save(path):
    img = img_to_array(load_img(path, target_size=(224, 224)))
    img = array('B', img.flatten().astype(np.uint8))
    with open("img.raw", "wb") as f:
        f.write(img)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_convert.py img_path")
        exit(0)
    
    img_path = sys.argv[1]
    convert_and_save(img_path)
