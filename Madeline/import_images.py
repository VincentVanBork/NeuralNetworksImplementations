from PIL import Image
import numpy as np


def flatten_image(name):
    im = Image.open(f'./numbers/{name}')
    data = im.load()
    width, height = im.size

    all_pixels = []
    for y in range(height):
        for x in range(width):
            cpixel = data[x, y]
            if any(cpixel):
                all_pixels.append(1)
            else:
                all_pixels.append(0)
    return all_pixels


def import_all_true():
    one = flatten_image("one_true.png")
    two = flatten_image("two_true.png")
    three = flatten_image("three_true.png")
    return np.array(one), np.array(two), np.array(three)
