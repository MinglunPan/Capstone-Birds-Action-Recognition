import imageio
import IPython
import matplotlib
import numpy as np

def load(file_path):
    return matplotlib.image.imread(file_path)
def generateGIF(img_list, save_path):
    imageio.mimsave(save_path, img_list)
def unit8(img_array):
    if img_array.dtype == 'float32':
        return (img_array*255).astype(np.uint8)
    else:
        return img_array