import numpy as np
import math, os, imageio
from collections import defaultdict
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
#from scikit-image.transform import rescale, resize, downscale_local_mean
import yaml, json, re
from sklearn.decomposition import PCA
from PIL import Image as Im
size = 256;

class Image:
    def __init__(self, path, gray=True):
        self.path = path
        self.pixels = []
        self.pixel_features = []
        self.label = []
        self.label_features = []

    def get_pixels(self):
        self.pixels = np.array(Im.open(self.path)).astype(np.uint8)
        
 #       self.pixels = ndimage.imread(self.path)
        '''
        self.pixels = resize(self.pixels, (size, size))
        #img = Im.open(self.path); img = img.resize((256, 256))
        #self.pixels = np.array(img.getdata())
        '''
        return self.pixels

    def get_pixel_features(self):
        #self.pixels = ndimage.imread(self.path)
        img = Im.open(self.path); img = img.resize((256, 256))
        self.pixel_features = np.array(img.getdata())
        self.pixel_features = np.array([[0 if item == 255 else 1 for item in row] for row in self.pixel_features])
        return self.pixel_features

    def get_label(self):
        self.label = self.path.split('/')[-2]
        return self.label

    def get_label_features(self):
        # wvec path
        #wvecs = get_wvecs()
        self.label_features = wvecs[self.label]
        return self.label_features


        
if __name__ == '__main__':
    path = '/Users/romapatel/Documents/proto/data/tu-berlin/sketches_png/'
    categories = os.listdir(path)
    for category in sorted(categories)[:2]:
        if 'pixel' in category or '.DS' in category: continue
        files = os.listdir(path + category)
        for filename in files[:2]:
            if '.DS' in filename: continue
            a = Image(path + category + '/' + filename)
            for row in a.get_pixels():
                print(row)
            for row in a.get_pixel_features():
                print(row)
            plt.imshow(a.get_pixels(), cmap='gray')
            plt.show()

            #plt.imshow(a.get_pixel_features())
            #plt.show()
            #plt.savefig('/Users/romapatel/Desktop/a.png')
