"""
 @Time    : 2023-03-21 07:29:40
 @Author  : Hong-Shuo Chen
 @E-mail  : hongshuo@usc.edu
 
 @Project : Camouflage Object Detection
 @File    : image_clustering.py
 @Function: Image Clustering
"""
import cv2
import numpy as np
import multiprocessing as mp
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

# convert rgb to other color space
def rgb2color(rgb, color_space='hsv'):
    rgb = rgb*255
    rgb = rgb.astype(np.uint8)
    if color_space == 'rgb': # 255 255 255
        return rgb
    elif color_space == 'hsv': # 179 255 255
        color = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    elif color_space == 'lab': # 255 255 255
        color = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    elif color_space == 'ycrcb': # 255 255 255
        color = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    elif color_space == 'luv': # 255 255 255
        color = cv2.cvtColor(rgb, cv2.COLOR_RGB2Luv)
    elif color_space == 'yuv': # 255 255 255
        color = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    else:
        raise ValueError("Color space not supported")
    return color

def quantize(x, bins):
    return np.digitize(x, np.linspace(0, 256, bins + 1)[1:-1])

def hsv_quantization(hsv, num_bins=[12,4,4]):
    h, s, v = hsv[:,:,:,0], hsv[:,:,:,1], hsv[:,:,:,2]
    if num_bins[0] == 12:
        h = ((h-8)//15+1)%12
    else:
        raise ValueError("Number of bins for hue not supported")
    s = quantize(s, num_bins[1])
    v = quantize(v, num_bins[2])
    q = h*num_bins[1]*num_bins[2] + s*num_bins[2] + v
    return q

def color_quantization(color, color_space='hsv', num_bins=[12,4,4]):
    if color_space == 'hsv':
        return hsv_quantization(color, num_bins)
    else:
        color[0] = quantize(color[0], num_bins[0])
        color[1] = quantize(color[1], num_bins[1])
        color[2] = quantize(color[2], num_bins[2])
    return color

# get histogram
def histogram(input, bins):
    hist, bin_edges = np.histogram(input, bins=np.arange(bins+1), density=True)
    return hist

class ImageClustering:
    def __init__(self, num_clusters=32, batch_size=100, num_bins=[12,4,4], color_space="hsv", random_state=0, num_workers=32):
        self.num_clusters = num_clusters
        self.num_workers = num_workers
        self.num_bins = num_bins
        self.bins = np.prod(num_bins)
        self.color_space = color_space
        if self.num_clusters != 1:
            # self.image_kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
            self.image_kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size, random_state=random_state)
        
    def partial_fit(self, images):
        if self.num_clusters == 1:
            return np.zeros(images.shape[0]).astype(int)
        
        images = images.permute(0,2,3,1).numpy()
        
        # rgb to other color space       
        rgb = []
        for img in images:
            rgb.append([img, self.color_space])
        with mp.Pool(self.num_workers) as pool:
            color = pool.starmap(rgb2color, rgb)
            color = np.array(color)

        # quantization
        q = color_quantization(color, self.color_space, self.num_bins)
        
        # calculate histogram
        inputs = []
        for img in q:
            inputs.append([img, self.bins])
        with mp.Pool(self.num_workers) as pool:
            hist = pool.starmap(histogram, inputs)
            hist = np.array(hist)
        
        # image clustering
        self.image_kmeans.partial_fit(hist)
        
    def predict(self, images):
        if self.num_clusters == 1:
            return np.zeros(images.shape[0]).astype(int)
        
        images = images.permute(0,2,3,1).numpy()
        # rgb to other color space   
        if len(images) == 1:
            color = []
            color.append(rgb2color(images[0], self.color_space))
            color = np.array(color)
        else: 
            rgb = []
            for img in images:
                rgb.append([img, self.color_space])
            with mp.Pool(self.num_workers) as pool:
                color = pool.starmap(rgb2color, rgb)
                color = np.array(color)
        
        # quantization
        q = color_quantization(color, self.color_space, self.num_bins)

        # calculate histogram
        if len(images) == 1:
            hist = []
            hist.append(histogram(q[0], self.bins))
            hist = np.array(hist)
        else:
            inputs = []
            for img in q:
                inputs.append([img, self.bins])
            with mp.Pool(self.num_workers) as pool:
                hist = pool.starmap(histogram, inputs)
                hist = np.array(hist)
        # image clustering
        y = self.image_kmeans.predict(hist)
        return y
