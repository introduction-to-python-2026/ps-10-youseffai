from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
  image = Image.open(path)
  image = np.array(image)
  return image

def edge_detection(image):

  grayscale_image = np.mean(image_array, axis=2)


  kernelY = np.array([
      [1, 2, 1],
      [0, 0, 0],
      [-1, -2, -1]
  ])


  kernelX = np.array([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
  ])

  edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
  edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)

  edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

  return edgeMAG
