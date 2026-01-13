from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
  image = Image.open(path)
  image = np.array(image)
  return image

def edge_detection(image_array):
  # Convert to grayscale
  grayscale_image = np.mean(image_array, axis=2)

  # Define kernelY
  kernelY = np.array([
      [1, 2, 1],
      [0, 0, 0],
      [-1, -2, -1]
  ])

  # Define kernelX
  kernelX = np.array([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
  ])

  # Apply convolution
  edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
  edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)

  # Compute edgeMAG
  edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

  return edgeMAG
