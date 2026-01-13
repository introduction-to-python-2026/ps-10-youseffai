from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

def load_image(path):
  image = Image.open(path)
  image = np.array(image)
  return image
print(load_image("/content/Screenshot 2025-11-21 143400.png"))


def edge_detection(image_array):

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


plt.imshow(edge_detection(load_image("/content/Screenshot 2025-11-21 143400.png")),cmap="grey")


edge_mag_clean_image = edge_detection(clean_image)
plt.figure(figsize=(10, 6))
plt.hist(edge_mag_clean_image.ravel(), bins=50, alpha=0.7)
plt.title('Histogram of Edge Magnitude (Noise-Free Image)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()




threshold_value = 70 
edge_binary = edge_mag_clean_image > threshold_value


plt.figure(figsize=(10, 8))
plt.imshow(edge_binary, cmap='gray')

plt.axis('off')
plt.show()
edge_image_to_save = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image_to_save.save('my_edges.png')
