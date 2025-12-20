from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import numpy as np

def test_edge():
    image = load_image('.tests/lena.jpg')
    image = median(image, ball(3))
    edge = edge_detection(image)
    edge_binary = edge > 50

    true = load_image('.tests/lena_edges.png')

    area = true.shape[0] * true.shape[1]
    score = np.sum(true == edge_binary)/area
    assert score > 0.9
    
