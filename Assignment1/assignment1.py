import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('./lenna.png')
plt.imshow(image)
plt.show()

image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
plt.imshow(image_flip)
plt.show()

image_rotate = image.transpose(Image.ROTATE_180)
plt.imshow(image_rotate)
plt.show()

image_resize = image.resize((int(image.width / 2), int(image.height / 2)))
plt.imshow(image_resize)
plt.show()