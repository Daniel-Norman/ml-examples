from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np


class Pixel(object):
    def __init__(self, attrs):
        self.attrs = attrs

    def distance(self, other):
        return np.linalg.norm(self.attrs - other.attrs)


def init_means(pixels, k):
    return np.random.choice(pixels, k, replace=False)


def cluster_mean(cluster):
    if len(cluster) == 0:
        return None
    return Pixel(np.mean([pixel.attrs for pixel in cluster], axis=0))


# Returns the k means of the provided image as Pixels
def perform_k_means(image, k, num_it):
    pixels = []

    for row in image:
        for pixel in row:
            pixels.append(Pixel(pixel))

    means = init_means(pixels, k)

    for it in xrange(num_it):
        print 'Iteration:', it
        clusters = []
        for _ in xrange(k):
            clusters.append([])

        for pixel in pixels:
            min_cluster = np.argmin([pixel.distance(mean) for mean in means])
            clusters[min_cluster].append(pixel)

        means = [cluster_mean(cluster) for cluster in clusters]

    return means


def main():
    # Note: using small scale and large k can lead to errors
    image_name = raw_input("Image name: ")
    scale = max(float(raw_input("Rescale amount (0.05 to 1.0): ")), 0.05)
    k = int(raw_input("K: "))
    num_it = int(raw_input("Number of iterations: "))

    original_image = ndimage.imread(image_name)
    scaled_image = misc.imresize(original_image, scale)

    means = perform_k_means(scaled_image, k, num_it)

    # Remap the original image pixels to the closest mean pixels
    w, h, _ = original_image.shape
    for y in xrange(h):
        for x in xrange(w):
            min_mean_index = np.argmin([np.linalg.norm(original_image[x][y] - mean.attrs) for mean in means])
            original_image[x][y] = means[min_mean_index].attrs

    plt.imshow(original_image)
    plt.show()

    to_save = raw_input("Save result as output.png (y/n)? ")

    if to_save == 'y':
        misc.imsave('output.png', original_image)


if __name__ == "__main__":
    main()
