import numpy as np
import cv2

def gaussian(x, sigma):
     return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x ** 2) / (2 * sigma ** 2))

def gaussian_filter(size, sigma):
    half_size = size // 2
    kernel = np.zeros(size)

    for i in range(size):
        x = i - half_size
        kernel[i] = gaussian(x, sigma)

    # Normalize the kernel
    kernel /= np.sum(kernel)

    return kernel

#create filter based on specified file or generate a gaussian filter
def create_filter(M, sigma=None, filter_file=None):
    if filter_file:
        # Read filter coefficients from a specified file
        g = np.loadtxt(filter_file)
    else:
        # Create a Gaussian filter with the given sigma
        if sigma is None:
            sigma = M / 4.0
        g = gaussian_filter(M, sigma)
        g /= np.sum(g)  # Normalize the coefficients

    return g

def custom_filter2D(image, kernel):
    height, width = image.shape
    kernel_size = len(kernel)
    half_kernel_size = kernel_size // 2

    filtered_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            pixel_value = 0.0
            for m in range(-half_kernel_size, half_kernel_size + 1):
                for n in range(-half_kernel_size, half_kernel_size + 1):
                    if (i + m >= 0 and i + m < height) and (j + n >= 0 and j + n < width):
                        pixel_value += image[i + m, j + n] * kernel[m + half_kernel_size] * kernel[n + half_kernel_size]
            filtered_image[i, j] = pixel_value

    return filtered_image

def convolve_image(image, g):
    h = custom_filter2D(image, g)
    return h

M = 9
N = 300
border_offset = (M - 1) // 2

#read images
image1 = cv2.imread("pic1grey300.jpg",cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("pic2grey300.jpg",cv2.IMREAD_GRAYSCALE)

#read filter from file
# filter_file = "filter.txt"
# g = create_filter(M, filter_file=filter_file)

#create a Gaussian filter
g = create_filter(M, sigma=M / 4.0)

h = convolve_image(image1, g)
h[:border_offset, :] = 0
h[-border_offset:, :] = 0
h[:, :border_offset] = 0
h[:, -border_offset:] = 0

#display images for part 1
cv2.imshow('image1',image1)
cv2.imshow('Convolved image1',h)
cv2.waitKey(0)

cv2.destroyAllWindows()