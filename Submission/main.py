import numpy as np
import cv2

def gaussian(x, y, sigma):
    return (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))

def gaussian_filter(size, sigma):
    # Ensure the filter size is odd for a centered filter
    if size % 2 == 0:
        size += 1

    half_size = size // 2
    filter = np.zeros((size, size))

    for x in range(-half_size, half_size + 1):
        for y in range(-half_size, half_size + 1):
            filter[x + half_size, y + half_size] = gaussian(x, y, sigma)

    # Normalize the filter
    filter /= np.sum(filter)

    return filter

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
    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Create an empty result image with the same dimensions as the input image
    result = np.zeros_like(image)

    # Compute the padding needed for valid convolution
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the input image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Iterate over each pixel in the input image
    for y in range(image_height):
        for x in range(image_width):
            # Extract the region of interest (ROI) from the padded image
            roi = padded_image[y:y + kernel_height, x:x + kernel_width]

            # Perform element-wise multiplication and sum to apply the kernel
            result[y, x] = np.sum(roi * kernel)

    return result

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