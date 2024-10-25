import numpy as np
import cv2
import matplotlib.pyplot as plt

###########
# PART 1
###########

#read image
image = plt.imread("pic1grey300.jpg",0)
# image = plt.imread("pic1grey300.jpg",0)
plt.imshow(image,cmap='gray')
plt.title("Original Image")
plt.show()
M = 9
N = image.shape[0]


def gaussian(size, sigma):
    ax = [i - ((size - 1) / 2) for i in range(size)]
    coefficient = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    gaussian = np.outer(coefficient, coefficient)
    return gaussian / gaussian.sum()

#create filter based on specified file or generate a gaussian filter
def create_filter(M, sigma=None, filter_file=None):
    if filter_file:
        # Read filter coefficients from a specified file
        g = np.loadtxt(filter_file)
    else:
        # Create a Gaussian filter with the given sigma
        if sigma is None:
            sigma = M / 4.0
        g = gaussian(M, sigma)
    return g

def convolve_2d(image, kernel):
    row, col = image.shape
    offset = (M - 1) // 2
    output = np.zeros((row, col))

    for i in range(offset,row-offset):
        for j in range(offset,col-offset):
            for k in range(M):
                for l in range(M):
                    x = i - (k - offset)
                    y = j - (l - offset)

                    if 0<=x < row and 0<=y < col:
                        output [i,j]+=kernel[k,l]*image[x,y]
    return output

# Read filter from textfile
# g = create_filter(M, filter_file="filter.txt")

#create a Gaussian filter
sigma = M/4.0
g = create_filter(M, sigma)

h = convolve_2d(image, g)

# display image for part 1
plt.imshow(h,cmap='gray')
plt.title("Convolved Image")
plt.show()

###########
# PART 2
###########

def convolve_1d(image, kernel):
    size = len(kernel)
    border = (size-1)//2
    output = np.copy(image)

    for i in range(border, len(image)-border):
        output[i] = np.sum(image[i-border:i+border+1] * kernel)
    return output

def convolve_2d_separable(image, gaussian_2d)->np.ndarray:
    gaussian_1d = gaussian_2d[:,0]
    gaussian_1d /= sum(gaussian_1d)
    
    h1 = np.zeros_like(image)
    for i in range(N):
        h1[i,:] = convolve_1d(image[i,:],gaussian_1d)

    h2 = h1
    for i in range(N):
        h2[:,i] = convolve_1d(h1[:,i],gaussian_1d)
    return h2

def gradient(image):
    row,col = image.shape
    h = image / 255.0
    hx,hy = np.zeros((row,col)),np.zeros((row,col))
    for i in range(row - 1):
        for j in range(col):
            hx[i,j] = h[i+1,j] - h[i,j]
    for i in range(row):
        for j in range(col-1):
            hy[i,j] = h[i,j+1] - h[i,j]
    return hx/10,hy/10,np.hypot(hx,hy)

sigma = 1
g = create_filter(M,sigma)
threshold = 20.0/255.0
h2 = convolve_2d_separable(image,g)
_,_,gradientMagnitude = gradient(h2)
Canny = np.zeros_like(gradientMagnitude)

for i in range(gradientMagnitude.shape[0]):
    for j in range(gradientMagnitude.shape[1]):
        Canny[i,j] = 1.0 if gradientMagnitude[i,j] > threshold else 0.0

plt.imshow(gradientMagnitude,cmap='gray')
plt.title("Canny Edges Image")
plt.show()

###########
# PART 3a
###########

g = create_filter(M,sigma=2)
convolved = convolve_2d_separable(image,g)

Ix, Iy, _ = gradient(convolved)
A = np.zeros((N,N))
B = np.zeros((N,N))
C = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        A[i,j] = np.square(Ix[i,j])
        B[i,j] = np.square(Iy[i,j])
        C[i,j] = Ix[i,j] * Iy[i,j]

smooth=[]
for x in [A,B,C]:
    g = create_filter(9,5)
    smooth.append(convolve_2d_separable(x,g))
Aprime, Bprime, Cprime = smooth

matrix = np.zeros((N,N,2,2))
matrix[:, :, 0, 0] = Aprime
matrix[:, :, 0, 1] = Cprime
matrix[:, :, 1, 0] = Cprime
matrix[:, :, 1, 1] = Bprime

def cornerness(matrix,alpha):
    R = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            R[i,j] = np.linalg.det(matrix[i,j]) - 0.04 * np.square((np.trace(matrix[i,j])))
            
    threshold = alpha*np.max(np.abs(R))

    for i in range(N):
        for j in range(N):
            R[i,j] = 1.0 if R[i,j] > threshold else 0.0
    return R
R = cornerness(matrix,0.001)

def nonmaxima_suppresion(R):
    row,col = R.shape
    corners = np.zeros((row,col))
    for i in range(1,row-1):
        for j in range(1,col-1):
            neighbors = R[i-1:i+2, j-1:j+2]
            if R[i,j] == np.max(neighbors):
                corners[i,j] = 1
    return corners
corners = nonmaxima_suppresion(R)

color = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
for i in range(4,N-5):
    for j in range(4,N-5):
        if corners[i,j] == 0:
            cv2.circle(color, (j,i), radius=1, color=(0, 255, 0))
plt.imshow(color)
plt.title("Circled Corners")
plt.show()

###########
# PART 3b
###########

hist = np.array([0,0,0,0,0,0,0,0],dtype=float)
itr = 0
gradient_vector = np.arctan2(Ix,Iy)
gradient_vector = gradient_vector * 180 / np.pi
for i in range(5,N-6):
    for j in range(5,N-6):
        if corners[i,j] == 0:
            for k in range(i-5,i+6):
                for l in range(j-5,j+6):
                    hist[round(gradient_vector[k,l]/45)] += gradientMagnitude[k,l]
            maxval = 0
            for o in range(len(hist)):
                if hist[o] > maxval:
                    maxval = hist[o]
                    itr = o
            histn = np.roll(hist,itr)
            print("pixel at (" + str(i) +","+str(j)+") has histogram of ")
            print(histn)

#########
# PART 4
#########

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image1 = cv2.imread("pic1grey300.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("pic2grey300.jpg", cv2.IMREAD_GRAYSCALE)

# Arrays to store selected points for both images
points1 = []
points2 = []

# Function to capture points in Image 1
def select_points_image1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points1.append((x, y))
        print(f"Image 1 - Point: {(x, y)}")

# Function to capture points in Image 2
def select_points_image2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points2.append((x, y))
        print(f"Image 2 - Point: {(x, y)}")

# Display image 1 and select points
cv2.namedWindow("Image 1")
cv2.setMouseCallback("Image 1", select_points_image1)
cv2.imshow("Image 1", image1)
print("Click on points in Image 1 and press any key when done.")
cv2.waitKey(0)  # Press any key after selecting points
cv2.destroyWindow("Image 1")

# Display image 2 and select points
cv2.namedWindow("Image 2")
cv2.setMouseCallback("Image 2", select_points_image2)
cv2.imshow("Image 2", image2)
print("Click on points in Image 2 and press any key when done.")
cv2.waitKey(0)  # Press any key after selecting points
cv2.destroyWindow("Image 2")

# Ensure we have at least 15 points in each image
if len(points1) >= 15 and len(points2) >= 15:
    # Convert points to NumPy arrays
    pts1 = np.float32(points1[:15])  # Points from Image 1 (reference image)
    pts2 = np.float32(points2[:15])  # Points from Image 2 (image to be transformed)

    # Estimate the affine transformation matrix to map Image 2 onto Image 1
    affine_matrix, inliers = cv2.estimateAffine2D(pts2, pts1)

    # Print the affine matrix
    print("Affine Transformation Matrix:")
    print(affine_matrix)

    # Separate the transformation matrix into rotation, scaling, and translation components
    rotation_scaling_matrix = affine_matrix[:, :2]
    translation_vector = affine_matrix[:, 2]

    print("\nRotation and Scaling Matrix:")
    print(rotation_scaling_matrix)

    print("\nTranslation Vector:")
    print(translation_vector)

    # Apply the transformation to Image 2 for visualization
    rows, cols = image1.shape  # Use dimensions of Image 1 (reference)
    transformed_image2 = cv2.warpAffine(image2, affine_matrix, (cols, rows))

    # Plot the original Image 2 and the transformed Image 2 side by side
    plt.subplot(1, 2, 1)
    plt.imshow(image2, cmap='gray')
    plt.title("Original Image 2")

    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image2, cmap='gray')
    plt.title("Transformed Image 2 to Match Image 1")

    plt.show()

else:
    print("Please select at least 15 points in both images.")

