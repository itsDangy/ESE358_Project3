import numpy as np
import cv2
import matplotlib.pyplot as plt

###########
# PART 1
###########

#read image
image = plt.imread("pic1grey300.jpg",0)
#image = plt.imread("pic1grey300.jpg",0)
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

#display image for part 1
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

for i in range(gradientMagnitude.shape[0]):
    for j in range(gradientMagnitude.shape[1]):
        gradientMagnitude[i,j] = 1.0 if gradientMagnitude[i,j] > threshold else 0.0

plt.imshow(gradientMagnitude,cmap='gray')
plt.title("Canny Edges Image")
plt.show()

###########
# PART 3
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

def compute_cornerness(matrix,alpha):
    R = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            R[i,j] = np.linalg.det(matrix[i,j]) - 0.04 * np.square((np.trace(matrix[i,j])))
            
    threshold = alpha*np.max(np.abs(R))

    for i in range(N):
        for j in range(N):
            R[i,j] = 1.0 if R[i,j] > threshold else 0.0
    return R
R = compute_cornerness(matrix,0.001)

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
for i in range(N):
    for j in range(N):
        if corners[i,j] == 0:
            cv2.circle(color, (j,i), radius=1, color=(0, 255, 0))
plt.imshow(color)
plt.title("Circled Corners")
plt.show()