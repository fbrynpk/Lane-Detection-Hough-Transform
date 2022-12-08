import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from skimage.feature import peak_local_max
from pylab import *

#Read Image and Turn to Grayscale
img = cv2.imread('test_images/4.jpg')
row,col,ch = img.shape
gray = [ ]

def rgb2gray(Img):
  for i in range(row):
    for j in range(col):
      gray.append((0.2989*Img[i][j][0]+0.587*Img[i][j][1]+0.114*Img[i][j][2]))  
      
rgb2gray(img)
gr = np.array(gray).reshape(row,col).astype(np.uint8)

#Gaussian Blur Filter
def padding_img(img,k_size,style='edge'):
  return np.pad(img,k_size,style)

def gaussian_kernel(size):
    sigma=0.3*((size-1)*0.5 - 1) + 0.8
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]   
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2)))     
    return g/np.sum(g)

img_org = gr
KernelSize = 3
def Gaussian_Filter(KernelSize,image):
    row,col,ch =img.shape
    Gaussian_Kernel = gaussian_kernel(KernelSize)

    resx=np.zeros((row,col),np.uint8)
    padimg = padding_img(image,(KernelSize-1)//2,style='reflect')

    for x in range(row):
        for y in range(col):
          respixel=0
          for i in range(KernelSize):
              for j in range(KernelSize):
                  pixel=padimg[x+i,y+j]* Gaussian_Kernel[i][j]
                  respixel+= pixel         
          resx[x,y]=respixel
    return resx

gauss = Gaussian_Filter(KernelSize,img_org)
cv2.imshow('Gaussian Filter',gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('results/result_img1-4.jpg',gauss)

#Sobel Filter to Calculate Gradient and Angle Direction
def iterate_regions(img, kernel_size):
    h, w = img.shape
    for i in range(h - kernel_size + 1):
        for j in range(w - kernel_size + 1):
            img_region = img[i:(i + kernel_size), j:(j + kernel_size)]
            yield img_region, i, j
            
def sobel(img, filtering_type):
    h, w = img.shape

    horizontal = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    
    vertical = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
        
    Gx = np.zeros((h - 2, w - 2))
    Gy = np.zeros((h - 2, w - 2))

    for img_region, i, j in iterate_regions(img, 3):
        if filtering_type == 'dx':
            Gx[i, j] += np.sum(img_region * horizontal)
        elif filtering_type == 'dy':
            Gy[i, j] += np.sum(img_region * vertical)
        elif filtering_type == 'magnitude':
            Gx[i, j] += np.sum(img_region * horizontal)
            Gy[i, j] += np.sum(img_region * vertical)

    gradient = np.sqrt(Gx**2 + Gy**2)
    gradient = np.pad(gradient, (1, 1), 'constant')
    angle = np.arctan2(Gy, Gx)
    angle = np.pad(angle, (1, 1), 'constant')

    output = (255 * np.uint8(gradient > 255) + gradient * np.uint8(gradient <= 255))
    angle += math.pi * np.uint8(angle < 0)

    return output, angle
  
  #Non-Maximum Suppression
def non_maximum_suppression(gradient,angle):
    gradient_copy = gradient.copy()
    height,width= gradient.shape
    for j in range(1, height-1):
        for i in range(1, width-1):
            if (angle[j, i] >= 0 and angle[j, i] < 22.5 / 180 * math.pi) or (angle[j, i] >= 157.5 / 180 * math.pi and angle[j, i] < math.pi):
                if gradient[j, i] < gradient[j, i-1] or gradient[j, i] < gradient[j, i+1]:
                    gradient_copy[j, i] = 0
            elif angle[j, i] >= 22.5 / 180 * math.pi and angle[j, i] < 67.5 / 180 * math.pi:
                if gradient[j, i] < gradient[j-1, i-1] or gradient[j, i] < gradient[j+1, i+1]:
                    gradient_copy[j, i] = 0
            elif angle[j, i] >= 67.5 / 180 * math.pi and angle[j, i] < 112.5 / 180 * math.pi:
                if gradient[j, i] < gradient[j-1, i] or gradient[j, i] < gradient[j+1, i]:
                    gradient_copy[j, i] = 0
            elif angle[j, i] >= 112.5 / 180 * math.pi and angle[j, i] < 157.5 / 180 * math.pi:
                if gradient[j, i] < gradient[j+1, i-1] or gradient[j, i] < gradient[j-1, i+1]:
                    gradient_copy[j, i] = 0               
    return gradient_copy

gradient,angle = sobel(gauss, 'magnitude')
After_non_maximum_supression = non_maximum_suppression(gradient,angle)

#Double Thresholding
def double_threshold(img, minimum, maximum):
    output = img.copy()
    output[(output<maximum)*(output>=minimum)]=minimum
    output[output>=maximum]=255
    output[output<minimum] = 0   
    return output
thresholding = double_threshold(After_non_maximum_supression, minimum = 205, maximum =255)

#Edge Tracking by Hysteresis
WEAK_PIXEL = 50
STRONG_PIXEL = 255

def get_neighbors(img, i, j):
    neighbors = []
    neighbors.append(img[i+1,j])
    neighbors.append(img[i-1,j])
    neighbors.append(img[i,j+1])
    neighbors.append(img[i,j-1])
    
    neighbors.append(img[i+1,j+1])
    neighbors.append(img[i-1,j-1])
    neighbors.append(img[i-1,j+1])
    neighbors.append(img[i+1,j-1])
    
    return np.array(neighbors, dtype=np.int)

def edge_tracking(img):
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            if img[i, j] == WEAK_PIXEL:
                neighbors = get_neighbors(img, i, j)
                has_strong_neighbor = (sum(neighbors == STRONG_PIXEL) != 0)
                img[i, j] = STRONG_PIXEL if has_strong_neighbor else 0
    return img
canny = edge_tracking(thresholding)

cv2.imshow('canny_edge',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('results/result_img2-4.jpg',canny)

#Hough Transformation
Hough_theta_step = np.pi / 180
Hough_rho_step = 1
Hough_threshold = 30

def get_lines(acc, thetas, rhos, threshold, min_distance):
    acc = np.where(acc>=threshold, acc, 0)
    lines_idx = peak_local_max(acc, min_distance=min_distance)
    sorted_idx = sorted(lines_idx, key=lambda x: acc[x[0]][x[1]], reverse=True)
    lines = []
    for ri,ti in sorted_idx:
        t = thetas[ti]
        r = rhos[ri]
        lines.append((t,r))
    return lines, sorted_idx


def hough_line(img, threshold, min_distance=20, theha_step=np.pi/180, rho_step=1, draw=False):
    theha_step_deg = np.rad2deg(theha_step)
    thetas = np.deg2rad(np.arange(-90.0, 90.0, step=theha_step_deg))
    width, height = img.shape
    max_r = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
    rhos = np.arange(-max_r, max_r + 1, step=rho_step)

    acc = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(len(thetas)):
            # Calculate rho. diag_len is added for a positive index
            t = thetas[t_idx]
            rho = int(round(x * np.cos(t) + y * np.sin(t))) + max_r
            acc[rho, t_idx] += 1

    lines,idx = get_lines(acc, thetas, rhos, threshold,min_distance)
    if draw:
        x,y = [t for r,t in idx], [r for r,t in idx]
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8,25))
        ax1.imshow(acc)
        ax2.imshow(acc)
        ax2.plot(x,y, 'ro')
        ax2.autoscale(False)
        
    return lines

lines = hough_line(canny, threshold=Hough_threshold, theha_step=Hough_theta_step,
                   rho_step=Hough_rho_step, min_distance=20, draw=True)

def plot_lines(img, lines, N=0):
    if lines is None: return
    dst = img
    size = len(lines) if N == 0 else min(N, len(lines))
    for i in range(size):
        rho = lines[i][1]
        theta = lines[i][0]
            
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            
        cv2.line(dst, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
    
    cv2.imshow('hough_line',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('results/result_img3-4.jpg',dst)
    
plot_lines(img, lines)