import math as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import cv2 

from moviepy.editor import VideoFileClip
from IPython.display import HTML

os.chdir("C:\\Users\\maxim\\Documents\\Road lane detection Python")

listeFichiers=os.listdir()
image=plt.imread('vid0000.jpg').copy()

##Gray image:

def RGBtoGray(img):
    return(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

imageGray=RGBtoGray(image)
    
# plt.figure(1)
# plt.imshow(imageGray,cmap="gray")
# plt.show()

##Gaussian blur:

def GaussianBlur(img,kernel):
    cv2.GaussianBlur(img,(kernel,kernel),0)
    return(img)

imageBlur=GaussianBlur(imageGray,5)

# plt.figure(2)
# plt.imshow(imageBlur,cmap="gray")
# plt.show()

##Edge detection:

imageEdge=cv2.Canny(imageBlur,100,300)

# plt.figure(3)
# plt.imshow(imageEdge,cmap="gray")
# plt.show()

##Region of the road:


# upR=[350,570]
# upL=[350,410]
# lowL=[540,100]
# lowR=[540,915]

upR=[270,800]
upL=[270,70]
lowL=[400,70]
lowR=[400,800]


def SegmentationRoad(img):
    imgc=np.copy(img)
    n,p=np.shape(imgc)
    for i in range(n):
        for j in range(p):
            #up
            if i<upR[0] and i<upL[0]:
                imgc[i,j]=0
            #left and right
            if j<lowL[1] or j>lowR[1]:
                imgc[i,j]=0
            #low
            if i>lowR[0] and i>upL[0]:
                imgc[i,j]=0
            # #triangle L
            # if i< and j<
            # #triangle R
            
    return(imgc)
    
imageSeg=SegmentationRoad(imageEdge)

# plt.figure(4)
# plt.imshow(imageSeg,cmap="gray")
# plt.show()

## Hough transform

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
              minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    
    draw_lines(line_img, lines)
    return line_img

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.    
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    
#hough lines
rho = 1
theta = np.pi/180
threshold = 30
min_line_len = 20 
max_line_gap = 20

imageHouged = hough_lines(imageSeg, rho, theta, 
                  threshold, min_line_len, max_line_gap)

# plt.figure(5)
# plt.imshow(imageHouged,cmap="gray")
# plt.show()

##Final image with red lines

def ComparaisonListes(A,B):
    if len(A)==len(B):
        for k in range(len(A)):
            if A[k]!=B[k]:
                return(False)
    else:
        return(False)
    return(True)

def AddLines(img,imgHouged):
    n,p=np.shape(img)[0],np.shape(img)[1]
    for i in range(n):
        for j in range(p):
            if ComparaisonListes(imgHouged[i,j],[255, 0, 0]):
                img[i,j]=[255, 0, 0]
    return(img)
    
imageFinale=AddLines(image,imageHouged)

plt.figure(6)
plt.imshow(imageFinale,cmap="gray")
plt.show()

def RGBtoBGR(img):
    img = img[:,:,::-1]
    return(img)

cv2.imwrite('imgfinale'+'.png', RGBtoBGR(imageFinale))

##Image en boucle:

def AnalyseLinesRoute(cheminAccesDossier,seulbas,seuilhaut):
    os.chdir(cheminAccesDossier)
    # os.mkdir('images traitées')
    listeImg=os.listdir()
    n=len(listeImg)
    for k in range(n):
        print(k)
        img=plt.imread(listeImg[k]).copy()
        imgHouged=hough_lines(SegmentationRoad(cv2.Canny(GaussianBlur(RGBtoGray(img),5),seulbas,seuilhaut)), rho, theta, 
                  threshold, min_line_len, max_line_gap)
        imgTraitree=AddLines(img,imgHouged)
        
        # os.chdir(cheminAccesDossier+'\\'+'images traitées')
        cv2.imwrite('img'+str(k)+'.jpg', RGBtoBGR(img))
        
        # os.chdir(cheminAccesDossier)

# ##Video :
# 
# def process_image(image):
#     # grayscale the image
#     grayscaled = grayscale(image)
# 
#     # apply gaussian blur
#     kernelSize = 5
#     gaussianBlur = gaussian_blur(grayscaled, kernelSize)
# 
#     # canny
#     minThreshold = 100
#     maxThreshold = 200
#     edgeDetectedImage = canny(gaussianBlur, minThreshold, maxThreshold)
# 
#     # apply mask
#     lowerLeftPoint = [130, 540]
#     upperLeftPoint = [410, 350]
#     upperRightPoint = [570, 350]
#     lowerRightPoint = [915, 540]
# 
#     pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, 
#                     lowerRightPoint]], dtype=np.int32)
#     masked_image = region_of_interest(edgeDetectedImage, pts)
# 
#     # hough lines
#     rho = 1
#     theta = np.pi/180
#     threshold = 30
#     min_line_len = 20 
#     max_line_gap = 20
# 
#     houged = hough_lines(masked_image, rho, theta, threshold, min_line_len, 
#                          max_line_gap)
# 
#     # outline the input image
#     colored_image = weighted_img(houged, image)
#     return colored_image
# 
# output = 'car_lane_detection.mp4'
# clip1 = VideoFileClip("insert_car_lane_video.mp4")
# white_clip = clip1.fl_image(process_image)
# # %time white_clip.write_videofile(output, audio=False)