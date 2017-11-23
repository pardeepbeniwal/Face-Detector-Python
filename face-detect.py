import cv2
#import matplotlib library
import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import time 
%matplotlib inline
test1 = cv2.imread('/home/pardeep/Desktop/11.png')
#convert the test image to gray image as opencv face detector expects gray images 
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_img, cmap='gray') 

haar_face_cascade = cv2.CascadeClassifier('/home/pardeep/Desktop/py_test/haarcascade_frontalface_alt.xml')
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);  
 
#print the number of faces found 
print('Faces found: ', len(faces))
