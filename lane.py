
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import rospy
import signal
import sys
from std_msgs.msg import String,Int32,Int32MultiArray

# def signal_handler(signal, frame):
#     print("\nCtrl+C detected. Exiting!")
#     sys.exit(0)

lanePub = rospy.Publisher('lane', Int32MultiArray, queue_size=10)
rospy.init_node('data')
rate = rospy.Rate(10000)
# signal.signal(signal.SIGINT, signal_handler)
img = cv2.imread("./frame.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

canny = cv2.Canny(gray_img, 200, 300)

roi_vertices = np.array([[0,720],[0,600], [600,240], [720,240], [1280,600],[1280,720]])

def roi(image, vertices):
    mask = np.zeros_like(image)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img


roi_image = roi(canny, np.array([roi_vertices], np.int32))
lines = cv2.HoughLinesP(roi_image, 1, np.pi/180, 110, minLineLength=100, maxLineGap=500)

def draw_lines(image, hough_lines):
    slopes=[]
    filter_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = round(float(y1-y2)/float(x1-x2),1)
        if len(slopes)==0 or slope not in slopes:
          slopes.append(slope)
          filter_lines.append(line)
    return image,slopes,filter_lines


transform_matrix_path = "./birdeyes/transform_matrix.npy"
M = np.load(transform_matrix_path)
M = np.array(M, np.float32)


final_img,slopes,filter_lines = draw_lines(img, lines)  # Result

filter_lines = np.array(filter_lines)

per_im =cv2.transpose(cv2.warpPerspective(img, M,(1800,1300)))
per_lines =cv2.perspectiveTransform(np.array(filter_lines.reshape(1,-1,2),dtype=float), M)
per_lines = per_lines.reshape(-1,4).astype(int)
cnt=0
filter_per_lines = []
for line in per_lines:
  y1, x1, y2, x2 = line
  if abs(x1-x2)<0.275*max(x1,x2):
    cv2.line(per_im, (int((x1+x2)/4 +max(x1,x2)/2), int((y1+y2)/4 +max(y1,y2)/2)), (max(x1,x2), max(y1,y2)), (0, 255, 0), 2)
    if y1<y2:
      filter_per_lines.append([x1,y1,x2,y2])
    else: filter_per_lines.append([x2,y2,x1,y1])
    cnt+=1


filter_per_lines = np.array(filter_per_lines)

filter_per_lines=filter_per_lines[filter_per_lines[:,2].argsort()]
line1 = filter_per_lines[0]
midy1 =  int((line1[1]+line1[3])/4 +line1[3]/2)
midx1 = int((line1[2])/2)
midx2 = int(midx1)
midy2 = int(line1[3])
cv2.line(per_im, (midx1,midy1), (midx2,midy2), (255, 0, 0), 2)
mid_lines = []
mid_lines.append([midx1,midy1,midx2,midy2])
for i in [0,1,4,5,6]:
  line1 = filter_per_lines[i]
  line2 = filter_per_lines[i+1]
  midy1 =  int((line1[1]+line1[3])/4 +line1[3]/2)
  midx1 = int((line1[2]+line2[2])/2)
  midx2 = int(midx1)
  midy2 = int(max(line1[3],line2[3]))
  mid_lines.append([midx1,midy1,midx2,midy2])
  cv2.line(per_im, (midx1,midy1), (midx2,midy2), (255, 0, 0), 2)

plt.imshow(per_im)
plt.show()
data = Int32MultiArray()
mid_lines = np.array(mid_lines)
data.data = mid_lines
print(mid_lines)
while not rospy.is_shutdown():
   lanePub.publish(data)
   rate.sleep()
