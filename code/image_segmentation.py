import cv2
import numpy as np
from IPython.display import display
import pandas as pd

pea_on_peg = cv2.VideoCapture("../dataset/X01_Pea_on_a_Peg_01.mp4", cv2.CAP_FFMPEG)

pea_on_peg_csv = pd.read_csv("../dataset/X01_Pea_on_a_Peg_01.csv")

display(pea_on_peg_csv[["PSM1_position_x", "PSM1_position_y", "PSM1_position_z"]])
display(pea_on_peg_csv[["PSM2_position_x", "PSM2_position_y", "PSM2_position_z"]])


# Get total count of frames
count = pea_on_peg.get(cv2.CAP_PROP_FRAME_COUNT)

# Move the seeker to the last frame
pea_on_peg.set(cv2.CAP_PROP_POS_FRAMES, count - 1)

# Utility functions to get the hsv measurements of BGR colors format
def get_colors():

    colors = []

    # Alternatively, find color in BGR
    pink = np.uint8([[[173, 81, 242]]])
    hsv_pink = cv2.cvtColor(pink, cv2.COLOR_BGR2HSV)

    lower_limit = hsv_pink[0][0][0] - 10, 100, 100
    upper_limit = hsv_pink[0][0][0] + 10, 255, 255

    lower_limit = np.asarray(lower_limit)
    upper_limit = np.asarray(upper_limit)

    colors.append((lower_limit, upper_limit))


    purple = np.uint8([[[108, 53, 81]]])
    hsv_purple = cv2.cvtColor(purple, cv2.COLOR_BGR2HSV)

    lower_limit = hsv_purple[0][0][0] - 10, 100, 100
    upper_limit = hsv_purple[0][0][0] + 10, 255, 255

    lower_limit = np.asarray(lower_limit)
    upper_limit = np.asarray(upper_limit)

    colors.append((lower_limit, upper_limit))


    blue = np.uint8([[[201, 152, 75]]])
    hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)

    lower_limit = hsv_blue[0][0][0] - 10, 100, 100
    upper_limit = hsv_blue[0][0][0] + 10, 255, 255

    lower_limit = np.asarray(lower_limit)
    upper_limit = np.asarray(upper_limit)

    colors.append((lower_limit, upper_limit))


    yellow = np.uint8([[[75, 237, 249]]])
    hsv_yellow = cv2.cvtColor(yellow, cv2.COLOR_BGR2HSV)

    lower_limit = hsv_yellow[0][0][0] - 10, 100, 100
    upper_limit = hsv_yellow[0][0][0] + 10, 255, 255

    lower_limit = np.asarray(lower_limit)
    upper_limit = np.asarray(upper_limit)

    colors.append((lower_limit, upper_limit))


    orange = np.uint8([[[78, 122, 237]]])
    hsv_orange = cv2.cvtColor(orange, cv2.COLOR_BGR2HSV)

    lower_limit = hsv_orange[0][0][0] - 10, 100, 100
    upper_limit = hsv_orange[0][0][0] + 10, 255, 255

    lower_limit = np.asarray(lower_limit)
    upper_limit = np.asarray(upper_limit)

    colors.append((lower_limit, upper_limit))

    return colors

    
'''
    Colors in hsv format:
    [
        pink, purple, blue, yellow, orange
    ]
''' 
colors = get_colors()

# Dictionary to display colors on screen
dict_colors = {}

dict_colors[0] = "Pink"
dict_colors[1] = "Purple"
dict_colors[2] = "Blue"
dict_colors[3] = "Yellow"
dict_colors[4] = "Orange"

# Kernels used to filter the selection
kernelOpen=np.ones((2,2))
kernelClose=np.ones((4,4))
# Font to display text
font = cv2.FONT_HERSHEY_SIMPLEX

# Empty list to store tuples
#   (x, y, r)
# respectively the center coordinates (x,y) and the radius r of the bounding circles around 
# the identified peas
list_positions = []

while (pea_on_peg.isOpened()):
    # Capture frame by frame (we have only one anyway)
    ret, img = pea_on_peg.read()

    if ret == True:

        # Resize image for faster processing
        img = cv2.resize(img, (340, 220))

        img = img[20:200, :, :]

        # Convert in hsv format
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for ind in range(len(colors)):
 
            mask = cv2.inRange(img_hsv, colors[ind][0], colors[ind][1])

            maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
            maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

            maskFinal = maskClose
            conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

            cv2.drawContours(img,conts,-1,(255,0,0),3)

            for i in range(len(conts)):
                (x,y), r = cv2.minEnclosingCircle(conts[i])
                print("Positions of peas:\n")
                list_positions.append((x,y,r))
                print(f"x: {x}; y: {y}; r: {r}\n")
                cv2.circle(img,(int(x),int(y)),int(r),(0,0,255), 2)
                cv2.putText(img, dict_colors[ind],(int(x),int(y)),font,0.25,(0,255,255),1)
            
            # output = cv2.bitwise_and(img_hsv, img_hsv, mask = mask)

            # cv2.imshow("maskClose",maskClose)
            # cv2.imshow("maskOpen",maskOpen)
            cv2.namedWindow('cam',cv2.WINDOW_NORMAL)
            # cv2.imshow("mask",mask)
            cv2.imshow("cam",img)

            cv2.waitKey(1)

    else:
        # This is the final image with the relative bounding boxes: at this point I can obtain the (x,y) coordinate of 
        # the beds w.r.t. to the image
        break


print(list_positions)

cv2.waitKey(0)

pea_on_peg.release()
cv2.destroyAllWindows()


