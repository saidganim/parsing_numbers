import os  
from selenium import webdriver  
from selenium.webdriver.common.keys import Keys  
from selenium.webdriver.chrome.options import Options

import base64
from PIL import Image
from io import BytesIO

import sys
import numpy as np
import cv2

# ================================================ PARSING PART
chrome_options = Options()  
chrome_options.add_argument("--headless")  

driver = webdriver.Chrome(executable_path=os.path.abspath("/usr/bin/chromedriver"),  
 chrome_options=chrome_options)  
# print 'label'
print 'Stealing number from link ' + sys.argv[1]
driver.get(sys.argv[1])
# print driver.page_source.encode("utf-8")
toclick = driver.find_element_by_class_name("js-item-phone-number")
toclick.click()
img = driver.find_element_by_class_name("js-item-phone-big-number")
img = img.find_element_by_css_selector('img')
img = img.get_attribute('src')
# print img[22:]
# im = Image.open(BytesIO(base64.b64decode(img[22:])))
im = base64.b64decode(img[22:])
filename = 'number.png'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
        f.write(im)
im = Image.open("number.png")
rgb_im = im.convert('RGB')
rgb_im.save('number.jpg')
driver.close()



# ===== TRAINING PART - I have attached ready file, please don't run this code again :) 
# im = cv2.imread('training.jpg')
# im3 = im.copy()

# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(5,5),0)
# thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

# #################      Now finding Contours         ###################

# _, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# samples =  np.empty((0,100))
# responses = []
# keys = [i for i in range(48,58)]

# for cnt in contours:
#     if cv2.contourArea(cnt)>50:
#         [x,y,w,h] = cv2.boundingRect(cnt)

#         if  h>28:
#             cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
#             roi = thresh[y:y+h,x:x+w]
#             roismall = cv2.resize(roi,(10,10))
#             cv2.imshow('norm',im)
#             key = cv2.waitKey(0)

#             if key == 27:  # (escape to quit)
#                 sys.exit()
#             elif key in keys:
#                 responses.append(int(chr(key)))
#                 sample = roismall.reshape((1,100))
#                 samples = np.append(samples,sample,0)

# responses = np.array(responses,np.float32)
# responses = responses.reshape((responses.size,1))
# print "training complete"

# np.savetxt('generalsamples.data',samples)
# np.savetxt('generalresponses.data',responses)

samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE, responses)

############################# testing part  #########################

im = cv2.imread('number.jpg')
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

_, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
result = ''
for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
            string = str(int((results[0][0])))
            result = string + result
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
print result
# cv2.imshow('im',im)
# cv2.imshow('out',out)
# cv2.waitKey(0)