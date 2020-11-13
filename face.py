#Copyright Anirban Kar (anirbankar21@gmail.com)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import cv2
import os

path = os.path.dirname(os.path.abspath(__file__))

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier(path+r'\Classifiers\face.xml')

Id=input('enter your id: ')
sampleNum=0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        sampleNum=sampleNum+1
        cv2.imwrite("dataSet/user"+Id+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif sampleNum>20:
        break
cv2.destroyAllWindows()