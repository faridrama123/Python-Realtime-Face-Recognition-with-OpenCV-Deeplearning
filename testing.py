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

import cv2,os
import numpy as np
from PIL import Image 

path = os.path.dirname(os.path.abspath(__file__))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(path+r'\trainer\trainer.yml')
cascadePath = path+"\Classifiers\face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
detector=cv2.CascadeClassifier(path+r'\Classifiers\face.xml')

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    
    while True:
        ret_val, img = cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(gray, 1.2,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            if(conf<60): # Accuracy Detect. <50 for better
                 if(Id==1):
                    Id="Youngmin"
                 elif(Id==2):
                    Id="Kwangmin"
            else:
                Id=""
            cv2.putText(img,str(Id)+""+str(""), (x,y+h),font, 1.1, (0,255,0)) #Draw the text
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()



