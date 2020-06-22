import cv2
import numpy as np
from os import makedirs
from os.path import isdir
import time
from matplotlib import pyplot as plt
import time 

face_classifier = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml')
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# 정면 얼굴 검출 함수
def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    # 얼굴이 없으면 패스!
    if faces is():
        return None
    # 얼굴이 있으면 얼굴 부위만 이미지로 만들고
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    # 리턴!
    return cropped_face


# 얼굴만 저장하는 함수
def take_pictures(name):
    # 카메라 ON    
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        # 카메라로 부터 사진 한장 읽어 오기
        ret, frame = cap.read()
        # 사진에서 얼굴 검출 , 얼굴이 검출되었다면 

        cv2.imshow("face",frame)

        if face_extractor(frame) is not None:  
            count+=1
            # 200 x 200 사이즈로 줄이거나 늘린다음
            face = cv2.resize(face_extractor(frame),(200,200))
            return face
        else:
            print("Face not Found")
            pass
        
        # 얼굴 사진 1장을 다 얻었거나 enter키 누르면 종료
        if cv2.waitKey(1)==13 or count==1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Colleting Samples Complete!!!')




if __name__ == "__main__":
    num = input("인식 대상 수를 입력해주세요: ")
    name_list = []
    face_list = []
   
    for i in range(int(num)):
        name = input("인식 대상의 이름을 입력해주세요: ")
        face = take_pictures(name)
        cv2.imshow("face",face)

    cap = cv2.VideoCapture(0)
    
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start = time.time()

        for i in range(1):
            template = face
            res = cv2.matchTemplate(frame,template,cv2.TM_SQDIFF)
            min_val,max_val,min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = min_loc
            bottom_right = (top_left[0]+200,top_left[1]+200)

            cv2.rectangle(frame,top_left,bottom_right,255,1)
        m = int((start - time.time())/60)
        n = int((start - time.time())%60)
        print("M:{} S:{}".format(str(m),str(n)))
        cv2.imshow('show',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
              break

    cap.release()
    cv2.destroyAllWindows()   

        
