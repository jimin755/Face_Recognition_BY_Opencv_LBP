import cv2
import numpy as np
from os import makedirs
from os.path import isdir
import os
import module2 as mo2
import time

# faces 폴더를 자동 생성되게 해놓았고 실제로 작동 되는지 확인 하였으나 
# 만약 안될시 faces 폴더를 실행 파일 있는 곳에 넣어주세요.

# module1이 메인입니다. module1을 실행 시켜주세요 
# OSS 사용하지 않고 직접 제작한 코드 입니다.


# 얼굴 저장 함수
face_dirs = 'faces/'

#face_classifier = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml') # 정면 인식 Haa`r
face_classifier = cv2.CascadeClassifier('C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface_improved.xml') # LBP

side_face_classifier = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_profileface.xml') # 측면 인식 Haar
#side_face_classifier = cv2.CascadeClassifier('C:/opencv/sources/data/lbpcascades/lbpcascade_profileface.xml') # LBP

def createfolder(dir):    # faces 폴더에 DataSet이 저장되는데 해당 폴더가 없을시 생성해주는 함수 
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error Should Make Creating fold in'+ str(dir))

# 정면 얼굴 검출 함수
def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,3)
    # 얼굴이 없으면 패스!
    if faces is():
        return None
    # 얼굴이 있으면 얼굴 부위만 이미지로 만들고
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    # 리턴!
    return cropped_face

def side_face_extractor(img):#측면 얼굴 인식 함수
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    side_faces = side_face_classifier.detectMultiScale(gray,1.3,3)
    # 얼굴이 없으면 패스!
    if side_faces is():
        return None
    # 얼굴이 있으면 얼굴 부위만 이미지로 만들고
    for(x,y,w,h) in side_faces:
        cropped_face = img[y:y+h, x:x+w]
    # 리턴!
    return cropped_face

# 얼굴만 저장하는 함수
def take_pictures(name):
    # 해당 이름의 폴더가 없다면 생성
    if not isdir(face_dirs+name):
        makedirs(face_dirs+name)

    # 카메라 ON    
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        # 카메라로 부터 사진 한장 읽어 오기
        ret, frame = cap.read()
        # haar의 얼굴인식은 어느정도 밝기의 영향을 받는다 
        # 따라서 너무 어두울 경우 인식에 영향을 주기 때문에 

        s =  "Turn your head to the left and right to recognize your face"

        if face_extractor(frame) is not None:  #정면 얼굴 탐지
            count+=1
            # 200 x 200 사이즈로 줄이거나 늘린다음
            face = cv2.resize(face_extractor(frame),(200,200))
            # 흑백으로 바꿈
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 200x200 흑백 사진을 faces/얼굴 이름/userxx.jpg 로 저장
            file_name_path = face_dirs + name + '/user'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(180,180),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        elif side_face_extractor(frame) is not None: #정면얼굴이 없다면 측면 얼굴
            count+=1
            # 200 x 200 사이즈로 줄이거나 늘린다음
            face = cv2.resize(side_face_extractor(frame),(200,200))
            # 흑백으로 바꿈
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 200x200 흑백 사진을 faces/얼굴 이름/userxx.jpg 로 저장
            file_name_path = face_dirs + name + '/user'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass
        
        cv2.putText(frame,s, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("face",frame)
        # 얼굴 사진 700장을 다 얻었거나 enter키 누르면 종료
        if cv2.waitKey(1)==13 or count==700:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Colleting Samples Complete!!!')


if __name__ == "__main__":
    # 사진 저장할 이름을 넣어서 함수 호출
    createfolder("./faces")
    # faces 폴더에 DataSet이 저장되는데 해당 폴더가 없을시 생성해주는 함수 
    pre_train = input("Dataset을 등록하시겠습니까 (y/n): ")

    if pre_train=='y':
        num = input("인식 시킬 대상의 수를 입력해주세요: ")
        name_list = []
        for i in range(int(num)):
             name = input("인식 대상의 이름을 입력하세요: ")
             take_pictures(name)
             name_list.append(name)
    elif pre_train=='n':
        print("기존의 데이터로 학습을 시작합니다.")
    else:
        print("잘못 입력하셨습니다. dataset 생성 단계부터 진행 하겠습니다.")
        num = input("인식 시킬 대상의 수를 입력해주세요: ")
        name_list = []
        for i in range(int(num)):
             name = input("인식 대상의 이름을 입력하세요: ")
             take_pictures(name)
             name_list.append(name)

    start_time = time.time()
    models = mo2.trains()
    end_time = time.time()-start_time
    m = int(end_time/60)
    s = int(end_time%60)
    print("running_time:{}m{}s".format(m,s))
    mo2.run(models)

