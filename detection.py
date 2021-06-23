import cv2
import os
#Yusuf Şahin 170310062 ÖÖ
cam = cv2.VideoCapture(0)

cam.set(3,640) #Kameranın genişliği (weight)
cam.set(4,480) #Kameranın uzunluğu (height)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Her bir kişi için bir yüz id değeri giriniz
face_id = input('\n Bir id giriniz= ')

print("\n [BİLGİ] Lütfen kameraya bakınız ve bekleyiniz...")
#Bireysel örnekleme yüz sayısını başlat
count = 0

while(True):
    ret, img = cam.read()
    img = cv2.flip(img, +1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1

        #Yakalanmış resimleri kaydet
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff #ESC basarak videodan çıkabilirsiniz
    if k ==27:
        break
    elif count >= 40: #80 yüz görüntüsü al ve videoyu durdur. Önceden değerini 80 yapmıştım ama 40 değerine düşürdüm.
        break

print("\n [BİLGİ] Lütfen bekleyiniz...")
cam.release()
cv2.destroyAllWindows()
