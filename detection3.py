import cv2
import numpy as np
import os
#Yusuf Şahin 170310062 ÖÖ
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml') #trained modeli yükleme işlemi
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['','Yusuf Sahin','Kamil Kaya', 'Bilal Sahin', 'Cemile Bardak']
#isimler ikinci sıradan başlıyor, bu yüzden ilk kısmı boş bırakıyoruz
profession = ['','Ogrenci', 'Makine Muhendisi', 'Muhendis', 'Ogretim Gorevlisi' ]

#Gerçek zamanlı video görüntüsünü açmak için
cam = cv2.VideoCapture(0)
cam.set(3,640) #Kameranın genişliği (weight)
cam.set(4,480) #Kameranın uzunluğu (height)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()

    img = cv2.flip(img, +1) #Dikey döndürme gerçekleştiriyoruz

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize= (int(minW), int(minH))
        )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        #%100'den az güvenli olduğu zaman kontrol edin ==> "0" yüz uyumluluğu olduğunda
        if (confidence < 100):
            id= names[id], profession[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "Tanimlanmamis Yuz"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera', img)

    k =cv2.waitKey(100) & 0xff #ESC basarak videodan çıkabilirsiniz
    if k ==27:
        break

print("\n [BİLGİ] Lütfen Bekleyiniz...")
cam.release()
cv2.destroyAllWindows()
