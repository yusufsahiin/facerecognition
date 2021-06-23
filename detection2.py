import cv2
import numpy as np
from PIL import Image #pillow kütüphanesi
import os
#Yusuf Şahin 170310062 ÖÖ
#Yüz resimlerinin datasına erişmemiz lazım
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Görüntüleri almamız ve verileri etiketlememiz için kullanacağımız fonksiyon
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') #grayscale dönüştürüyoruz
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return  faceSamples,ids

print("\n [BİLGİ] Yüz işlemleri yapılıyor. Lütfen bekleyiniz...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

#Save the  model into trainer/trainer
recognizer.save('trainer/trainer.yml') #recognizer.write()  pi'da çalışmaktadır, PC'de değil

#En sonunda kaç tane tanımlanmış yüz bulunduğunu yaz
print("\n [BİLGİ] {0} yüz bulundu. Programdan çıkılıyor".format(len(np.unique(ids))))
