from deepface import DeepFace

demography = DeepFace.analyze("test.jpg")

print("Yaş: ", demography["age"])
print("Cinsiyet: ", demography["gender"])
print("Duygu: ", demography["dominant_emotion"])
print("Milliyet: ", demography["dominant_race"])