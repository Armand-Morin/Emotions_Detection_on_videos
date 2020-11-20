import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# On charge le modele de reconnaisance d'une face dans une image
model = model_from_json(open("fer.json", "r").read())

# On charge les poids dans les couches du reseau
model.load_weights('fer.h5')

# C'est le module de reconnaisance d'une face dans une image
face_haar_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

# on indique la video que l'on veut lire
cap = cv2.VideoCapture('../BA.mp4')

# On crer un dictionnaire dan lesquelle on compte le nombre de frames qui correspondent a l'emotion
Emotions = {}
Emotions['Happy']=0
Emotions['Surprise']=0
Emotions['Sad']=0
Emotions['Neutral']=0
Emotions['Angry']=0
nb=0

# c'est une boucle qui est toujours vérifiée
# On peut aussi mettre while(cap.isOpenned()): pour dire tant que la video dure.
while True:
    # On analyse l'image frame par frame en lisant la video
    ret, test_img = cap.read()  # On capture des frames
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # On detecte une face sur l'image
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        # On trace le rectangle sur la frame
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

        # ROI c'est la region of interest = la tete ou face dans notre cas qui est grise
        # on coupe les images pour ne garder que la tete
        roi_gray = gray_img[y:y + w, x:x + h]

        # On redimentionne la face que l'on obtient en utilisant une interpolation qui est pas necessaire
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        # On fait une prediction de l'emotion sur la face, ensuite on regarde la classe
        predictions = model.predict(img_pixels)

        # on garde l'emotion la plus probable parmis celles dans la liste
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'happy', 'fear', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        # On ecrit sur la frame l'emotion trouvée
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
        nb+=1
        if predicted_emotion in Emotions:
            Emotions[predicted_emotion] += 1
        print('The film is happy at' ,(Emotions['Happy']/nb)*100,'%')
        print('The film is surprising at',(Emotions['Surprise']/nb)*100,'%')
        print('The film is sad at',(Emotions['Sad']/nb)*100,'%')
        print('The film is neutral at', (Emotions['Neutral']/nb)*100,'%')
        print('The film is angry at', (Emotions['Angry']/nb)*100,'%')

    # On affiche l'image en sortie
    cv2.imshow('Facial emotion analysis ', test_img)

    # Si on clique sur la touche q on exit le programme
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
