from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# C'est le module de reconnaisance d'une face dans une image
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# C'est le module de detection d'émotion qui résulte de l'entrainement sur une base de données
classifier = load_model('Emotion_Detection.h5')

# On vat essayer de classer les frames dans les différentes categories suivantes (on peut en rajouter d'autres)
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# on indique la video que l'on veut lire
cap = cv2.VideoCapture('LA.mov')

Emotions = {}
Emotions['Happy']=0
Emotions['Surprise']=0
Emotions['Sad']=0
Emotions['Neutral']=0
Emotions['Angry']=0
nb=0
# c'est une boucle qui est toujours vérifiée
# On peut aussi mettre while(cap.isOpenned()): pour dire tant que la video dure.
while cap.isOpened():
    # On analyse l'image frame par frame en lisant la video
    ret, frame = cap.read()
    # C'est la liste dans laquelle on stocke toutes les empotions qui vont etres rencontées dans notre vidéo
    # Pour chaques nouvelles frames on rajoute une ligne
    labels = []
    # on convertit les images en gris pour l'analyse
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # On detecte une face sur l'image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # On trace le rectangle sur la frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # ROI c'est la region of interest = la tete ou face dans notre cas qui est grise
        roi_gray = gray[y:y + h, x:x + w]

        # On redimentionne la face que l'on obtient en utilisant une interpolation qui est pas necessaire
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # La condition if s'assure que la face que l'on etudie n'est pas composée que de 0
        # (si la somme est nulle c'est que toutes les valeurs sont nulles car il n'y a que des valeurs positives dans la matrice)
        if np.sum([roi_gray]) != 0:

            # On predit l'emotion par la methode des plus proches voisins
            # L'emotion sur la frame que l'on annalyse est comparée a celles qui sont connues et deja identifiées dans le fichier emotion_detection

            # Pour faire les predictions il faut utiliser des flotants entre 0 et 1 donc on divise par  255
            roi = roi_gray.astype('float') / 255.0

            # On convertit l'image en une matrice
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # On fait une prediction de l'emotion sur la face, ensuite on regarde la classe
            preds = classifier.predict(roi)[0]

            # On affiche la prediction
            label = class_labels[preds.argmax()]
            print("label = ", label)
            # On choisit de positionner le text en (x,y) sur la video
            label_position = (x, y)
            # On ecrit sur la frame l'emotion trouvée
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            nb+=1
            if label in Emotions:
                Emotions[label] += 1
            print('The film is happy at' ,(Emotions['Happy']/nb)*100,'%')
            print('The film is surprising at',(Emotions['Surprise']/nb)*100,'%')
            print('The film is sad at',(Emotions['Sad']/nb)*100,'%')
            print('The film is neutral at', (Emotions['Neutral']/nb)*100,'%')
            print('The film is angry at', (Emotions['Angry']/nb)*100,'%')
        else:
            # Si on ne trouve pas de frame sur la frame on affiche No face found sur l'image
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # On affiche l'image en sortie
    cv2.imshow('Emotion Detector', frame)

    # Si on clique sur la touche q on exit le programme
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
