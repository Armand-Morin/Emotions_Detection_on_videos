import cv2
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.models import model_from_json

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Convertion de l'image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# On crée differents modeles pour chaques parametres que l'on souhaite predire
# Cette fonction crée un model vgg face
def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    return model


# Pour determiner l'age de la personne, on utilise le modele vgg face
def ageModel():
    model = loadVggFaceModel()
    base_model_output = Sequential()
    base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    age_model = Model(inputs=model.input, outputs=base_model_output)
    # On charge les poids dans notre reseau de neurone
    # On obtient donc un reseau deja entrainné car on n'avait pas le temps de trouver
    # une base de donée pour entrainner le reseau dessus
    age_model.load_weights("age_model_weights.h5")

    return age_model


# Pour determiner le genre de la personne , on utilise aussi un modele vgg face
def genderModel():
    model = loadVggFaceModel()
    base_model_output = Sequential()
    base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    gender_model = Model(inputs=model.input, outputs=base_model_output)
    # pareil les poids sont directement chargé
    gender_model.load_weights("gender_model_weights.h5")

    return gender_model


def emotionModel():
    emotion_model = model_from_json(open("fer.json", "r").read())
    # Ici le reseau je l'ai entrainné avec le code training-program qui retourne un fichier fer.h5
    # qui contient les poids ne chaques neurones
    emotion_model.load_weights('fer.h5')
    return emotion_model


age_model = ageModel()
gender_model = genderModel()
emotion_model = emotionModel()

# Pour determiner l'age d'une personne on vas suposer qu'elle a au plus 101.
# Car sinon il y a trop de sorties
# On calcule l'age en faisonat une moyenne de toutes les sorties ponderees par la proba de cette reponse
output_indexes = np.array([i for i in range(0, 101)])
# On capture une video en entree
cap = cv2.VideoCapture('../BA.mov')
# On cree des listes dans lesquelles on stocke le contenu de chaques frame.
Genres = []
Ages = []
Emotions = []
while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # On detecte une face
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x, y, w, h) in faces:
        if w > 130:  # on ignore les petites faces qui sont des erreurs
            # On dessine un rectangle sur l'image principale
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 128, 128), thickness=7)
            # On extrait les faces detectées
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # On les decoupes

            try:
                # vgg-face expects inputs (224, 224, 3)
                roi_gray = gray_img[y:y + w, x:x + h]
                detected_face = cv2.resize(detected_face, (224, 224))
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixel = image.img_to_array(roi_gray)
                img_pixel = np.expand_dims(img_pixel, axis=0)
                img_pixel /= 255
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                # On normalise
                img_pixels /= 255

                # prediction de l'age
                age_distributions = age_model.predict(img_pixels)
                apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis=1))[0]))

                # prediction du genre
                gender_distribution = gender_model.predict(img_pixels)[0]
                gender_index = np.argmax(gender_distribution)

                # prediction de l'emotion
                emotion_predictions = emotion_model.predict(img_pixel)
                max_index = np.argmax(emotion_predictions[0])

                if gender_index == 0:
                    gender = "F"  # On affiche F au dessus de la tete d'une femme
                else:
                    gender = "H"
                # C'est la liste des emotions que l'on vas essayer de predire
                # On peut en mettre plus mais il faut tout re entrainer
                emotions = ('angry', 'disgust', 'happy', 'fear', 'sad', 'surprise', 'neutral')
                # L'emotion predite est celle qui est la plus probable
                predicted_emotion = emotions[max_index]

                # On affiche les resultats sur les images
                cv2.putText(img, str(apparent_age), (x + int(w / 2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 111, 255), 2)
                cv2.putText(img, str(gender), (x + int(w / 2) + 100, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 111, 255),
                            2)
                cv2.putText(img, str(predicted_emotion), (x + int(w / 2) - 120, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 111, 255), 2)
                # Dans chaque listes on rajoute le nom de l'emotion detectéé, le genre de la personne, et son age
                Genres.append('gender')
                Ages.append('apparent_age')
                Emotions.append('predicted_emotion')
            except Exception as e:
                print("exception", str(e))

    cv2.imshow('Analyse du film en cours : .....', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break
print(Genres)
print(Ages)
print(Emotions)
cap.release()
cv2.destroyAllWindows()
