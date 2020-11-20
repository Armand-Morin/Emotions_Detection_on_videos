import cv2
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array

# Il y a deux fichiers volumineux a telecharger pour les poids
# https://drive.google.com/file/d/1YCox_4kJ-BYeXq27uUbasu--yz28zUMV/view?usp=sharing
# https://drive.google.com/file/d/1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk/view?usp=sharing
# Convertion de l'image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Genre = {'Homme': 0, 'Femme': 0}
Age = []
nb = 0


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


def moy(L):
    i = 0
    for x in L:
        i += int(x)
    return i / len(L)


age_model = ageModel()
gender_model = genderModel()

# Pour determiner l'age d'une personne on vas suposer qu'elle a au plus 101.
# Car sinon il y a trop de sorties
# On calcule l'age en faisonat une moyenne de toutes les sorties ponderees par la proba de cette reponse
output_indexes = np.array([i for i in range(0, 101)])
# On capture une video en entree
cap = cv2.VideoCapture('../BA.mp4')

while True:
    ret, img = cap.read()
    # On detecte une face
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x, y, w, h) in faces:
        if w > 130:  # on ignore les petites faces qui sont des erreurs
            # On dessine un rectangle sur l'image principale
            cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 128), 1)  # draw rectangle to main image

            # On extrait les faces detectées
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face

            try:
                # vgg-face expects inputs (224, 224, 3)
                detected_face = cv2.resize(detected_face, (224, 224))

                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                # prediction de l'age
                age_distributions = age_model.predict(img_pixels)
                apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis=1))[0]))

                # prediction du genre
                gender_distribution = gender_model.predict(img_pixels)[0]
                gender_index = np.argmax(gender_distribution)

                if gender_index == 0:
                    gender = "Femme"
                    Genre['Femme'] += 1
                else:
                    gender = "Homme"
                    Genre['Homme'] += 1
                nb += 1
                print('The presence of women in the film is', (Genre['Femme'] / nb) * 100, '%')
                print('The presence of men in the film is', (Genre['Homme'] / nb) * 100, '%')
                Age.append(apparent_age)
                print(Age)
                print(moy(Age))
                # On affiche les resultats sur les images
                cv2.putText(img, str(apparent_age), (x + 2 * int(w / 2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255),
                            2)
                cv2.putText(img, str(gender), (x + int(w / 2) - 42, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

            except Exception as e:
                print("exception", str(e))

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
