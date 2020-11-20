import pandas as pd
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils

# On utilise panda pour lire les données du ficher.csv
# Ce fichier contient les caracteristiques de la face du visage sous forme de chiffres et le label
# de l'emotion associée
df = pd.read_csv('fer2013.csv')

X_train, train_y, X_test, test_y = [], [], [], []

# En fait dans le fichier fer2013.csv il y des données d'entrainement et de test donc il faut les
# separées dans des listes. C'est ce que fait cette boucle for.
for index, row in df.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val, 'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val, 'float32'))
            test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

num_features = 64
# ensortie de notre reseau de neurone, on veut que pour chaques images rentrée en entrée on puisse predire
# une des 7 emotions considerées. On peut prendre plus de 7 emotions en compte mais c'est plus delicat
num_labels = 7
batch_size = 64
# C'est le nombre d'iteration du processus. Plus on itere plus le parametre de precision acc pour accuracy
epochs = 10
width, height = 48, 48

X_train = np.array(X_train, 'float32')
train_y = np.array(train_y, 'float32')
X_test = np.array(X_test, 'float32')
test_y = np.array(test_y, 'float32')

train_y = np_utils.to_categorical(train_y, num_classes=num_labels)
test_y = np_utils.to_categorical(test_y, num_classes=num_labels)

# On normalise les vecteurs entre 0 et 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# print(f"shape:{X_train.shape}")
## Maintenant on construit le reseau neuronal convolutif (CNN)

# 1ère couche
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

# 2eme couche de convolution
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

# 3eme couche de convolution
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

# On connete le reseau
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))

# model.summary()

# On compile le modèle
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])


# On entraine le modèle
model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, test_y),
          shuffle=True)

# On enregistre les poids des parametres entre chaque couches de neurones puis on vat
# recharger le fichier pour l'utiliser lors de la detection
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
