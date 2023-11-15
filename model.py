import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(64, input_dim=11, activation='relu'))
model.add(Dropout(0.5))  # Ajout de la couche de dropout
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))  # Ajout de la couche de dropout
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# Affichage de la structure du modèle
model.summary()