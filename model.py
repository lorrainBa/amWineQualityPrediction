import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=11, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Utilisation de 'sigmoid' pour garantir des prédictions entre 0 et 1
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model