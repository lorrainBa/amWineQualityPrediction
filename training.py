import pandas as pd
from sklearn.model_selection import train_test_split
from model import create_model

import matplotlib.pyplot as plt

#Charger le CSV
df = pd.read_csv('Wines.csv')

# Diviser les données en caractéristiques (X) et la variable cible (y)
X = df.drop(['quality', 'Id'], axis=1)  # Exclure 'quality' et 'Id' des caractéristiques
y = df['quality'] /10 

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Afficher les dimensions des ensembles d'entraînement et de test
print("Dimensions de l'ensemble d'entraînement (X_train, y_train):", X_train.shape, y_train.shape)
print("Dimensions de l'ensemble de test (X_test, y_test):", X_test.shape, y_test.shape)




# Créer le modèle en utilisant la fonction définie dans model.py
model = create_model()

# Entraîner le modèle avec les données d'entraînement
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

#test



# Sélectionner une ligne particulière (par exemple, la première ligne)
sample_row = df.iloc[[7]]

# Exclure 'Id' de la caractéristique
X_sample = sample_row.drop(['quality', 'Id'], axis=1)

# Utiliser le modèle pour faire une prédiction
prediction = model.predict(X_sample)

# Afficher la prédiction
print("Prediction:", prediction)





# Afficher la courbe de la loss pendant l'entraînement
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error Loss')
plt.legend()
plt.show()


