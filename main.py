import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Charger les données
df = pd.read_csv("sport.csv")

# Séparer les variables
X = df[["age", "sommeil", "ecran", "energie"]]  # Entrées
y = df["sport"]                                 # Sortie

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer et entraîner le modèle
ia = LogisticRegression()
ia.fit(xtrain, ytrain)

def demander_entier(user_value):
    "Demande un entier à l'utilisateur (ex : âge, heures)"
    while True:
        valeur = input(user_value)
        if valeur.isdigit():
            return int(valeur)
        print("❌ Entrée invalide. Merci d'entrer un nombre entier.")

def demander_energie(user_value):
    "Demande un entier entre 0 et 10 pour le niveau d'énergie"
    while True:
        valeur = input(user_value)
        if valeur.isdigit():
            valeur = int(valeur)
            if 0 <= valeur <= 10:
                return valeur
            else:
                print("Merci d’entrer une valeur entre 0 et 10.")
        else:
            print("Entrée invalide. Merci d'entrer un nombre entier.")
            
def predire_sport():

    age = demander_entier("Quel est ton âge ? ")
    sommeil = demander_entier("Combien d'heures de sommeil par nuit ? ")
    ecran = demander_entier("Combien d'heures d'écran par jour ? ")
    energie = demander_energie("Sur 10, à combien évalues-tu ton énergie ? ")

    new_data = pd.DataFrame([[age, sommeil, ecran, energie]])

    prediction = ia.predict(new_data)

    if prediction[0] == 1:
        print("Tu fais du sport ! Continue comme ça")
    else:
        print("Tu ne sembles pas faire de sport. Il n’est jamais trop tard 😉")

if __name__ = "__main__":
    predire_sport()
