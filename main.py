import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Charger les données
df = pd.read_csv("sante.csv")

# Séparer les variables
X = df[["age", "sommeil", "ecran", "energie"]]  # Entrées
y = df["sport"]                                 # Sortie

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer et entraîner le modèle
ia = LogisticRegression()
ia.fit(xtrain, ytrain)

def predire_sport(ia):
    try:
        age = int(input("Quel est ton âge ? "))
        sommeil = int(input("Combien d'heures de sommeil par nuit ? "))
        ecran = int(input("Combien d'heures d'écran par jour ? "))
        energie = int(input("Sur 10, à combien évalues-tu ton énergie ? "))
    except ValueError:
        print("Entrée invalide. Merci d'entrer uniquement des nombres.")
        return

    new_data = np.array([[age, sommeil, ecran, energie]])
    prediction = ia.predict(new_data)

    if prediction[0] == 1:
        print("Tu fais du sport ! Continue comme ça")
    else:
        print("😕 Tu ne sembles pas faire de sport. Il n'est jamais trop tard pour s'y mettre 😉")

predire_sport(ia)