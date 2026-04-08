


from xml.parsers.expat import model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



df = pd.read_csv('/content/gld_price_data.csv')
df.head()

df.describe()

df.info()

df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.info()

df.isnull().sum()

sns.set(style="whitegrid")

# Ajuster la taille de la figure pour une meilleure lisibilité
plt.figure(figsize=(10, 6))

# Créer un boxplot pour chaque variable numérique
ax = sns.boxplot(data=df, orient="v")

# Utiliser une échelle logarithmique pour les valeurs en y (pour gérer les écarts de valeurs comme le loyer)
ax.set_yscale('log')

# Améliorer la mise en page
plt.xticks(rotation=45)
plt.tight_layout()

# Afficher le graphique
plt.show()

sns.set(style="ticks")
sns.pairplot(df)

correlations = df.select_dtypes(include=np.number).corr(method='pearson')
# Spécifie que nous utilisons la méthode de Pearson pour calculer les corrélations.
f, ax = plt.subplots(figsize = (5, 5))
# Crée une figure (f) et des axes (ax) avec une taille de 5x5 pouces.
sns.heatmap(correlations, annot = True)
#Utilise la bibliothèque Seaborn pour créer une carte thermique (heatmap) de la matrice de corrélation. L'argument annot=True affiche les valeurs des coefficients de corrélation dans chaque cellule de la heatmap.

df['GLD'].describe()

sns.distplot(df['GLD'],color='green')

X = df.drop(['Date','GLD'],axis=1)
Y = df['GLD']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.2, random_state=2)

from sklearn.linear_model import LinearRegression
profit_model = LinearRegression()

# Entraînement du modèle
profit_model.fit(X_train, Y_train)

# Prédictions
y_train_pred = profit_model.predict(X_train)
y_test_pred = profit_model.predict(X_test)
print("\n=== ÉVALUATION DU MODÈLE DE RÉGRESSION LINÉAIRE MULTIPLE ===")
print("\nCoefficients du modèle:")
for feature, coef in zip(X, profit_model.coef_):
    print(f"  {feature}: {coef:.2f}")

print(f"\nIntercept: {profit_model.intercept_:.2f}")

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
mse_train = mean_squared_error(Y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(Y_train, y_train_pred)
r2_train = r2_score(Y_train, y_train_pred)

print(f"Erreur quadratique moyenne (MSE): {mse_train:.2f}")
print(f"Racine de l'erreur quadratique moyenne (RMSE): {rmse_train:.2f}")
print(f"Erreur absolue moyenne (MAE): {mae_train:.2f}")
print(f"Coefficient de détermination (R²): {r2_train:.4f}")

print("\n--- Performances sur l'ensemble de test ---")
mse_test = mean_squared_error(Y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(Y_test, y_test_pred)
r2_test = r2_score(Y_test, y_test_pred)

print(f"Erreur quadratique moyenne (MSE): {mse_test:.2f}")
print(f"Racine de l'erreur quadratique moyenne (RMSE): {rmse_test:.2f}")
print(f"Erreur absolue moyenne (MAE): {mae_test:.2f}")
print(f"Coefficient de détermination (R²): {r2_test:.4f}")



