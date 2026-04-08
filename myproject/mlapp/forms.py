# mlapp/forms.py
from django import forms

class GoldPredictionForm(forms.Form):
    # Adaptez ces champs selon les caractéristiques de votre modèle
    # Exemple pour la prédiction du prix de l'or :
    spx = forms.FloatField(label='S&P 500 Price', required=True)
    uso = forms.FloatField(label='USO Price', required=True)
    slv = forms.FloatField(label='Silver Price', required=True)
    eur_usd = forms.FloatField(label='EUR/USD Exchange Rate', required=True)
    
    # Vous pouvez ajouter plus de champs selon votre modèle