
# mlapp/views.py
from django.shortcuts import render
from django.http import HttpResponse
from .forms import GoldPredictionForm
import joblib
import numpy as np
import os
from pathlib import Path

# Configuration du chemin du modèle
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH_DEFAULT = BASE_DIR / 'mlapp' / 'model' / 'model.pkl'
MODEL_PATH_ALT = BASE_DIR / 'mlapp' / 'model.pkl'
MODEL_PATH = MODEL_PATH_DEFAULT if MODEL_PATH_DEFAULT.exists() else MODEL_PATH_ALT

# Chargement global du modèle (une seule fois)
try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        model_loaded = True
        print(f"✅ Modèle chargé avec succès depuis : {MODEL_PATH}")
    else:
        model_loaded = False
        print(f"❌ Modèle non trouvé. Cherché: {MODEL_PATH_DEFAULT} ou {MODEL_PATH_ALT}")
except Exception as e:
    model_loaded = False
    print(f"❌ Erreur lors du chargement : {e}")

def predict(request):
    result = None
    error_message = None
    
    if not model_loaded:
        return render(request, 'mlapp/predict.html', {
            'error': f'Modèle non trouvé. Veuillez placer model.pkl dans : {MODEL_PATH}',
            'form': GoldPredictionForm()
        })
    
    if request.method == 'POST':
        form = GoldPredictionForm(request.POST)
        if form.is_valid():
            try:
                # Récupérer les données du formulaire
                # Adaptez selon les caractéristiques de VOTRE modèle
                features = [
                    form.cleaned_data['spx'],
                    form.cleaned_data['uso'],
                    form.cleaned_data['slv'],
                    form.cleaned_data['eur_usd'],
                ]
                
                # Convertir en numpy array pour la prédiction
                features_array = np.array([features])
                
                # Faire la prédiction
                prediction = model.predict(features_array)
                
                # Formater le résultat
                # Adaptez selon que c'est une classification ou régression
                if len(prediction) == 1:
                    if isinstance(prediction[0], (int, float)):
                        result = f"💰 Prix prédit : ${prediction[0]:.2f}"
                    else:
                        result = f"📊 Résultat : {prediction[0]}"
                else:
                    result = f"📈 Prédiction : {prediction}"
                
                print(f"✅ Prédiction effectuée : {result}")  # Debug
                
            except Exception as e:
                error_message = f"Erreur lors de la prédiction : {str(e)}"
                print(f"❌ {error_message}")  # Debug
        else:
            error_message = "Formulaire invalide. Vérifiez vos données."
    else:
        form = GoldPredictionForm()
    
    return render(request, 'mlapp/predict.html', {
    'form': form,
    'result': result,
    'error': error_message,
    'model_loaded': model_loaded,  # Ajoutez ceci
    'model_path': MODEL_PATH,       # Ajoutez ceci
})

# Vue simple pour tester
def home(request):
    return render(request, 'mlapp/predict.html', {
        'form': GoldPredictionForm(),
        'info': "Bienvenue sur l'application de prédiction du prix de l'or"
    })