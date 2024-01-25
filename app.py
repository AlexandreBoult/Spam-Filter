import streamlit as st
import pandas as pd
# Importez vos fonctions de main.py
from test2 import make_prediction, accuracy, classification

# Chargement des données
df = pd.read_csv("SMSSpamCollection", header=None, sep='\t', names=['cat', 'msg'])

st.title("Détection de Spam")

# Champ de saisie pour l'utilisateur
user_input = st.text_area("Entrez le texte à analyser")

if st.button("Prédire"):
    make_prediction(df)
    predic = preduser_input