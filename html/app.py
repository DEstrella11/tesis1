from flask import Flask, render_template, request
from feature_extraction import FeatureExtraction
import joblib
import pandas as pd
import os

app = Flask(__name__)
model = joblib.load("C:/Users/SKY/Documents/teisahjdfks/Modelo_entrenado.pkl")  # Cargar el modelo entrenado
phishing_links = []  # Lista para almacenar los enlaces phishing
excel_file = 'phishing_links.xlsx'  # Nombre del archivo de Excel

# Crear el archivo de Excel si no existe
if not os.path.exists(excel_file):
    df = pd.DataFrame({'Enlaces phishing': []})
    df.to_excel(excel_file, index=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        features = FeatureExtraction(url).features
        prediction = model.predict([features])[0]  # Realizar la predicción con el modelo

        if prediction == -1:
            result = "El enlace es phishing."
            phishing_links.append(url)  # Agregar el enlace phishing a la lista

            # Guardar la lista en el archivo de Excel
            df = pd.DataFrame({'Enlaces phishing': phishing_links})
            df.to_excel(excel_file, index=False)

        else:
            result = "El enlace no es phishing."

        return render_template('index.html', url=url, result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()
