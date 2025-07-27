# test_model.py (Corregido)

import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Ignorar las advertencias de versión para una salida más limpia
warnings.filterwarnings("ignore", category=UserWarning)

def evaluate_model():
    """Carga el modelo y lo evalúa contra un conjunto de prueba."""
    print("--- 1. Evaluación del Modelo Existente ---")
    
    # Cargar los datos originales
    df = pd.read_csv('datos_liga_procesados.csv')
    X = df.drop(columns=['Resultado'])
    y_test_text = df['Resultado'] # Mantenemos los nombres originales para el reporte
    
    # Dividir los datos para tener un conjunto de prueba
    _, X_test, _, y_test = train_test_split(X, y_test_text, test_size=0.2, random_state=42, shuffle=False)
    
    # Cargar el modelo y el decodificador de etiquetas
    model = joblib.load('artifacts/random_forest_model.joblib')
    encoder = joblib.load('artifacts/label_encoder.joblib')
    
    # Realizar predicciones (estas serán numéricas: 0, 1, 2)
    y_pred_numeric = model.predict(X_test)
    
    # --- CORRECCIÓN AQUÍ ---
    # Decodificamos las predicciones numéricas a texto (ej: 0 -> 'A')
    y_pred_text = encoder.inverse_transform(y_pred_numeric)
    
    # Calcular y mostrar las métricas usando las etiquetas de texto
    accuracy = accuracy_score(y_test, y_pred_text)
    report = classification_report(y_test, y_pred_text, target_names=encoder.classes_)
    
    print(f"✅ Accuracy del modelo en datos de prueba: {accuracy:.2%}")
    print("\n✅ Reporte de Clasificación:")
    print(report)
    print("-" * 40)


def predict_new_match(sample_data):
    """Carga el modelo y realiza una nueva predicción."""
    print("\n--- 2. Realizando una Nueva Predicción de Prueba ---")
    
    # Cargar artefactos
    model = joblib.load('artifacts/random_forest_model.joblib')
    scaler = joblib.load('artifacts/scaler.joblib')
    encoder = joblib.load('artifacts/label_encoder.joblib')
    with open('artifacts/model_columns.json', 'r') as f:
        model_columns = json.load(f)
        
    # Crear DataFrame a partir de los datos de ejemplo
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0

    # Rellenar los datos del partido
    home_team_col = f"HomeTeam_{sample_data['home_team']}"
    away_team_col = f"AwayTeam_{sample_data['away_team']}"

    if home_team_col in input_df.columns:
        input_df.loc[0, home_team_col] = 1
    if away_team_col in input_df.columns:
        input_df.loc[0, away_team_col] = 1

    input_df.loc[0, 'H_Forma_Goles_Anotados'] = sample_data['h_form_goals']
    input_df.loc[0, 'A_Forma_Goles_Anotados'] = sample_data['a_form_goals']
    input_df.loc[0, 'HST'] = sample_data['h_shots_on_target']
    input_df.loc[0, 'AST'] = sample_data['a_shots_on_target']
    
    # Normalizar los datos
    numeric_cols = scaler.get_feature_names_out()
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Realizar la predicción
    probabilities = model.predict_proba(input_df)[0]
    prediction_numeric = model.predict(input_df)[0]
    
    # --- CORRECCIÓN AQUÍ ---
    # Decodificar la predicción final y las claves del diccionario de probabilidades
    prediction_label = encoder.inverse_transform([prediction_numeric])[0]
    prob_dict = {encoder.inverse_transform([i])[0]: f"{prob:.2%}" for i, prob in enumerate(probabilities)}

    print(f"Partido: {sample_data['home_team']} vs {sample_data['away_team']}")
    print(f"⚽ Predicción Final: {prediction_label}")
    print(f"📊 Probabilidades: {prob_dict}")


if __name__ == "__main__":
    try:
        evaluate_model()
        
        new_match_data = {
          "home_team": "Real Madrid",
          "away_team": "Sevilla",
          "h_form_goals": 2.1,
          "a_form_goals": 1.5,
          "h_shots_on_target": 7.0,
          "a_shots_on_target": 4.5,
        }
        predict_new_match(new_match_data)
    except FileNotFoundError as e:
        print(f"Error: No se encontró un archivo necesario. ({e})")
        print("Asegúrate de que 'datos_liga_procesados.csv' y la carpeta 'artifacts' estén presentes.")