# app/main.py
import pandas as pd
import joblib
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictionRequest, PredictionResponse

# Inicializar la app de FastAPI
app = FastAPI(title="LaLiga Prediction API")

# Configurar CORS para permitir peticiones desde tu frontend de React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, limita esto a la URL de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar los artefactos del modelo al iniciar la API
model = joblib.load('artifacts/random_forest_model.joblib')
scaler = joblib.load('artifacts/scaler.joblib')
with open('artifacts/model_columns.json', 'r') as f:
    model_columns = json.load(f)

@app.get("/")
def read_root():
    """Endpoint raíz para verificar que la API está funcionando."""
    return {"status": "ok", "message": "LaLiga Prediction API is running!"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint para recibir datos de un partido y devolver una predicción.
    """
    # 1. Crear un DataFrame con la misma estructura que el de entrenamiento
    input_data = pd.DataFrame([dict(request)], columns=model_columns)
    
    # 2. Rellenar los valores que faltan (one-hot encoding)
    # Creamos un DataFrame vacío con las columnas del modelo
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0 # Inicializamos la fila con ceros

    # Establecemos los valores recibidos en la solicitud
    home_team_col = f"HomeTeam_{request.home_team}"
    away_team_col = f"AwayTeam_{request.away_team}"

    if home_team_col in input_df.columns:
        input_df.loc[0, home_team_col] = 1
    if away_team_col in input_df.columns:
        input_df.loc[0, away_team_col] = 1

    # Aquí agregarías el resto de características del request
    # Ejemplo simplificado:
    input_df.loc[0, 'H_Forma_Goles_Anotados'] = request.h_form_goals
    input_df.loc[0, 'A_Forma_Goles_Anotados'] = request.a_form_goals
    input_df.loc[0, 'HST'] = request.h_shots_on_target
    input_df.loc[0, 'AST'] = request.a_shots_on_target
    
    # 3. Normalizar los datos numéricos con el mismo scaler del entrenamiento
    numeric_cols = scaler.get_feature_names_out()
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # 4. Realizar la predicción
    probabilities = model.predict_proba(input_df)[0]
    prediction_index = probabilities.argmax()
    prediction_label = model.classes_[prediction_index]

    # 5. Formatear la respuesta
    response_probs = {cls: prob for cls, prob in zip(model.classes_, probabilities)}

    return PredictionResponse(
        prediction=prediction_label,
        probabilities=response_probs
    )