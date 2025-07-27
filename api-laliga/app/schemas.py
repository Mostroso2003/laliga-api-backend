# app/schemas.py
from pydantic import BaseModel
from typing import Dict

class PredictionRequest(BaseModel):
    # En un modelo más complejo, aquí recibiríamos todas las características.
    # Para este ejemplo, simplificamos a solo los equipos.
    home_team: str
    away_team: str
    # Ejemplo de otras características que podríamos recibir:
    h_form_goals: float
    a_form_goals: float
    h_shots_on_target: float
    a_shots_on_target: float


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]