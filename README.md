# 🎲 MLB Prediction AI

Una **inteligencia artificial completa** para pronosticar resultados de juegos de Grandes Ligas de Béisbol (MLB) utilizando **Machine Learning**, datos en tiempo real de la API oficial de MLB y factores climáticos.

## 🎯 Características Principales

✅ **Predicción de Ganador** - Predice qué equipo ganará el juego (Local vs Visitante)
✅ **Predicción de Runs** - Estima el total de carreras que se anotarán
✅ **API en Tiempo Real** - Integración con la API oficial de MLB Stats
✅ **Datos Climáticos** - Factores meteorológicos que afectan el juego
✅ **Machine Learning Avanzado** - Modelos XGBoost entrenados con datos históricos
✅ **REST API** - Endpoints Flask para hacer predicciones
✅ **Feature Engineering** - Ingeniería avanzada de características

## 📊 Variables Utilizadas

### Estadísticas de Equipos
- Carreras anotadas y permitidas
- Promedio de bateo
- Jonrones
- Strikeouts y Walks
- ERA (Efectividad de los lanzadores)
- WHIP (Runners por inning lanzado)
- Récord de victorias/derrotas
- Promedio de juegos ganados

### Factores Climáticos
- 🌡️ Temperatura (impacta jonrones)
- 💨 Velocidad y dirección del viento
- 💧 Humedad
- 🌧️ Precipitación
- 📊 Presión atmosférica

### Información de Estadios
- Altitud (ej: Coors Field = 5,277 ft = más jonrones)
- Capacidad
- Año de construcción
- Factor de parque (park factor)

### Características Temporales
- Día de la semana
- Mes y fase de temporada
- Días de descanso

## 🚀 Inicio Rápido

### 1. Clonar el repositorio

```bash
git clone https://github.com/uwillcamero71-glitch/mlb-prediction-ai.git
cd mlb-prediction-ai
```

### 2. Crear ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tu API key de OpenWeatherMap
```

### 5. Entrenar los modelos

```bash
python train_models.py
```

Esto generará datos sintéticos y entrenará dos modelos:
- **Modelo de Ganador**: Predicción de qué equipo gana
- **Modelo de Runs**: Predicción del total de carreras

### 6. Ejecutar la API

```bash
python app.py
```

La API estará disponible en `http://localhost:5000`

## 📡 Endpoints de la API

### 1. Health Check
```bash
GET /health
```

Respuesta:
```json
{
  "status": "healthy",
  "timestamp": "2024-09-10T19:30:00",
  "models_loaded": true
}
```

### 2. Listar Equipos
```bash
GET /teams
```

### 3. Juegos Próximos
```bash
GET /games/upcoming?days=7
```

### 4. Predicción de Juego (Principal)
```bash
POST /predict
Content-Type: application/json

{
  "home_team_id": 147,
  "away_team_id": 108,
  "game_datetime": "2024-09-10T19:05:00Z",
  "venue_id": 3
}
```

**Respuesta:**
```json
{
  "status": "success",
  "prediction": {
    "winner": {
      "prediction": "home_win",
      "confidence": 62.5,
      "probabilities": {
        "away_win": 37.5,
        "home_win": 62.5
      }
    },
    "total_runs": {
      "predicted_runs": 8.3,
      "lower_bound": 5.8,
      "upper_bound": 10.8,
      "over_under_8": "OVER"
    }
  }
}
```

### 5. Estado de Modelos
```bash
GET /models/status
```

### 6. Cargar Modelos
```bash
POST /models/load
```

## 📂 Estructura del Proyecto

```
mlb-prediction-ai/
├── src/
│   ├── __init__.py
│   ├── data_processor.py      # Procesamiento y feature engineering
│   ├── mlb_api_client.py      # Cliente para APIs MLB y clima
│   └── ml_models.py           # Modelos de Machine Learning
├── models/                     # Modelos entrenados (generados)
│   ├── winner_model.pkl
│   ├── runs_model.pkl
│   └── scaler.pkl
├── app.py                      # Aplicación Flask
├── train_models.py             # Script de entrenamiento
├── config.py                   # Configuración
├── requirements.txt            # Dependencias
├── .env.example               # Plantilla de variables
└── README.md                  # Este archivo
```

## 🧠 Modelos Machine Learning

### Modelo de Ganador
- **Tipo**: Clasificación Binaria (XGBoost)
- **Entrada**: 50+ características
- **Salida**: Probabilidad de victoria (Local/Visitante)
- **Métricas**: Accuracy, Precision, Recall, F1-Score, AUC

### Modelo de Runs
- **Tipo**: Regresión (XGBoost)
- **Entrada**: 50+ características
- **Salida**: Total de carreras predichas
- **Métricas**: RMSE, MAE, R² Score

## 📈 Factores de Predicción

### Impacto en Jonrones 🚀
- **Temperatura**: Cada grado Fahrenheit adicional ≈ 1% más jonrones
- **Altitud**: Coors Field (5,277 ft) = +8% jonrones vs mar
- **Viento**: Viento hacia los jardines = más jonrones
- **Humedad**: Aire seco = pelota viaja más lejos

### Impacto en Victorias ⚾
- Diferencial de carreras del equipo
- Porcentaje de victorias histórico
- ERA de los lanzadores
- Ventaja de jugar en casa (+3% aproximadamente)

## 🔑 IDs de Estadios Principales

| Team | Venue ID | Stadium | Altitude |
|------|----------|---------|----------|
| NYY | 3 | Yankee Stadium | 20 ft |
| BOS | 3 | Fenway Park | 21 ft |
| CHC | 17 | Wrigley Field | 673 ft |
| LAD | 22 | Dodger Stadium | 340 ft |
| COL | 27 | Coors Field | **5,277 ft** ⭐ |
| SF | 25 | AT&T Park | 63 ft |
| ARI | 15 | Chase Field | 1,107 ft |

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Flask** - Framework web REST API
- **XGBoost** - Modelos de Machine Learning
- **Pandas** - Manipulación de datos
- **Scikit-learn** - Preprocesamiento y métricas
- **Requests** - Cliente HTTP para APIs

## 📊 Ejemplo de Uso Completo

```python
from src.data_processor import DataProcessor
from src.ml_models import MLBPredictionModel
from src.mlb_api_client import MLBAPIClient

# Inicializar componentes
client = MLBAPIClient()
processor = DataProcessor()
model = MLBPredictionModel()

# Cargar modelos entrenados
model.load_models('models/winner_model.pkl', 'models/runs_model.pkl', 'models/scaler.pkl')

# Obtener datos del juego
home_stats = {'runs_scored': 5, 'batting_avg': 0.270, ...}
away_stats = {'runs_scored': 4, 'batting_avg': 0.250, ...}
weather = {'temperature': 78, 'humidity': 55, 'wind_speed': 10, ...}
venue = {'altitude': 5277, 'capacity': 50430, ...}

# Procesar features
features = processor.process_game_data(
    'Colorado Rockies', 'Los Angeles Dodgers',
    '2024-09-10T19:10:00Z',
    home_stats, away_stats, weather, venue
)

# Hacer predicción
prediction = model.predict_game(features)
print(f"Ganador: {prediction['winner_prediction']['prediction']}")
print(f"Confianza: {prediction['winner_prediction']['confidence']}%")
print(f"Runs: {prediction['runs_prediction']['predicted_runs']}")
```

## 🧪 Testing

```bash
# Ejecutar entrenamiento con datos sintéticos
python train_models.py

# Probar API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team_id": 147,
    "away_team_id": 108,
    "game_datetime": "2024-09-10T19:05:00Z",
    "venue_id": 3
  }'
```

## 🔐 Seguridad

- Nunca commitear el archivo `.env` (está en `.gitignore`)
- Usar variables de entorno para API keys
- Las claves API deben estar protegidas en producción

## 📝 Licencia

MIT License - Ver LICENSE.md

## 👨‍💻 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

Si tienes preguntas o problemas, abre un issue en GitHub.

## 🎓 Mejoras Futuras

- [ ] Integración con datos históricos reales de MLB Stats API
- [ ] Dashboard web interactivo
- [ ] Predicciones en tiempo real durante juegos
- [ ] Modelo de probabilidades de playoffs
- [ ] Análisis de lesiones de jugadores
- [ ] Sistema de apuestas recomendadas
- [ ] Predicción de estadísticas individuales de jugadores
- [ ] API de WebSocket para predicciones en vivo

---

⭐ **¡Si te gusta este proyecto, dale una estrella en GitHub!** ⭐
