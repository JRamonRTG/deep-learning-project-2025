<div align="center">

# **Proyecto IA ML – Introducción a Deep Learning**  
### Redes Neuronales aplicadas a datos reales  
**Autor:** *Ramón Terraza*  
**Universidad Galileo**  
**2025**

---

## Arquitecturas Implementadas
- **MLP** (Big Five Personality Prediction)  
- **CNN** (Clasificación de Tortugas Tierra vs Agua)  
- **RNN – BiLSTM** (Clasificación de Frases en Dovahzul, idioma ficticio del juego The Elder Scrolls V: Skyrim)

---

### Datasets Table

| Proyecto | Dataset | Fuente | Enlace |
|----------|---------|---------|--------|
| **MLP** | Big Five Personality Test (≈1M respuestas) | Kaggle | [Dataset en Kaggle](https://www.kaggle.com/datasets/tunguz/big-five-personality-test/data) |
| **CNN** | Tortoise Images Dataset (land vs water) | Varias fuentes (Kaggle, Images.CV, imágenes filtradas manualmente) |  Kaggle: [Dolphin/Tortoise Dataset](https://www.kaggle.com/datasets/deepakat002/dolphintortoise?resource=download&select=ff9d9ce2602aeafe.jpg) <br>  images.cv Dataset: [Tortoise Classification Dataset](https://images.cv/dataset/tortoise-image-classification-dataset) <br>  images.cv Search: [Turtle Labeled Images](https://images.cv/search-labeled-image-dataset?query=Turtle) |
| **RNN** | Dovahzul–English Parallel Corpus | Dataset personalizado + ampliado | [Thuum.org – sección *Downloads*](https://www.thuum.org/) |

---



</div>


---

# Comparativa General de Modelos

| Característica | **MLP – Big Five** | **CNN – Tortugas** | **RNN – Dovahzul** |
|----------------|---------------------|---------------------|---------------------|
| Tipo de problema | Clasificación (5 clases) | Clasificación (2 clases) | Clasificación (6 clases) |
| Dataset | 1M+ filas (tras limpieza ≈ 850k) | 1,200 imágenes (seleccionadas) | 8,000+ frases (ampliadas) |
| Input | 50 respuestas Likert + atributos | Imagen 224×224 RGB | Secuencia de caracteres |
| Arquitectura | 4 capas densas + Dropout | 3 bloques Conv-BN-ReLU | BiLSTM + Embeddings |
| Optimizador | Adam | Adam + LR Scheduler | Adam |
| Regularización | Dropout | Data Augmentation | Oversampling + Dropout |
| Accuracy final test | **≈ 80% - 90%** | **≈ 86%** | **≈ 83%** |
| Métrica F1-score | F1-score macro ≈ 0.80 | F1-score ≈ 0.86 | F1-score macro ≈ 0.82 |
| Limitaciones | Dataset ruidoso | Dataset pequeño | Clases desbalanceadas |
| Posibles mejoras | Normalización avanzada | Transfer learning (ResNet) | Transformers |

---

# Arquitectura Global del Proyecto

```ascii
                           ┌───────────────────────────┐
                           │        DATASETS           │
                           ├───────────────────────────┤
                           │ Big Five (MLP)            │
                           │ Tortugas (CNN)            │
                           │ Dovahzul (RNN)            │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                       ┌────────────────────────────────────┐
                       │   PREPROCESAMIENTO POR MODELO      │
                       ├────────────────────────────────────┤
                       │ MLP: Limpieza NaN, estandarización │
                       │ CNN: Split imágenes + augmentation │
                       │ RNN: Tokenización + balanceo       │
                       └─────────────┬────────────┬─────────┘
                                     │            │
                                     ▼            ▼
         ┌────────────────────┐   ┌─────────────────────┐   ┌────────────────────┐
         │    MLP MODEL       │   │     CNN MODEL        │   │      RNN MODEL      │
         │ Big Five Prediction│   │ Tortoise Classifier  │   │ Dovahzul Classifier │
         ├────────────────────┤   ├─────────────────────┤   ├─────────────────────┤
         │ Dense layers +     │   │ Conv2D + BN + ReLU   │   │ Embedding + BiLSTM  │
         │ dropout + Adam     │   │ Fully Connected head │   │ Fully Connected head │
         └──────────┬─────────┘   └─────────┬───────────┘   └──────────┬──────────┘
                    │                       │                          │
                    ▼                       ▼                          ▼
            ┌────────────────┐      ┌────────────────┐       ┌────────────────────┐
            │   TRAIN LOOP   │      │   TRAIN LOOP    │       │    TRAIN LOOP       │
            │  Loss + metrics│      │  Augmentation   │       │ CrossEntropy + F1   │
            └────────┬───────┘      └────────┬────────┘       └─────────┬──────────┘
                     │                        │                           │
                     ▼                        ▼                           ▼
            ┌────────────────┐       ┌────────────────┐      ┌────────────────────┐
            │   VALIDACIÓN   │       │  VALIDACIÓN    │      │     VALIDACIÓN     │
            └────────┬───────┘       └────────┬──────┘      └──────────┬─────────┘
                     │                        │                          │
                     ▼                        ▼                          ▼
          ┌────────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
          │     TEST + F1      │   │  TEST + CONFUSION    │   │ TEST + MATRIZ CONF │
          └────────────────────┘   └──────────────────────┘   └────────────────────┘
```

## Mas sobre datasets utilizados

### Big Five Personality Test — 1M samples (Kaggle)

Dataset masivo con más de 1 millón de respuestas.
Incluye 50 ítems (OCEAN) evaluados en escala Likert.

Variables:

- 50 preguntas del test OCEAN
- Etiqueta generada: rasgo dominante (O, C, E, A, N)
- siendo: Clasificación multiclase (5 clases)

### Tortoise Classification (Water vs Land)

Dataset descargado desde images.cv, kaggle y seleccion de imagenes con carpetas separadas:

/land → tortugas terrestres

/water → tortugas marinas

Transformaciones aplicadas:

- RandomResizedCrop
- HorizontalFlip
- Rotation
- ColorJitter
- siendo: Clasificación binaria

### Dovahzul–English Parallel Corpus (Dragon Language)

Dataset con  frases traducidas del lenguaje ficticio Dovahzul.
Se generaron etiquetas nuevas:

- greeting
- power
- attack
- object
- action
- other

siendo: Clasificación multiclase con texto
Se aplicaron embeddings + LSTM bidireccional.

## Modelos Implementados

### MLP — Big Five Personality Prediction

Notebook: 01_MLP.ipynb

Arquitectura: 

- 3 capas densas ocultas
- Activación ReLU
- Dropout en todas las capas
- Normalización de datos
- Optimizador Adam
- CrossEntropy para clasificación de 5 clases

Resultados principales:

- Accuracy (test): ~80% - 90%
- Mejoras obtenidas:
- Limpieza agresiva de NaNs en 50 ítems
- Eliminación de respuestas inválidas (código = 0)
- Balanceo de clases

Modelo MLP (3–5 capas)

### CNN — Clasificación de tortugas tierra vs agua

Notebook: 02_CNN.ipynb

Arquitectura:

- 3 bloques Conv2D + BatchNorm + ReLU + MaxPool
- Capa fully connected con Dropout
- Data augmentation agresiva
- Scheduler ReduceLROnPlateau

Resultados principales:

- Accuracy en Test: ~86%
- Precision y Recall balanceados
- Matriz de confusión estable y sin sesgo por clase

### RNN (BiLSTM) — Clasificación de frases en Dovahzul

Notebook: 03_RNN.ipynb

Arquitectura:

- Embedding layer
- LSTM bidireccional de 128 unidades
- Capa fully connected para clasificación
- Stratified sampling + oversampling por clase
- Tokenización carácter–nivel

Resultados principales

- Accuracy en Test: ~83%
- F1-score equilibrado entre clases subrepresentadas
- Matriz de confusión con pocas confusiones semánticas



## Conclusiones Finales

### Aprendizaje:
- Construcción de pipelines de Deep Learning
- Tratamiento de datasets reales, ruidosos y desbalanceados
- Uso correcto de train/val/test
- Entrenamiento con GPU y optimización mediante LR scheduler

### Implementación de 3 arquitecturas distintas:
- MLP
- CNN
- RNN (LSTM)

### Desafíos
- Limpieza del dataset Big Five
- Balanceo de clases en Dovahzul
- Overfitting en el modelo CNN

### Posibles mejoras

- Implementar Transformers para NLP
- Usar transfer learning en visión (ResNet, EfficientNet)
- Implementar un VAE para generación de frases o perfiles de personalidad
- Aplicar técnicas de hiperparámetros (Optuna, Ray Tune)