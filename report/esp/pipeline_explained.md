# Pipeline de ML: Explicación Paso a Paso

Este documento explica el pipeline completo de machine learning para clasificación médica desbalanceada, con justificación de decisiones en cada etapa.

---

## Visión General del Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Cargar     │───▶│  Análisis   │───▶│ Estrategia  │───▶│  Entrenar   │
│  Datos      │    │ Desbalance  │    │ Remuestreo  │    │   Modelo    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  Decisión   │◀───│  Análisis   │◀───│  Evaluar    │◀─────────┘
│  Despliegue │    │  de Costo   │    │  en Test    │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## Etapa 1: Carga y Exploración de Datos

### Entrada
- CSV con 569 registros de pacientes
- 30 características de mediciones del núcleo celular
- Objetivo binario: M (maligno) o B (benigno)

### Operaciones

```python
def load_data(path: str = "data/breast_cancer.csv") -> tuple:
    df = pd.read_csv(path)
    X = df.drop(columns=["id", "diagnosis"])
    y = (df["diagnosis"] == "M").astype(int)
    return X, y
```

### Punto de Decisión: Selección de Características

**Pregunta**: ¿Deberíamos reducir la dimensionalidad?

**Respuesta**: No para este dataset.
- 30 características es manejable para SVM
- Las características médicas son interpretables y útiles para revisión clínica
- PCA oscurecería la importancia de características

---

## Etapa 2: Análisis de Desbalance

### Operaciones

```python
def get_class_distribution(y) -> dict:
    benign = (y == 0).sum()
    malignant = (y == 1).sum()
    return {
        "benign": benign,
        "malignant": malignant,
        "imbalance_ratio": round(benign / malignant, 2)
    }
```

### Salida Visual

![Distribución de Clases](../../figures/class_distribution.png)
*Figura 1: Visualización de distribución de clases revela la proporción de desbalance 1.68:1.*

### Salida
```
benigno: 357
maligno: 212
proporcion_desbalance: 1.68
```

### Punto de Decisión: ¿Es Necesario el Remuestreo?

**Heurísticas de umbral**:
- Proporción < 1.5: Usualmente no se necesita remuestreo
- Proporción 1.5-3: Considerar remuestreo según asimetría de costos
- Proporción > 3: Remuestreo casi siempre requerido

**Nuestro caso**: Proporción 1.68 + alta asimetría de costos → **Remuestreo o pesos de clase recomendados**

---

## Etapa 3: División Train/Test

### Operaciones

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,        # Preservar proporción de clases en ambos conjuntos
    random_state=42    # Reproducibilidad
)
```

### Punto de Decisión: Proporción de División

**Pregunta**: ¿80/20 o 70/30?

**Respuesta**: 80/20 es suficiente.
- 569 muestras es pequeño; necesitamos máximos datos de entrenamiento
- Estratificación asegura que test tenga muestras malignas representativas
- Para producción: usar validación cruzada anidada

---

## Etapa 4: Estrategia de Remuestreo

### Opción A: Pesos de Clase (Recomendada)

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
```

**Cómo funciona**:
- Ajusta función de pérdida para penalizar más errores de clase minoritaria
- Peso inversamente proporcional a frecuencia de clase

**Fortalezas**:
- Sin generación de datos sintéticos
- Preserva distribución original de datos
- Computacionalmente eficiente

### Opción B: SMOTE

```python
from imblearn.over_sampling import SMOTE

resampler = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
```

**Cómo funciona SMOTE**:
1. Seleccionar una muestra minoritaria
2. Encontrar sus k vecinos minoritarios más cercanos (k=5 por defecto)
3. Crear muestra sintética en el segmento de línea entre ellos
4. Repetir hasta que las clases estén balanceadas

**Fortalezas**:
- Sin pérdida de información (no descarta muestras mayoritarias)
- Crea interpolaciones plausibles en el espacio de características

**Debilidades**:
- Puede crear muestras ruidosas en regiones dispersas
- Asume que la clase minoritaria es convexa

---

## Etapa 5: Escalamiento de Características

### Operaciones

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # ¡Usar parámetros de entrenamiento!
```

### Crítico: Sin Fuga de Datos

El scaler se ajusta **solo** con datos de entrenamiento. Los datos de test se transforman usando estadísticas de entrenamiento.

**Incorrecto**:
```python
scaler.fit(X_all)  # Fuga distribución de test al entrenamiento
```

**Correcto**:
```python
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Etapa 6: Entrenamiento del Modelo

### Operaciones

```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',            # Función de base radial para fronteras no lineales
    C=1.0,                   # Regularización (predeterminado)
    class_weight='balanced', # Manejar desbalance
    random_state=42
)
model.fit(X_train_scaled, y_train)
```

### Punto de Decisión: ¿Por Qué SVM?

**Pros**:
- Efectivo en espacios de alta dimensionalidad (30 características)
- Eficiente en memoria (almacena solo vectores de soporte)
- Funciona bien con margen de separación claro

**Contras**:
- No produce probabilidades calibradas por defecto
- Más lento en datasets muy grandes

**Alternativa**: Random Forest o Regresión Logística para calibración de probabilidad.

---

## Etapa 7: Evaluación

### Operaciones

```python
y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
# [[VN, FP],
#  [FN, VP]]
```

### Salida Visual

![Matrices de Confusión](../../figures/confusion_matrices_comparison.png)
*Figura 2: Matrices de confusión para los tres modelos. Cada celda muestra el conteo de predicciones.*

### Métricas Calculadas

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| Exactitud | (VP+VN)/(VP+VN+FP+FN) | Corrección general |
| Precisión | VP/(VP+FP) | "De los predichos malignos, ¿cuántos son correctos?" |
| Recall | VP/(VP+FN) | "De los malignos reales, ¿cuántos encontramos?" |
| F1 | 2×(Prec×Rec)/(Prec+Rec) | Media armónica |

![Comparación de Métricas](../../figures/metrics_comparison.png)
*Figura 3: Comparación de métricas entre modelos. Note cómo la exactitud enmascara diferencias en recall.*

### Crítico: Por Qué el Recall Importa Más

En tamizaje de cáncer:
- **Falso Negativo** = Paciente con cáncer informado que está sano → tratamiento retrasado → muerte
- **Falso Positivo** = Paciente sano enviado a biopsia → ansiedad + $500

El recall mide directamente la tasa de falsos negativos.

---

## Etapa 8: Análisis ROC y Precisión-Recall

### Curvas ROC

![Curvas ROC](../../figures/roc_curves.png)
*Figura 4: Curvas ROC muestran capacidad de discriminación. Alto AUC indica buena separación general.*

### Curvas Precisión-Recall

![Curvas Precisión-Recall](../../figures/precision_recall_curves.png)
*Figura 5: Curvas PR son más informativas para datos desbalanceados. Enfocarse en región de alto recall.*

---

## Etapa 9: Análisis de Costo

### Operaciones

```python
def cost_analysis(results: dict, cost_fn=50000, cost_fp=500) -> dict:
    fn = results["false_negatives"]
    fp = results["false_positives"]
    return {
        "total_cost": fn * cost_fn + fp * cost_fp
    }
```

### Salida Visual

![Comparación de Costos](../../figures/cost_comparison.png)
*Figura 6: Desglose de costos por modelo. Costos de falsos negativos dominan el costo total.*

### Punto de Decisión: Establecer Proporción de Costos

**Pregunta**: ¿Cómo determinar la proporción de costos FN:FP?

**Fuentes**:
1. **Literatura médica**: Costos de retraso de tratamiento, tasas de supervivencia
2. **Datos legales**: Compensación promedio por mala praxis en diagnóstico fallido
3. **Datos operacionales**: Costos de procedimiento de biopsia

Nuestra proporción 100:1 es conservadora. Entornos médicos reales pueden usar 500:1 o más.

---

## Etapa 10: Validación del Modelo

### Validación Cruzada

![Validación Cruzada](../../figures/cross_validation_recall.png)
*Figura 7: Validación cruzada 5-fold confirma estabilidad del modelo a través de divisiones de datos.*

### Curvas de Aprendizaje

![Curvas de Aprendizaje](../../figures/learning_curves.png)
*Figura 8: Curvas de aprendizaje diagnostican sesgo/varianza. Convergencia indica que no hay sobreajuste.*

---

## Etapa 11: Selección de Modelo

### Matriz de Comparación

| Modelo | Exactitud | Recall | Costo Total |
|--------|-----------|--------|-------------|
| SVM Naive | 97.4% | 92.9% | $150,000 |
| SVM Ponderado | 98.2% | 97.6% | $50,500 |
| SMOTE + SVM | 97.4% | 95.2% | $100,500 |

### Decisión

**Seleccionado**: SVM Ponderado

**Justificación**:
- Máximo recall (97.6%)
- Menor costo ($50,500)
- **66% de reducción en costo** vs enfoque naive

La métrica de exactitud habría sido ambigua—la métrica de costo es decisiva.

---

## Etapa 12: Consideraciones de Producción

### Ajuste de Umbral

El umbral predeterminado es 0.5. Para tamizaje médico de alto riesgo:

```python
# Obtener scores de probabilidad
model = SVC(probability=True)
y_scores = model.predict_proba(X_test)[:, 1]

# Reducir umbral para capturar más casos malignos
threshold = 0.3
y_pred = (y_scores >= threshold).astype(int)
```

### Monitoreo

Rastrear estas métricas semanalmente en producción:
1. Recall en casos malignos confirmados
2. Tasa de falsos positivos
3. Deriva de características (¿están cambiando las distribuciones de entrada?)

### Disparadores de Reentrenamiento

- Recall cae por debajo de 95%
- Nuevo equipo de imagen cambia distribuciones de características
- Nuevos subtipos tumorales identificados

---

## Resumen

El pipeline demuestra que **la selección de métricas es una decisión de diseño**, no un valor técnico predeterminado. Cada etapa involucra decisiones que deben alinearse con el objetivo de negocio: minimizar daño al paciente, no maximizar un número en un dashboard.
