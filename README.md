# ML Metrics That Matter

### Cuando el 97% de Exactitud Cuesta Vidas

<p align="center">
  <img src="figures/cost_comparison.png" alt="Comparación de Costos" width="700"/>
</p>

<p align="center">
  <strong>El modelo "mejor" por exactitud cuesta 3 veces más que el "peor".</strong><br>
  <em>Este repositorio explica por qué—y qué hacer al respecto.</em>
</p>

<p align="center">
  <a href="#documentación">Documentación</a> •
  <a href="#el-caso-de-estudio">Caso de Estudio</a> •
  <a href="#resultados">Resultados</a> •
  <a href="#cómo-ejecutar">Ejecutar</a>
</p>

---

## Documentación

Este proyecto incluye documentación exhaustiva para diferentes audiencias y niveles de profundidad técnica.

### Documentos Disponibles

| Documento | ¿Para Quién? | ¿Qué Contiene? |
|-----------|--------------|----------------|
| [**Resumen Ejecutivo**](report/esp/executive_summary.md) | Reclutadores, Gerentes, Directivos | Impacto de negocio, resultados clave, competencias demostradas |
| [**Caso de Estudio**](report/esp/case_study.md) | Revisores de Portafolio, Entrevistadores | Narrativa completa del problema, análisis y solución |
| [**Reporte Técnico**](report/esp/technical_report.md) | Líderes Técnicos, Ingenieros ML | Implementación, benchmarks, arquitectura del código |
| [**Pipeline Explicado**](report/esp/pipeline_explained.md) | Científicos de Datos, Analistas | Pipeline paso a paso con justificación de cada decisión |
| [**Fórmulas Matemáticas**](report/esp/mathematical_formulas.md) | Estadísticos, Actuarios | Definiciones formales, derivaciones matemáticas |
| [**Marco Teórico**](report/esp/theoretical_framework.md) | Academia, Investigadores | Fundamentos teóricos completos, referencias académicas |

### English Documentation

| Document | Audience | Content |
|----------|----------|---------|
| [Executive Summary](report/eng/executive_summary.md) | Recruiters, Managers | Business impact, key results |
| [Case Study](report/eng/case_study.md) | Portfolio reviewers | Full narrative |
| [Technical Report](report/eng/technical_report.md) | Tech Leads, Engineers | Implementation details |
| [Pipeline Explained](report/eng/pipeline_explained.md) | Data Scientists | Step-by-step decisions |
| [Mathematical Formulas](report/eng/mathematical_formulas.md) | Statisticians | Formal definitions |
| [Theoretical Framework](report/eng/theoretical_framework.md) | Researchers | Complete theory |

---

## El Caso de Estudio

### Contexto: Un Hospital, Una Decisión Crítica

Imagina que eres el director de tecnología de un sistema hospitalario. Tu equipo de ciencia de datos ha desarrollado un modelo de inteligencia artificial para asistir a los radiólogos en la detección temprana de cáncer de mama.

El modelo analiza imágenes de aspirados con aguja fina y clasifica cada muestra como **maligna** (cáncer) o **benigna** (sin cáncer). Una herramienta así podría salvar vidas al detectar tumores que el ojo humano podría pasar por alto.

Tu equipo te presenta tres opciones:

```
┌─────────────────────────────────────────────────────────────────┐
│                   PROPUESTA DEL EQUIPO DE ML                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   "Hemos entrenado tres modelos. Aquí están los resultados:"    │
│                                                                 │
│   ┌─────────────────┬──────────────┬──────────────┐             │
│   │     Modelo      │   Exactitud  │    Recall    │             │
│   ├─────────────────┼──────────────┼──────────────┤             │
│   │  SVM Naive      │    97.4%     │    92.9%     │             │
│   │  SVM Ponderado  │    98.2%     │    97.6%     │             │
│   │  SMOTE + SVM    │    97.4%     │    95.2%     │             │
│   └─────────────────┴──────────────┴──────────────┘             │
│                                                                 │
│   "Recomendamos el SVM Ponderado por su mayor exactitud."       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**La pregunta que debes hacerte:** ¿Es la exactitud la métrica correcta para tomar esta decisión?

### El Problema Oculto: Desbalance de Clases

<p align="center">
  <img src="figures/class_distribution.png" alt="Distribución de Clases" width="600"/>
</p>

El dataset contiene **357 muestras benignas** y **212 malignas**—una proporción de 1.68:1. A primera vista, esto no parece problemático. Pero considera esto:

> **Un modelo que predice "benigno" para TODOS los pacientes logra 63% de exactitud.**
>
> También falla en detectar el **100% de los cánceres**.

Este es el corazón del problema: **la exactitud trata todos los errores como iguales**, pero en medicina, no lo son.

### La Asimetría Que Lo Cambia Todo

En el diagnóstico médico, existen dos tipos de errores:

| Tipo de Error | Qué Significa | Consecuencia Real | Costo Estimado |
|---------------|---------------|-------------------|----------------|
| **Falso Positivo** | Decirle a un paciente sano que podría tener cáncer | Biopsia innecesaria, ansiedad, días de trabajo perdidos | ~$500 |
| **Falso Negativo** | Decirle a un paciente con cáncer que está sano | El tumor crece, hace metástasis, el paciente puede morir | ~$50,000+ |

Esta proporción de **100:1** no es arbitraria. Refleja:
- Costos de tratamiento tardío vs. temprano
- Demandas por negligencia médica
- El valor incalculable de una vida humana

### Lo Que Los Números No Te Dicen

Volvamos a la tabla original, pero ahora con la información completa:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LA HISTORIA COMPLETA                                 │
├─────────────────┬──────────────┬──────────────┬─────────────┬──────────┤
│     Modelo      │   Exactitud  │    Recall    │  Cánceres   │  Costo   │
│                 │              │              │  Perdidos   │  Total   │
├─────────────────┼──────────────┼──────────────┼─────────────┼──────────┤
│  SVM Naive      │    97.4%     │    92.9%     │      3      │ $150,000 │
│  SVM Ponderado  │    98.2%     │    97.6%     │      1      │  $50,500 │
│  SMOTE + SVM    │    97.4%     │    95.2%     │      2      │ $100,500 │
└─────────────────┴──────────────┴──────────────┴─────────────┴──────────┘

         El modelo con MISMA exactitud puede costar 3 VECES más.
```

**El insight crítico:** El SVM Ponderado no solo es "mejor" por exactitud—ahorra **$99,500 por cada 100 pacientes** y detecta **2 cánceres adicionales** que el modelo naive habría pasado por alto.

Esos 2 pacientes tienen nombres, familias, y vidas por delante.

---

## Resultados

### Matrices de Confusión: Donde Se Esconden Los Errores

<p align="center">
  <img src="figures/confusion_matrices_comparison.png" alt="Matrices de Confusión" width="800"/>
</p>

Cada celda cuenta una historia:
- **Esquina superior izquierda:** Pacientes sanos correctamente identificados ✓
- **Esquina inferior derecha:** Cánceres correctamente detectados ✓
- **Esquina superior derecha:** Falsos positivos (biopsias innecesarias)
- **Esquina inferior izquierda:** **Falsos negativos (cánceres perdidos)** ← El número crítico

### Comparación de Métricas

<p align="center">
  <img src="figures/metrics_comparison.png" alt="Comparación de Métricas" width="700"/>
</p>

Observa cómo **la exactitud se ve casi igual** entre modelos, pero **el recall varía significativamente**. Para diagnóstico médico, el recall es lo que importa.

### Curvas ROC y Precisión-Recall

<p align="center">
  <img src="figures/roc_curves.png" alt="Curvas ROC" width="700"/>
</p>

<p align="center">
  <img src="figures/precision_recall_curves.png" alt="Curvas Precisión-Recall" width="700"/>
</p>

En tamizaje médico, priorizamos la **región de alto recall** (lado derecho de la curva PR), aceptando menor precisión para capturar más cánceres.

### Validación del Modelo

<p align="center">
  <img src="figures/cross_validation_recall.png" alt="Validación Cruzada" width="700"/>
</p>

La validación cruzada 5-fold confirma que el SVM Ponderado mantiene **recall consistentemente alto** con baja varianza.

<p align="center">
  <img src="figures/learning_curves.png" alt="Curvas de Aprendizaje" width="700"/>
</p>

La convergencia de los scores de entrenamiento y validación indica **que no hay sobreajuste**—el modelo generaliza bien.

---

## Marco de Decisión

```
┌──────────────────────────────────────────────────────────────────┐
│          ANTES DE ENTRENAR CON DATOS DESBALANCEADOS              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. CUANTIFICAR LA PROPORCIÓN DE COSTOS                         │
│      └── ¿Cuánto peor es un falso negativo vs falso positivo?    │
│                                                                  │
│   2. ESTABLECER UMBRALES MÍNIMOS DE RECALL                       │
│      └── ¿Cuál es la tasa mínima aceptable de detección?         │
│                                                                  │
│   3. ELEGIR LA ESTRATEGIA                                        │
│      ├── Pesos de clase (simple, preserva datos)                 │
│      ├── SMOTE (sobremuestreo sintético)                         │
│      └── Tomek Links (limpieza de frontera)                      │
│                                                                  │
│   4. EVALUAR CON MÉTRICAS PONDERADAS POR COSTO                   │
│      └── Costo Total = (FN × Costo_FN) + (FP × Costo_FP)         │
│                                                                  │
│   5. REPORTAR MATRICES DE CONFUSIÓN                              │
│      └── Los scores únicos esconden modos de falla críticos      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Cómo Ejecutar

```bash
# Clonar el repositorio
git clone https://github.com/oscgonz19/ml-metrics-that-matter.git
cd ml-metrics-that-matter

# Instalar dependencias
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn

# Ejecutar el análisis
python src/model.py
```

**Salida esperada:**

```
============================================================
      BREAST CANCER CLASSIFICATION: THE ACCURACY TRAP
============================================================

                      Dataset Summary
────────────────────────────────────────────────────────────
  Benign samples:     357
  Malignant samples:  212
  Imbalance ratio:    1.68:1

------------------------------------------------------------
                       MODEL RESULTS
------------------------------------------------------------

Model:               Weighted SVM
Accuracy:            98.2%
Recall:              97.6%
False Negatives:     1 (missed cancers)
Estimated Cost:      $50,500

============================================================
                      DECISION SUMMARY
============================================================

  AHORRO POTENCIAL: $99,500 por cada 100 pacientes
```

---

## Estructura del Repositorio

```
ml-metrics-that-matter/
│
├── src/
│   └── model.py                 # Pipeline completo de análisis
│
├── data/
│   └── breast_cancer.csv        # Dataset Wisconsin Breast Cancer
│
├── figures/                     # Todas las visualizaciones
│   ├── class_distribution.png
│   ├── confusion_matrices_comparison.png
│   ├── cost_comparison.png
│   ├── cross_validation_recall.png
│   ├── learning_curves.png
│   ├── metrics_comparison.png
│   ├── precision_recall_curves.png
│   └── roc_curves.png
│
├── report/
│   ├── eng/                     # Documentación en inglés
│   └── esp/                     # Documentación en español
│
├── README.md                    # Este archivo (español)
└── README_EN.md                 # Versión en inglés
```

---

## Métodos Comparados

| Enfoque | Cómo Funciona | Mejor Para | Limitaciones |
|---------|---------------|------------|--------------|
| **SVM Naive** | Optimización estándar | Línea base | Sesgo hacia mayoría |
| **Pesos de Clase** | Penaliza errores de minoría | La mayoría de casos | Requiere ajuste |
| **SMOTE** | Sobremuestreo sintético | Datasets pequeños | Crea artefactos |
| **Tomek Links** | Limpieza de frontera | Combinado con SMOTE | Impacto mínimo solo |

---

## Conclusiones Clave

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   "La exactitud es un proxy de lo que nos importa.         │
│    Cuando el proxy diverge de la realidad,                 │
│    optimiza la realidad—no el proxy."                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

1. **La exactitud miente** cuando las clases tienen costos diferentes
2. **El recall salva vidas** en tamizaje médico
3. **El análisis de costos** alinea ML con objetivos de negocio
4. **Las matrices de confusión** revelan lo que las métricas únicas esconden

---

## Stack Tecnológico

- **Python 3.9+**
- **scikit-learn** - Modelos ML y evaluación
- **imbalanced-learn** - SMOTE, Tomek Links
- **matplotlib/seaborn** - Visualizaciones
- **pandas/numpy** - Procesamiento de datos

---

## Referencias

- He, H., & Garcia, E. A. (2009). [Learning from Imbalanced Data](https://www.jair.org/index.php/jair/article/view/10302). *Journal of Artificial Intelligence Research*
- Chawla, N. V., et al. (2002). [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813). *Journal of Artificial Intelligence Research*
- Hastie, T., Tibshirani, R., & Friedman, J. [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)

---

<p align="center">
  <strong>La lección:</strong> Las métricas codifican valores. Elige métricas que codifiquen <em>tus</em> valores.
</p>
