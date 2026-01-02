# La Trampa de la Exactitud en el Diagnóstico Médico

## El Escenario

Un sistema hospitalario quiere implementar un modelo de machine learning para asistir a los radiólogos en el tamizaje de cáncer de mama. El modelo clasificará muestras tumorales como **malignas** o **benignas** basándose en mediciones del núcleo celular de aspirados con aguja fina.

El equipo de ciencia de datos presenta dos modelos:
- **Modelo A**: 97% de exactitud
- **Modelo B**: 94% de exactitud

El hospital elige el Modelo A. Esta decisión costará vidas.

---

## Qué Salió Mal

### La Asimetría Oculta

El conjunto de datos contiene **357 muestras benignas** y **212 malignas**—un desbalance de 1.7:1. Esto parece moderado, pero las consecuencias son severas.

![Distribución de Clases](../../figures/class_distribution.png)
*Figura 1: El desbalance de clases en nuestro dataset de cáncer de mama. Aunque 1.68:1 parece moderado, crea sesgo sistemático hacia la clase mayoritaria.*

Un "modelo" trivial que predice **benigno para cada paciente** logra:
- **63% de exactitud** (clasifica correctamente todos los casos benignos)
- **0% de recall** en tumores malignos
- **Falla en detectar el 100% de los cánceres**

Esto expone el problema central: la exactitud trata todos los errores como iguales.

### Por Qué la Exactitud Miente

| Tipo de Error | Qué Ocurrió | Consecuencia |
|---------------|-------------|--------------|
| Falso Positivo | Benigno → Maligno | Biopsia innecesaria, ansiedad del paciente |
| Falso Negativo | Maligno → Benigno | Tratamiento retrasado, metástasis, muerte |

En el tamizaje médico, **los falsos negativos matan pacientes**. Un modelo que optimiza la exactitud minimizará los errores totales sesgándose hacia la clase mayoritaria, fallando sistemáticamente en detectar casos raros pero críticos.

---

## La Realidad de la Función de Costo

Los costos médicos y legales son asimétricos:

| Error | Costo Estimado |
|-------|----------------|
| Falso Positivo | $500 (procedimiento de biopsia) |
| Falso Negativo | $50,000+ (retraso en tratamiento, litigios) |

Esta proporción de 100:1 es conservadora. Las demandas por muerte injusta debido a diagnósticos fallidos regularmente exceden $1M.

### Comparación de Modelos Bajo Costos Reales

Comparamos tres enfoques: un SVM naive, un SVM ponderado con balanceo de clases, y remuestreo SMOTE con SVM.

![Comparación de Costos](../../figures/cost_comparison.png)
*Figura 2: Desglose del costo total por modelo. El SVM Ponderado logra el menor costo total minimizando falsos negativos.*

| Modelo | Exactitud | Falsos Negativos | Falsos Positivos | Costo Total |
|--------|-----------|------------------|------------------|-------------|
| SVM Naive | 97.4% | 3 | 0 | $150,000 |
| SVM Ponderado | 98.2% | 1 | 1 | $50,500 |
| SMOTE + SVM | 97.4% | 2 | 1 | $100,500 |

**El SVM Ponderado ahorra $99,500 por cada 100 pacientes comparado con el modelo Naive.**

---

## Visualizando las Fallas del Modelo

### Matrices de Confusión: Donde se Ocultan los Errores

La matriz de confusión revela lo que la exactitud oculta:

![Matrices de Confusión](../../figures/confusion_matrices_comparison.png)
*Figura 3: Matrices de confusión lado a lado para los tres modelos. El SVM Naive (izquierda) muestra 3 falsos negativos—tres cánceres no detectados. El SVM Ponderado (centro) reduce esto a solo 1.*

Cada falso negativo en el cuadrante superior derecho representa un paciente con cáncer al que se le dijo que estaba sano.

### Análisis ROC y Precisión-Recall

![Curvas ROC](../../figures/roc_curves.png)
*Figura 4: Curvas ROC mostrando capacidad de discriminación. Todos los modelos logran alto AUC, pero esto enmascara diferencias críticas en la región sensible al costo.*

![Curvas Precisión-Recall](../../figures/precision_recall_curves.png)
*Figura 5: Curvas Precisión-Recall. Para tamizaje médico, priorizamos la región de alto recall (lado derecho de la curva), aceptando menor precisión para capturar más cánceres.*

---

## Qué Sale Mal con Cada Enfoque

### Enfoque 1: Ignorar el Desbalance

El SVM estándar optimiza la frontera de decisión para minimizar las clasificaciones erróneas totales. Con más muestras benignas, la frontera se desplaza hacia predecir benigno, sacrificando el recall en casos malignos.

**Modo de falla**: Alta exactitud, cánceres no detectados.

### Enfoque 2: Submuestreo Aleatorio

Descartar muestras de la clase mayoritaria para lograr balance.

**Modo de falla**: Con desbalance severo (ej., 380 benignos vs 17 malignos), el submuestreo descarta el 95% de los datos. El modelo carece de poder estadístico y sobreajusta a la pequeña muestra restante.

### Enfoque 3: Sobremuestreo SMOTE

Sintetizar nuevas muestras de la clase minoritaria interpolando entre las existentes.

**Modo de falla**: SMOTE asume que la clase minoritaria forma clusters convexos. Si los tumores malignos tienen múltiples subtipos distintos, la interpolación crea casos sintéticos biológicamente implausibles.

### Enfoque 4: Pesos de Clase

Ajustar la función de pérdida para penalizar más los errores de la clase minoritaria.

**Modo de éxito**: Logra alto recall sin generación de datos sintéticos, manteniendo la distribución original de datos.

---

## Validación del Modelo

### Resultados de Validación Cruzada

![Validación Cruzada](../../figures/cross_validation_recall.png)
*Figura 6: Scores de recall en validación cruzada 5-fold. El SVM Ponderado muestra recall consistentemente alto con baja varianza, indicando rendimiento robusto.*

### Curvas de Aprendizaje: Sesgo vs Varianza

![Curvas de Aprendizaje](../../figures/learning_curves.png)
*Figura 7: Curvas de aprendizaje para el SVM Ponderado. La convergencia de scores de entrenamiento y validación indica que el modelo no está sobreajustando y podría beneficiarse de datos adicionales.*

---

## Comparación Integral de Métricas

![Comparación de Métricas](../../figures/metrics_comparison.png)
*Figura 8: Comparación completa de métricas entre todos los modelos. Mientras la exactitud aparece similar, el recall (la métrica crítica para tamizaje médico) varía significativamente.*

---

## La Pregunta Correcta

El equipo de ciencia de datos preguntó: *"¿Qué modelo tiene mayor exactitud?"*

Debieron preguntar: *"¿Cuál es el costo de cada tipo de error, y cómo minimizamos el daño total?"*

### Selección de Métricas para Datos Médicos Desbalanceados

| Métrica | Qué Mide | Cuándo Usar |
|---------|----------|-------------|
| **Recall** | % de positivos reales correctamente identificados | Cuando los falsos negativos son costosos |
| **Precisión** | % de positivos predichos que son correctos | Cuando los falsos positivos son costosos |
| **F1 Score** | Media armónica de precisión y recall | Compromiso balanceado |
| **F2 Score** | F-score ponderado favoreciendo recall | Cuando el recall importa más |
| **AUC-ROC** | Capacidad de discriminación en todos los umbrales | Comparar modelos globalmente |

Para tamizaje de cáncer: **optimizar recall**, aceptar menor precisión, monitorear la tasa de falsos positivos para factibilidad operacional.

---

## Marco de Decisión

Antes de entrenar cualquier modelo con datos desbalanceados:

1. **Cuantificar la proporción de costos** de falsos negativos vs falsos positivos
2. **Establecer un umbral mínimo de recall** basado en requisitos del dominio
3. **Elegir estrategia de remuestreo** basada en características del dataset:
   - Dataset pequeño + desbalance severo → SMOTE con precaución
   - Dataset grande + desbalance moderado → pesos de clase o Tomek Links
   - Múltiples subgrupos minoritarios → variantes de SMOTE basadas en clusters
4. **Evaluar con métricas ponderadas por costo**, no exactitud cruda
5. **Reportar matrices de confusión**, no scores de un solo número

---

## Conclusión

La métrica de exactitud es peligrosa en clasificación desbalanceada porque:
- Oculta fallas sistemáticas en clases minoritarias
- Crea incentivos perversos para ignorar casos raros
- Da falsa confianza a stakeholders no familiarizados con limitaciones de ML

En diagnóstico médico, la clase minoritaria es frecuentemente la más importante. Un modelo que logra 99% de exactitud fallando en detectar el 100% de los cánceres no es un éxito—es una responsabilidad legal.

**La lección**: Las métricas codifican valores. Elige métricas que codifiquen *tus* valores, no la conveniencia del algoritmo.
