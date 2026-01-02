# Resumen Ejecutivo: Clasificación Médica Orientada a Costos

## Problema de Negocio

Un sistema hospitalario necesitaba un modelo de ML para asistir en el tamizaje de cáncer de mama. El enfoque estándar—optimizar para exactitud—habría desplegado un modelo que sistemáticamente fallaba en detectar tumores malignos.

## Hallazgo Clave

**La exactitud es la métrica incorrecta para diagnóstico médico.**

Un modelo que predice "benigno" para cada paciente logra 63% de exactitud mientras falla en detectar el 100% de los cánceres. Cuando los costos de las clases son asimétricos, la optimización de exactitud produce resultados dañinos.

![Distribución de Clases](../../figures/class_distribution.png)
*El desbalance del dataset: 357 benignos vs 212 malignos crea sesgo oculto.*

## Resultados

| Métrica | SVM Naive | SVM Ponderado | SMOTE + SVM |
|---------|-----------|---------------|-------------|
| Exactitud | 97.4% | 98.2% | 97.4% |
| Recall | 92.9% | 97.6% | 95.2% |
| Cánceres No Detectados | 3 | 1 | 2 |
| Costo Estimado | $150,000 | $50,500 | $100,500 |

El SVM Ponderado ahorra **$99,500 por cada 100 pacientes** y detecta 2 cánceres adicionales comparado con el enfoque naive.

![Comparación de Costos](../../figures/cost_comparison.png)
*Desglose de costos: Los falsos negativos (cánceres no detectados) dominan el costo total.*

## Evidencia Visual

### Comparación de Rendimiento de Modelos

![Comparación de Métricas](../../figures/metrics_comparison.png)
*Mientras la exactitud aparece similar entre modelos, el recall—la métrica que importa para seguridad del paciente—varía significativamente.*

### Análisis de Matriz de Confusión

![Matrices de Confusión](../../figures/confusion_matrices_comparison.png)
*Las matrices de confusión revelan lo que la exactitud oculta: el SVM Naive falla en detectar 3 cánceres, mientras el SVM Ponderado solo falla en 1.*

## Enfoque Técnico

1. Identificación del desbalance de clases (proporción 1.68:1 benigno a maligno)
2. Cuantificación de costos de error asimétricos (falso negativo: $50,000 vs falso positivo: $500)
3. Comparación de tres enfoques: SVM naive, SVM ponderado, y remuestreo SMOTE
4. Evaluación de modelos usando recall y métricas ponderadas por costo en lugar de exactitud

## Validación del Modelo

![Validación Cruzada](../../figures/cross_validation_recall.png)
*Validación cruzada 5-fold confirma que el SVM Ponderado mantiene alto recall consistentemente a través de divisiones de datos.*

## Competencias Demostradas

- **Análisis de Decisiones**: Traducir matrices de confusión a costos de negocio
- **Evaluación de Riesgos**: Identificar modos de falla en pipelines estándar de ML
- **Comunicación con Stakeholders**: Explicar por qué métricas "mejores" pueden significar peores resultados
- **Conocimiento del Dominio**: Comprender implicaciones médicas y legales de errores de clasificación
- **Análisis Visual**: Crear visualizaciones convincentes que cuentan la historia de costos

## Recomendación

Para cualquier tarea de clasificación con costos de error asimétricos:
1. Definir la proporción de costos antes de la selección del modelo
2. Establecer umbrales mínimos de recall basados en requisitos del dominio
3. Reportar matrices de confusión, no scores de un solo número
4. Validar que mejoras en exactitud no vengan a costa de casos minoritarios críticos

---

*Este caso de estudio demuestra que el despliegue efectivo de ML requiere entender el contexto de negocio, no solo optimizar métricas.*
