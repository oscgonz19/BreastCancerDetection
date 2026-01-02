# Marco Teórico: Clasificación Desbalanceada en Diagnóstico Médico

Una referencia académica completa que cubre los fundamentos estadísticos y de aprendizaje automático necesarios para comprender la clasificación con conciencia de costos en conjuntos de datos desbalanceados.

---

## Tabla de Contenidos

1. [Teoría del Aprendizaje Estadístico](#1-teoría-del-aprendizaje-estadístico)
2. [Clasificación Binaria](#2-clasificación-binaria)
3. [Métricas de Evaluación y sus Limitaciones](#3-métricas-de-evaluación-y-sus-limitaciones)
4. [El Problema del Desbalance de Clases](#4-el-problema-del-desbalance-de-clases)
5. [Máquinas de Vectores de Soporte](#5-máquinas-de-vectores-de-soporte)
6. [Métodos de Remuestreo](#6-métodos-de-remuestreo)
7. [Aprendizaje Sensible a Costos](#7-aprendizaje-sensible-a-costos)
8. [Teoría de Decisión y Riesgo](#8-teoría-de-decisión-y-riesgo)
9. [Selección de Modelos y Validación](#9-selección-de-modelos-y-validación)
10. [Referencias](#10-referencias)

---

## 1. Teoría del Aprendizaje Estadístico

### 1.1 El Problema del Aprendizaje

La teoría del aprendizaje estadístico proporciona los fundamentos matemáticos del aprendizaje automático. El problema fundamental se puede enunciar de la siguiente manera:

Dado un conjunto de entrenamiento $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$ extraído independientemente de una distribución de probabilidad desconocida $P(\mathbf{X}, Y)$, encontrar una función $f: \mathcal{X} \rightarrow \mathcal{Y}$ que minimice el riesgo esperado.

### 1.2 Riesgo y Funciones de Pérdida

El **riesgo esperado** (o riesgo verdadero) de una hipótesis $f$ se define como:

$$R(f) = \mathbb{E}_{(\mathbf{x}, y) \sim P}[L(y, f(\mathbf{x}))]$$

donde $L(y, \hat{y})$ es una función de pérdida que mide el costo de predecir $\hat{y}$ cuando la etiqueta verdadera es $y$.

Funciones de pérdida comunes incluyen:

| Función de Pérdida | Fórmula | Caso de Uso |
|--------------------|---------|-------------|
| Pérdida 0-1 | $L(y, \hat{y}) = \mathbb{1}[y \neq \hat{y}]$ | Clasificación |
| Pérdida Cuadrática | $L(y, \hat{y}) = (y - \hat{y})^2$ | Regresión |
| Pérdida Hinge | $L(y, \hat{y}) = \max(0, 1 - y\hat{y})$ | SVM |
| Pérdida Logarítmica | $L(y, \hat{y}) = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})$ | Regresión Logística |

### 1.3 Minimización del Riesgo Empírico

Como $P(\mathbf{X}, Y)$ es desconocida, aproximamos el riesgo esperado con el **riesgo empírico**:

$$\hat{R}(f) = \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(\mathbf{x}_i))$$

El principio de **Minimización del Riesgo Empírico (MRE)** establece que debemos elegir la hipótesis que minimiza el riesgo empírico:

$$f^* = \arg\min_{f \in \mathcal{H}} \hat{R}(f)$$

donde $\mathcal{H}$ es el espacio de hipótesis.

### 1.4 Compensación Sesgo-Varianza

El error de predicción esperado puede descomponerse en tres componentes:

$$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \text{Sesgo}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2$$

donde:
- **Sesgo**: Error por suposiciones erróneas en el algoritmo de aprendizaje
- **Varianza**: Error por sensibilidad a fluctuaciones en el conjunto de entrenamiento
- **Error irreducible** ($\sigma^2$): Ruido inherente al problema

Esta compensación es fundamental: modelos complejos tienen bajo sesgo pero alta varianza (sobreajuste), mientras que modelos simples tienen alto sesgo pero baja varianza (subajuste).

### 1.5 Generalización y Dimensión VC

La **dimensión Vapnik-Chervonenkis (VC)** mide la capacidad de un espacio de hipótesis. Para un espacio de hipótesis $\mathcal{H}$, la dimensión VC es el mayor número de puntos que pueden ser destrozados (separados perfectamente) por $\mathcal{H}$.

La cota de generalización establece que con probabilidad al menos $1 - \delta$:

$$R(f) \leq \hat{R}(f) + \sqrt{\frac{h(\log(2n/h) + 1) - \log(\delta/4)}{n}}$$

donde $h$ es la dimensión VC y $n$ es el tamaño de la muestra.

---

## 2. Clasificación Binaria

### 2.1 Definición del Problema

En clasificación binaria, tenemos:
- Espacio de entrada: $\mathcal{X} \subseteq \mathbb{R}^d$
- Espacio de salida: $\mathcal{Y} = \{0, 1\}$ o $\{-1, +1\}$
- Objetivo: Aprender $f: \mathcal{X} \rightarrow \mathcal{Y}$

### 2.2 Fronteras de Decisión

Un clasificador particiona el espacio de entrada en regiones de decisión. La **frontera de decisión** es la superficie donde:

$$P(Y = 1 | \mathbf{X} = \mathbf{x}) = P(Y = 0 | \mathbf{X} = \mathbf{x}) = 0.5$$

Para clasificadores lineales, esta frontera es un hiperplano:

$$\mathbf{w}^T \mathbf{x} + b = 0$$

### 2.3 Clasificación Probabilística

Muchos clasificadores producen estimaciones de probabilidad $\hat{p}(\mathbf{x}) = P(Y = 1 | \mathbf{X} = \mathbf{x})$. La predicción final se realiza por umbralización:

$$\hat{y} = \begin{cases} 1 & \text{si } \hat{p}(\mathbf{x}) \geq \tau \\ 0 & \text{en otro caso} \end{cases}$$

El umbral $\tau$ (típicamente 0.5) puede ajustarse para compensar entre diferentes tipos de errores.

### 2.4 Clasificador Óptimo de Bayes

El **clasificador óptimo de Bayes** minimiza la pérdida 0-1 esperada:

$$f^*(\mathbf{x}) = \arg\max_{y \in \{0, 1\}} P(Y = y | \mathbf{X} = \mathbf{x})$$

Este clasificador logra la tasa de error más baja posible, llamada **tasa de error de Bayes**:

$$R^* = \mathbb{E}_{\mathbf{x}}[\min(P(Y=1|\mathbf{x}), P(Y=0|\mathbf{x}))]$$

---

## 3. Métricas de Evaluación y sus Limitaciones

### 3.1 La Matriz de Confusión

Para clasificación binaria, las predicciones caen en cuatro categorías:

|  | Predicho Negativo | Predicho Positivo |
|--|-------------------|-------------------|
| **Real Negativo** | Verdadero Negativo (VN) | Falso Positivo (FP) |
| **Real Positivo** | Falso Negativo (FN) | Verdadero Positivo (VP) |

### 3.2 Métricas Básicas

**Exactitud**:
$$\text{Exactitud} = \frac{VP + VN}{VP + VN + FP + FN}$$

**Tasa de Error**:
$$\text{Tasa de Error} = 1 - \text{Exactitud} = \frac{FP + FN}{VP + VN + FP + FN}$$

### 3.3 Métricas Específicas por Clase

**Precisión** (Valor Predictivo Positivo):
$$\text{Precisión} = \frac{VP}{VP + FP} = P(\text{Real Positivo} | \text{Predicho Positivo})$$

**Recall** (Sensibilidad, Tasa de Verdaderos Positivos):
$$\text{Recall} = \frac{VP}{VP + FN} = P(\text{Predicho Positivo} | \text{Real Positivo})$$

**Especificidad** (Tasa de Verdaderos Negativos):
$$\text{Especificidad} = \frac{VN}{VN + FP} = P(\text{Predicho Negativo} | \text{Real Negativo})$$

**Tasa de Falsos Positivos**:
$$\text{TFP} = \frac{FP}{FP + VN} = 1 - \text{Especificidad}$$

### 3.4 Métricas Combinadas

**F1 Score** (Media armónica de precisión y recall):
$$F_1 = 2 \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\text{Precisión} + \text{Recall}} = \frac{2VP}{2VP + FP + FN}$$

**F-beta Score** (Media armónica ponderada):
$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\beta^2 \cdot \text{Precisión} + \text{Recall}}$$

- $\beta > 1$: Enfatiza recall
- $\beta < 1$: Enfatiza precisión
- $\beta = 2$: Recall es dos veces más importante que precisión

**Coeficiente de Correlación de Matthews**:
$$\text{MCC} = \frac{VP \cdot VN - FP \cdot FN}{\sqrt{(VP+FP)(VP+FN)(VN+FP)(VN+FN)}}$$

MCC varía de -1 (clasificación errónea perfecta) a +1 (clasificación perfecta), con 0 indicando rendimiento aleatorio.

### 3.5 Métricas Independientes del Umbral

**Curva ROC**: Gráfico de TVP vs TFP a través de todos los umbrales.

**AUC-ROC** (Área Bajo la Curva ROC):
$$\text{AUC} = \int_0^1 \text{TVP}(\text{TFP}^{-1}(t)) \, dt$$

Interpretación probabilística:
$$\text{AUC} = P(\hat{p}(\mathbf{x}^+) > \hat{p}(\mathbf{x}^-))$$

donde $\mathbf{x}^+$ es una muestra positiva aleatoria y $\mathbf{x}^-$ es una muestra negativa aleatoria.

**Curva Precisión-Recall**: Gráfico de Precisión vs Recall a través de todos los umbrales.

**Precisión Promedio**:
$$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$$

### 3.6 Por Qué la Exactitud Falla con Datos Desbalanceados

Considere un conjunto de datos con 95% de muestras negativas y 5% positivas. Un clasificador trivial que predice "negativo" para todas las entradas logra:

- Exactitud: 95%
- Precisión: indefinida (0/0)
- Recall: 0%
- F1: 0%

**Insight clave**: La exactitud está dominada por el rendimiento de la clase mayoritaria y puede ocultar un fallo completo en la clase minoritaria.

---

## 4. El Problema del Desbalance de Clases

### 4.1 Definición y Prevalencia

El desbalance de clases ocurre cuando la distribución de clases está significativamente sesgada:

$$\frac{n_{\text{mayoría}}}{n_{\text{minoría}}} >> 1$$

Proporciones de desbalance en aplicaciones reales:
- Diagnóstico médico: 10:1 a 1000:1
- Detección de fraude: 100:1 a 10000:1
- Detección de defectos: 50:1 a 500:1

### 4.2 Por Qué el Desbalance Causa Problemas

Los algoritmos de aprendizaje estándar asumen distribuciones de clases balanceadas. Con datos desbalanceados:

1. **Cambio de probabilidad a priori**: El clasificador aprende a favorecer la clase mayoritaria
2. **Distorsión de frontera**: Las fronteras de decisión se desplazan hacia la clase minoritaria
3. **Engaño métrico**: La exactitud parece alta mientras el rendimiento de la clase minoritaria es pobre
4. **Representación insuficiente**: Muy pocas muestras minoritarias para aprender la distribución subyacente

### 4.3 Tipos de Desbalance

**Desbalance entre clases**: Número desigual de muestras por clase.

**Desbalance dentro de clase**: La clase minoritaria contiene múltiples sub-conceptos con representación desigual.

**Desbalance relativo** vs **Rareza absoluta**: Una proporción 100:1 con 10,000 muestras minoritarias es diferente de 100:1 con 100 muestras minoritarias.

### 4.4 El Problema de los Pequeños Disjuntos

En conjuntos de datos desbalanceados, las muestras de clase minoritaria frecuentemente forman clusters pequeños y desconectados (**disjuntos**). Estos son difíciles de aprender porque:

- El tamaño pequeño de muestra lleva a estimaciones poco confiables
- Los clasificadores pueden tratarlos como ruido
- Las tasas de error en pequeños disjuntos son desproporcionadamente altas

---

## 5. Máquinas de Vectores de Soporte

### 5.1 SVM Lineal: Clasificador de Margen Máximo

Dados datos linealmente separables, SVM encuentra el hiperplano que maximiza el margen entre clases.

**Formulación primal**:
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$$

sujeto a:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n$$

El ancho del margen es $\frac{2}{\|\mathbf{w}\|}$, así que minimizar $\|\mathbf{w}\|^2$ maximiza el margen.

### 5.2 SVM de Margen Suave

Para datos no separables, introducir variables de holgura $\xi_i \geq 0$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

sujeto a:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

El parámetro $C$ controla la compensación:
- $C$ grande: Menos tolerancia a clasificación errónea (puede sobreajustar)
- $C$ pequeño: Más tolerancia a clasificación errónea (puede subajustar)

### 5.3 Formulación Dual

Usando dualidad Lagrangiana, el problema se convierte en:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

sujeto a:
$$\sum_{i=1}^{n} \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C$$

La solución depende solo de productos internos $\mathbf{x}_i^T \mathbf{x}_j$, habilitando el truco del kernel.

### 5.4 Métodos de Kernel

Reemplazar productos internos con una función kernel $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$:

**Kernel lineal**:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$$

**Kernel polinomial**:
$$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$$

**Kernel de Función de Base Radial (RBF)**:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

El kernel RBF mapea datos a un espacio de dimensión infinita, permitiendo fronteras de decisión complejas.

### 5.5 Función de Decisión

La decisión de clasificación es:

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$$

donde $SV$ es el conjunto de vectores de soporte (muestras con $\alpha_i > 0$).

### 5.6 SVM y Desbalance de Clases

La optimización estándar de SVM trata todos los errores igualmente. Con datos desbalanceados:
- El margen se empuja hacia la clase minoritaria
- Las muestras de clase minoritaria son más propensas a ser clasificadas erróneamente
- El clasificador favorece la clase mayoritaria

**Soluciones**:
1. Diferentes costos de clasificación errónea por clase
2. Remuestreo antes del entrenamiento
3. Ajustar el umbral de decisión

---

## 6. Métodos de Remuestreo

### 6.1 Visión General de Enfoques

Los métodos de remuestreo modifican los datos de entrenamiento para abordar el desbalance de clases:

| Enfoque | Método | Efecto |
|---------|--------|--------|
| Sobremuestreo | Agregar muestras minoritarias | Aumenta representación minoritaria |
| Submuestreo | Eliminar muestras mayoritarias | Reduce dominancia mayoritaria |
| Híbrido | Combinar ambos | Balancea compensaciones |

### 6.2 Sobremuestreo Aleatorio

Duplicar aleatoriamente muestras de clase minoritaria hasta que las clases estén balanceadas.

**Algoritmo**:
1. Calcular la diferencia: $N = n_{\text{mayoría}} - n_{\text{minoría}}$
2. Seleccionar aleatoriamente $N$ muestras de clase minoritaria (con reemplazo)
3. Agregar copias al conjunto de entrenamiento

**Ventajas**: Simple, preserva información

**Desventajas**: Copias exactas llevan a sobreajuste; las fronteras de decisión se vuelven demasiado específicas

### 6.3 Submuestreo Aleatorio

Eliminar aleatoriamente muestras de clase mayoritaria hasta que las clases estén balanceadas.

**Algoritmo**:
1. Calcular objetivo: $N = n_{\text{minoría}}$
2. Seleccionar aleatoriamente $N$ muestras de clase mayoritaria
3. Descartar muestras mayoritarias restantes

**Ventajas**: Reduce tiempo de entrenamiento, puede eliminar muestras ruidosas

**Desventajas**: Pierde información potencialmente útil; puede descartar muestras mayoritarias importantes cerca de la frontera

### 6.4 SMOTE (Técnica de Sobremuestreo Sintético de Minoría)

SMOTE genera muestras minoritarias sintéticas por interpolación.

**Algoritmo**:
```
Para cada muestra minoritaria x_i:
    1. Encontrar k vecinos minoritarios más cercanos
    2. Seleccionar aleatoriamente un vecino x_j
    3. Generar muestra sintética:
       x_nuevo = x_i + λ(x_j - x_i)
       donde λ ~ Uniforme(0, 1)
```

**Formulación matemática**:
$$\mathbf{x}_{\text{nuevo}} = \mathbf{x}_i + \lambda \cdot (\mathbf{x}_j - \mathbf{x}_i), \quad \lambda \in [0, 1]$$

**Ventajas**:
- Crea nuevas muestras plausibles
- Reduce sobreajuste comparado con sobremuestreo aleatorio
- Expande la región de clase minoritaria

**Desventajas**:
- Puede crear muestras ruidosas si la clase minoritaria no es convexa
- Puede generar muestras en regiones de clase mayoritaria (superposición)
- Asume que el espacio de características es significativo para interpolación

### 6.5 Variantes de SMOTE

**Borderline-SMOTE**: Solo sobremuestrear muestras minoritarias cerca de la frontera de decisión.

**SMOTE-ENN**: Aplicar SMOTE, luego limpiar con Vecinos Más Cercanos Editados.

**ADASYN** (Muestreo Sintético Adaptativo): Generar más muestras sintéticas para instancias minoritarias más difíciles de clasificar.

### 6.6 Tomek Links

Un Tomek link es un par de muestras $(\mathbf{x}_i, \mathbf{x}_j)$ donde:
- $y_i \neq y_j$ (clases diferentes)
- $d(\mathbf{x}_i, \mathbf{x}_j) < d(\mathbf{x}_i, \mathbf{x}_k)$ para todo $k$ con $y_k = y_j$
- $d(\mathbf{x}_i, \mathbf{x}_j) < d(\mathbf{x}_j, \mathbf{x}_l)$ para todo $l$ con $y_l = y_i$

En palabras: cada muestra es el vecino más cercano del otro de la clase opuesta.

**Uso**:
- Eliminar muestras mayoritarias en Tomek links (limpia frontera)
- Eliminar ambas muestras (elimina regiones ambiguas)

**Limitación**: Solo afecta muestras de frontera; no aborda el desbalance central.

### 6.7 Vecinos Más Cercanos Editados (ENN)

Eliminar muestras cuya clase difiere de la mayoría de sus k vecinos más cercanos.

**Algoritmo**:
```
Para cada muestra (x_i, y_i):
    Encontrar k vecinos más cercanos
    Si mayoría de vecinos tienen clase ≠ y_i:
        Eliminar (x_i, y_i)
```

Esto limpia muestras ruidosas y de frontera de ambas clases.

### 6.8 Métodos Híbridos

**SMOTE + Tomek Links**: Aplicar SMOTE, luego eliminar Tomek links para limpiar la frontera.

**SMOTE + ENN**: Aplicar SMOTE, luego aplicar ENN para eliminar muestras sintéticas ruidosas.

---

## 7. Aprendizaje Sensible a Costos

### 7.1 Motivación

En muchas aplicaciones, diferentes tipos de errores tienen diferentes costos:

| Aplicación | Error Costoso | Error Menos Costoso |
|------------|---------------|---------------------|
| Tamizaje de cáncer | Falso Negativo (cáncer no detectado) | Falso Positivo (biopsia innecesaria) |
| Filtrado de spam | Falso Positivo (correo perdido) | Falso Negativo (spam en bandeja) |
| Detección de fraude | Falso Negativo (fraude no detectado) | Falso Positivo (transacción bloqueada) |

### 7.2 Matriz de Costos

Definir costos para cada resultado:

|  | Predicho Negativo | Predicho Positivo |
|--|-------------------|-------------------|
| **Real Negativo** | $C_{VN}$ (usualmente 0) | $C_{FP}$ |
| **Real Positivo** | $C_{FN}$ | $C_{VP}$ (usualmente 0) |

### 7.3 Costo Esperado

El costo esperado de un clasificador es:

$$\mathbb{E}[\text{Costo}] = C_{FP} \cdot FP + C_{FN} \cdot FN + C_{VP} \cdot VP + C_{VN} \cdot VN$$

Con $C_{VP} = C_{VN} = 0$:

$$\mathbb{E}[\text{Costo}] = C_{FP} \cdot FP + C_{FN} \cdot FN$$

### 7.4 Enfoques de Aprendizaje Sensible a Costos

**1. Remuestreo para reflejar costos**:

Sobremuestrear clase minoritaria por factor proporcional a proporción de costos:
$$\text{Tasa de sobremuestreo} = \frac{C_{FN}}{C_{FP}}$$

**2. Algoritmos sensibles a costos**:

Modificar el algoritmo de aprendizaje para incorporar costos directamente:
$$\min_{\mathbf{w}} \sum_{i: y_i = 0} C_{FP} \cdot L_i + \sum_{i: y_i = 1} C_{FN} \cdot L_i$$

**3. Ajuste de umbral**:

Para clasificadores probabilísticos, el umbral óptimo por costo es:
$$\tau^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$

### 7.5 Ponderación de Clases en SVM

Modificar el objetivo de margen suave:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C_0 \sum_{i: y_i=0} \xi_i + C_1 \sum_{i: y_i=1} \xi_i$$

Establecer $C_1 > C_0$ penaliza los falsos negativos más fuertemente.

Heurística común (ponderación balanceada):
$$C_k = C \cdot \frac{n}{2 \cdot n_k}$$

donde $n$ es el total de muestras y $n_k$ es muestras en clase $k$.

---

## 8. Teoría de Decisión y Riesgo

### 8.1 Teoría de Decisión Estadística

Una **regla de decisión** $\delta: \mathcal{X} \rightarrow \mathcal{A}$ mapea observaciones a acciones. En clasificación:
- Observaciones: Vectores de características $\mathbf{x}$
- Acciones: Predicciones de clase $\{0, 1\}$

### 8.2 Funciones de Pérdida y Riesgo

La **función de pérdida** $L(y, a)$ cuantifica el costo de tomar acción $a$ cuando el estado verdadero es $y$.

El **riesgo** de una regla de decisión es su pérdida esperada:
$$R(\delta) = \mathbb{E}[L(Y, \delta(\mathbf{X}))]$$

### 8.3 Riesgo de Bayes y Decisiones Óptimas

Para una distribución a priori dada $P(Y)$ y verosimilitud $P(\mathbf{X}|Y)$, la **decisión óptima de Bayes** minimiza la pérdida esperada:

$$\delta^*(\mathbf{x}) = \arg\min_a \sum_y L(y, a) P(Y = y | \mathbf{X} = \mathbf{x})$$

El **riesgo de Bayes** es el riesgo mínimo alcanzable:
$$R^* = \mathbb{E}[L(Y, \delta^*(\mathbf{X}))]$$

### 8.4 Decisión Óptima de Bayes Sensible a Costos

Con costos asimétricos, la regla de decisión óptima es:

Predecir $Y = 1$ si:
$$\frac{P(Y = 1 | \mathbf{x})}{P(Y = 0 | \mathbf{x})} > \frac{C_{FP}}{C_{FN}}$$

Equivalentemente:
$$P(Y = 1 | \mathbf{x}) > \frac{C_{FP}}{C_{FP} + C_{FN}}$$

### 8.5 Análisis de Riesgo en Diagnóstico Médico

Para tamizaje de cáncer con:
- $C_{FN} = \$50,000$ (cáncer no detectado)
- $C_{FP} = \$500$ (biopsia innecesaria)

La proporción de costos es $100:1$, así que el umbral óptimo es:
$$\tau^* = \frac{500}{500 + 50000} = 0.0099 \approx 1\%$$

Esto significa: predecir maligno si $P(\text{maligno}|\mathbf{x}) > 1\%$.

---

## 9. Selección de Modelos y Validación

### 9.1 El Método Holdout

Dividir datos en conjuntos de entrenamiento y prueba:
- Conjunto de entrenamiento: Usado para ajustar el modelo
- Conjunto de prueba: Usado para estimar rendimiento de generalización

**Limitación**: Desperdicia datos; la estimación tiene alta varianza con conjuntos pequeños.

### 9.2 Validación Cruzada K-Fold

1. Particionar datos en $K$ pliegues iguales
2. Para $k = 1, \ldots, K$:
   - Entrenar en pliegues $\{1, \ldots, K\} \setminus \{k\}$
   - Probar en pliegue $k$
3. Promediar rendimiento a través de pliegues

**Estimación CV K-fold**:
$$\hat{R}_{CV} = \frac{1}{K} \sum_{k=1}^{K} \hat{R}_k$$

Opciones comunes: $K = 5$ o $K = 10$.

### 9.3 Validación Cruzada Estratificada

Para datos desbalanceados, usar muestreo **estratificado** para preservar proporciones de clase en cada pliegue.

Sin estratificación, algunos pliegues pueden tener muy pocas o ninguna muestra minoritaria.

### 9.4 Validación Cruzada Anidada

Para selección de modelo con ajuste de hiperparámetros:

**Bucle externo**: Estimar rendimiento de generalización
**Bucle interno**: Seleccionar hiperparámetros

Esto previene fuga de información del conjunto de prueba a la selección de modelo.

### 9.5 Comparación Estadística de Clasificadores

**Test de McNemar**: Comparar dos clasificadores en el mismo conjunto de prueba.

$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$

donde:
- $n_{01}$: Muestras correctas por clasificador 1, incorrectas por clasificador 2
- $n_{10}$: Muestras correctas por clasificador 2, incorrectas por clasificador 1

Bajo $H_0$ (rendimiento igual): $\chi^2 \sim \chi^2_1$

### 9.6 Calibración

Un clasificador está **bien calibrado** si:
$$P(Y = 1 | \hat{p}(\mathbf{x}) = p) = p$$

La calibración puede evaluarse usando:
- Diagramas de confiabilidad
- Error de Calibración Esperado (ECE)
- Score de Brier

**Escalamiento de Platt** y **regresión isotónica** son métodos comunes de calibración post-hoc.

---

## 10. Referencias

### Textos Fundamentales

1. Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory*. Springer.

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2da ed.). Springer.

3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

### Desbalance de Clases

4. He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.

5. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

6. Tomek, I. (1976). Two Modifications of CNN. *IEEE Transactions on Systems, Man, and Cybernetics*, 6(11), 769-772.

### Aprendizaje Sensible a Costos

7. Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. *Proceedings of the 17th International Joint Conference on Artificial Intelligence*, 973-978.

8. Domingos, P. (1999). MetaCost: A General Method for Making Classifiers Cost-Sensitive. *Proceedings of the 5th International Conference on Knowledge Discovery and Data Mining*, 155-164.

### Máquinas de Vectores de Soporte

9. Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273-297.

10. Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.

### Métricas de Evaluación

11. Powers, D. M. (2011). Evaluation: From Precision, Recall and F-measure to ROC, Informedness, Markedness and Correlation. *Journal of Machine Learning Technologies*, 2(1), 37-63.

12. Davis, J., & Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. *Proceedings of the 23rd International Conference on Machine Learning*, 233-240.

---

## Resumen

Este marco teórico establece que:

1. **La teoría del aprendizaje estadístico** proporciona cotas sobre generalización pero asume ciertas funciones de pérdida
2. **La exactitud no es una métrica universal**—codifica la suposición de que todos los errores son igualmente costosos
3. **El desbalance de clases** rompe suposiciones estándar y requiere técnicas especializadas
4. **Los métodos de remuestreo** (SMOTE, Tomek Links) abordan el desbalance a nivel de datos
5. **El aprendizaje sensible a costos** aborda el desbalance a nivel de algoritmo
6. **La teoría de decisión** proporciona el marco formal para elegir métricas alineadas con costos del mundo real
7. **La selección de modelos** debe usar validación estratificada y métricas apropiadas

El insight clave es que **la selección de métricas no es un detalle técnico—codifica valores y prioridades** que deben alinearse con el dominio de aplicación.
