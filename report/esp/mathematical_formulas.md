# Fundamentos Matemáticos

Este documento proporciona definiciones matemáticas formales para los métodos y métricas utilizados en el caso de estudio de clasificación desbalanceada.

---

## 1. Formalización del Problema

### Clasificación Binaria

Dado un conjunto de datos $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ donde:
- $\mathbf{x}_i \in \mathbb{R}^d$ es un vector de características (d = 30 mediciones del núcleo celular)
- $y_i \in \{0, 1\}$ es la etiqueta de clase (0 = benigno, 1 = maligno)

El objetivo es aprender una función $f: \mathbb{R}^d \rightarrow \{0, 1\}$ que minimice la pérdida esperada.

### Desbalance de Clases

Sean $n_0$ y $n_1$ el número de muestras en cada clase. La proporción de desbalance es:

$$\rho = \frac{n_0}{n_1} = \frac{357}{212} \approx 1.68$$

---

## 2. Máquina de Vectores de Soporte (SVM)

### Formulación Primal

Para datos linealmente separables, SVM resuelve:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$$

sujeto a:

$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i$$

### SVM de Margen Suave

Para datos no separables, introducir variables de holgura $\xi_i \geq 0$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

sujeto a:

$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \forall i$$

donde $C > 0$ es el parámetro de regularización que controla el balance entre maximización del margen y error de clasificación.

### Formulación Dual

El problema dual es:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

sujeto a:

$$\sum_{i=1}^{n} \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C$$

### Extensión con Kernel (RBF)

Reemplazar el producto interno con función kernel:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

donde $\gamma = \frac{1}{2\sigma^2}$ controla el ancho del kernel.

Función de decisión:

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$$

donde $SV$ es el conjunto de vectores de soporte (puntos con $\alpha_i > 0$).

---

## 3. Algoritmo SMOTE

### Técnica de Sobremuestreo Sintético de Minoría

Para cada muestra de clase minoritaria $\mathbf{x}_i$:

1. Encontrar k vecinos más cercanos en clase minoritaria: $\{\mathbf{x}_{i_1}, \ldots, \mathbf{x}_{i_k}\}$

2. Seleccionar aleatoriamente un vecino $\mathbf{x}_{i_j}$

3. Generar muestra sintética:

$$\mathbf{x}_{nuevo} = \mathbf{x}_i + \lambda \cdot (\mathbf{x}_{i_j} - \mathbf{x}_i)$$

donde $\lambda \sim \text{Uniforme}(0, 1)$

### Tasa de Sobremuestreo

Para balancear clases, generar $N$ muestras sintéticas donde:

$$N = n_0 - n_1 = 357 - 212 = 145$$

Después de SMOTE: $n_0 = n_1 = 357$

---

## 4. Tomek Links

### Definición

Un par $(\mathbf{x}_i, \mathbf{x}_j)$ forma un Tomek link si:
- $y_i \neq y_j$ (clases diferentes)
- $d(\mathbf{x}_i, \mathbf{x}_j) < d(\mathbf{x}_i, \mathbf{x}_k)$ para todo $\mathbf{x}_k$ con $y_k = y_j$
- $d(\mathbf{x}_i, \mathbf{x}_j) < d(\mathbf{x}_j, \mathbf{x}_l)$ para todo $\mathbf{x}_l$ con $y_l = y_i$

donde $d(\cdot, \cdot)$ es la distancia Euclidiana.

### Estrategia de Limpieza

Eliminar muestras de clase mayoritaria que participan en Tomek links, limpiando la frontera de decisión.

---

## 5. Métricas de Evaluación

### Matriz de Confusión

$$\begin{array}{c|cc}
& \hat{y}=0 & \hat{y}=1 \\
\hline
y=0 & VN & FP \\
y=1 & FN & VP
\end{array}$$

### Exactitud

$$\text{Exactitud} = \frac{VP + VN}{VP + VN + FP + FN}$$

**Limitación**: Para datos desbalanceados, la exactitud está dominada por el rendimiento de la clase mayoritaria.

### Precisión

$$\text{Precisión} = \frac{VP}{VP + FP} = P(\text{correcto} | \text{predicho positivo})$$

### Recall (Sensibilidad, Tasa de Verdaderos Positivos)

$$\text{Recall} = \frac{VP}{VP + FN} = P(\text{detectado} | \text{realmente positivo})$$

### Especificidad (Tasa de Verdaderos Negativos)

$$\text{Especificidad} = \frac{VN}{VN + FP} = P(\text{detectado} | \text{realmente negativo})$$

### F1 Score

Media armónica de precisión y recall:

$$F_1 = 2 \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\text{Precisión} + \text{Recall}} = \frac{2VP}{2VP + FP + FN}$$

### F-beta Score

Generalización con ponderación de recall:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\beta^2 \cdot \text{Precisión} + \text{Recall}}$$

- $\beta > 1$: enfatiza recall
- $\beta < 1$: enfatiza precisión
- $\beta = 2$: recall es dos veces más importante que precisión

---

## 6. Evaluación Sensible a Costos

### Costo Esperado

Dada la matriz de costos:

$$\begin{array}{c|cc}
& \hat{y}=0 & \hat{y}=1 \\
\hline
y=0 & 0 & c_{FP} \\
y=1 & c_{FN} & 0
\end{array}$$

Costo total esperado:

$$\mathcal{L} = c_{FN} \cdot FN + c_{FP} \cdot FP$$

### Proporción de Costos

$$r = \frac{c_{FN}}{c_{FP}} = \frac{50000}{500} = 100$$

### Umbral Óptimo

Para clasificador probabilístico $P(y=1|\mathbf{x})$, el umbral óptimo por costo es:

$$\tau^* = \frac{c_{FP}}{c_{FP} + c_{FN}} = \frac{500}{500 + 50000} \approx 0.01$$

Esto sugiere clasificar como maligno cuando $P(y=1|\mathbf{x}) > 0.01$.

---

## 7. Análisis ROC

### Curva ROC

Gráfico de Tasa de Verdaderos Positivos vs Tasa de Falsos Positivos a través de todos los umbrales:

$$\text{TVP}(\tau) = \frac{VP(\tau)}{VP(\tau) + FN(\tau)}$$

$$\text{TFP}(\tau) = \frac{FP(\tau)}{FP(\tau) + VN(\tau)}$$

### AUC (Área Bajo la Curva)

$$\text{AUC} = \int_0^1 \text{TVP}(\text{TFP}^{-1}(t)) \, dt$$

Equivalente a la probabilidad de que una muestra positiva aleatoria sea rankeada más alto que una muestra negativa aleatoria:

$$\text{AUC} = P(f(\mathbf{x}^+) > f(\mathbf{x}^-))$$

---

## 8. Ponderación de Clases en SVM

### Margen Suave Ponderado

Modificar el objetivo para penalizar más los errores de clase minoritaria:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C_0 \sum_{i: y_i=0} \xi_i + C_1 \sum_{i: y_i=1} \xi_i$$

donde:

$$C_1 = C \cdot \frac{n_0}{n_1}, \quad C_0 = C$$

Esto es equivalente a `class_weight='balanced'` en scikit-learn.

---

## 9. Significancia Estadística

### Test de McNemar

Para comparar dos clasificadores en el mismo conjunto de prueba:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

donde:
- $b$ = muestras correctamente clasificadas por modelo 1 pero no por modelo 2
- $c$ = muestras correctamente clasificadas por modelo 2 pero no por modelo 1

Bajo $H_0$ (rendimiento igual), $\chi^2 \sim \chi^2_1$.

---

## Tabla Resumen

| Métrica | Fórmula | Rango | Óptimo |
|---------|---------|-------|--------|
| Exactitud | $(VP+VN)/N$ | [0, 1] | 1 |
| Precisión | $VP/(VP+FP)$ | [0, 1] | 1 |
| Recall | $VP/(VP+FN)$ | [0, 1] | 1 |
| F1 | $2PR/(P+R)$ | [0, 1] | 1 |
| AUC | $\int \text{ROC}$ | [0.5, 1] | 1 |
| Costo | $c_{FN} \cdot FN + c_{FP} \cdot FP$ | $[0, \infty)$ | 0 |
