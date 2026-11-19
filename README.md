# Machine Learning


# Introducción al Machine Learning 

Conceptos fundamentales, aplicaciones, tipos de algoritmos y herramientas clave del **Machine Learning**.

---

## 1. Definición y Concepto
> *"Machine Learning es el subcampo de las ciencias de la computación y una rama de la Inteligencia Artificial, cuyo objetivo es simular el aprendizaje automático, gracias a modelos matemáticos, que nos sirven para detectar patrones en los datos y realizar predicciones."*

### Ejemplos Prácticos
* **Banca (Riesgo Crediticio):** Evaluación de variables históricas (dinero en cuenta, historial de impagos, situación laboral, estado civil y deudas) para predecir si un nuevo cliente será buen o mal pagador.
* **Consumo de Combustible (Modelado Matemático):** 
  * *Modelo simple:* Relación lineal entre distancia y gasolina consumida (`y = w₁x₁ + w₀`).
  * *Modelo multivariable:* Incorporación de nuevas variables (*features*) como tipo de coche, número de personas, equipaje y altura recorrida (`y = w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + w₅x₅ + w₀`).

---

## 2. Aplicaciones por Sector

| Sector | Casos de Uso Principales |
| :--- | :--- |
| **Banca y Seguros** | Detección de fraude, anticipación al impago, productos personalizados. |
| **Marketing** | Comportamiento del cliente, análisis de sentimientos en campañas, experiencia de usuario. |
| **Retail y Logística** | Patrones de compra, anticipación de entregas/demandas, predicción de averías. |
| **Sector Salud** | Clasificadores de radiografías, predicción de enfermedades, gestión de saturación hospitalaria. |
| **Redes Sociales y Media** | Recomendadores de productos y contenido multimedia (ej. Netflix, Spotify). |
| **Mercados** | Detección de tendencias, anomalías y predicción en mercados energéticos y de valores. |

---

## 3. Tipos de Algoritmos de Machine Learning

### A. Aprendizaje Supervisado (*Supervised Learning*) - ~80% de los casos
Trabaja con **datos etiquetados**. El algoritmo aprende de un histórico de características (*features*) donde la variable objetivo (*target*) es conocida.
* **Regresión (Predicción de un número):** Proyección de ventas, estimación de parámetros para evitar averías.
* **Clasificación (Predicción de una clase/categoría):** Diagnóstico por imágenes médicas, análisis de texto en redes sociales, detección de *default* bancario.

### B. Aprendizaje No Supervisado (*Unsupervised Learning*)
Trabaja con **datos sin etiquetar**. No existe una variable objetivo previa; el algoritmo explora y encuentra patrones por sí mismo.
* **Clustering:** Agrupación por similitudes (segmentación de clientes, clasificación biológica).
* **Detección de anomalías:** Identificación de patrones atípicos (fraudes, intrusiones en red).
* **Reducción de dimensionalidad:** Técnicas previas a modelos supervisados para reducir el número de *features* y facilitar la visualización.

### C. Aprendizaje por Refuerzo (*Reinforcement Learning*)
Algoritmos (agentes) que aprenden interactuando con un entorno mediante un sistema de acciones, estados y recompensas.
* **Aplicaciones:** Conducción autónoma, navegación robótica, IA en videojuegos, recomendadores de noticias y Procesamiento de Lenguaje Natural (NLP).

---

## 4. Jerarquía: IA vs. Machine Learning vs. Deep Learning

* **Inteligencia Artificial (IA):** Conjunto de algoritmos que buscan imitar los comportamientos cognitivos del ser humano (chatbots, asistentes virtuales, reconocimiento de voz/imagen, vehículos autónomos).
* **Machine Learning (ML):** Rama de la IA centrada en modelos matemáticos y estadísticos que aprenden de los datos sin programación explícita de reglas. Se apoya en la intersección de Matemáticas/Estadística, Ciencias de la Computación y Conocimiento del Negocio (*Data Science*).
* **Deep Learning (DL):** Subconjunto del ML basado en redes neuronales artificiales multicapa inspiradas en el cerebro humano. Sobresale especialmente en tareas complejas como la visión artificial (*Computer Vision*).

---

## 5. Ejemplos Reales y Limitaciones Actuales

* **Casos de Éxito:**
  * *Asistentes y Chatbots:* Bots conversacionales para reservas de vuelos (ej. Skyscanner).
  * *Computer Vision:* Detección de vehículos en carretera y herramientas de accesibilidad para describir imágenes a personas con discapacidad visual.
  * *Robótica:* Desarrollo de robots avanzados como *Sophia* y *Handle* (Boston Dynamics).
* **Limitaciones y Retos:** Es una tecnología joven con margen de madurez. Existen precedentes de fallos importantes, como accidentes con sistemas de piloto automático (Tesla), sesgos aprendidos rápidamente por chatbots en redes sociales (*Microsoft Tay* en Twitter) o fallos de comprensión en asistentes de voz (Alexa).

---

## 6. Ecosistema de Tecnologías y Herramientas

* **Lenguajes y Entornos:** Python, R, Jupyter.
* **Manipulación de Datos y ML Clásico:** Pandas, NumPy, Scikit-learn.
* **Frameworks de Deep Learning:** TensorFlow, PyTorch, Keras.
* **Cloud Computing:** Google Cloud Platform (GCP), Amazon Web Services (AWS).
* **Visualización y BI:** Tableau, Google Data Studio, Google Sheets.








<hr>






# Machine Learning: Conceptos y Aprendizaje Supervisado

Conceptos esenciales sobre la creación, evaluación, optimización y validación de modelos de **Aprendizaje Supervisado** en Machine Learning.

---

## 1. ¿Qué es un Modelo de Machine Learning?
* **Definición:** Es un conjunto de fórmulas matemáticas (`f(x)`) que expresan relaciones y patrones entre variables, permitiendo estudiar comportamientos complejos y traducir relaciones de datos de la vida real a las matemáticas.
* **Funcionamiento:** A partir de un conjunto de datos de entrada (*inputs*), el modelo predice un resultado o salida (*output*).
* **Ejemplos de aplicación:**
  * Predicción de predisposición al cáncer de pulmón según edad, tabaquismo y enfermedades previas.
  * Evaluación de riesgo bancario (buen o mal pagador).
  * Recomendación de compra de productos según gustos y búsquedas en redes sociales.

---

## 2. Creación del Modelo y Terminología Clave
Para construir un modelo se necesita un **histórico de datos** que sirva de muestra para aprender patrones (de forma similar a como un alumno estudia ejercicios resueltos antes de un examen). Una vez entrenado, el modelo recibe casos nuevos y genera predicciones.

### Terminología Fundamental (`X` e `Y`)
* **Features (`X`):** Son todas las variables de entrada (*inputs*) utilizadas para alimentar el modelo (ej. *Fumador*, *Edad*).
* **Target (`Y`):** Es la variable objetivo a predecir, el *output* del modelo. Por norma general es una única variable, ya sea numérica o categórica (ej. *¿Ha tenido cáncer?*).

---

## 3. Evaluación y Selección del Mejor Modelo
A priori no sabemos qué algoritmo funcionará mejor con nuestros datos, por lo que aplicamos el **método científico**: probar varios modelos y quedarnos con aquel que **minimice los errores** de predicción (por ejemplo, comparando una regresión lineal con regresiones polinómicas de distintos grados).

### El Objetivo Real: Generalizar
* **El problema:** Un modelo puede tener un error muy bajo con los datos que ya conoce, pero disparar su error al enfrentarse a datos reales en producción porque no sabe **generalizar**.
* **La solución (Train / Test Split):** Dividir el dataset original antes de entrenar:
  * **Train (~70-85%, habitualmente 80%):** Datos usados para entrenar los distintos modelos.
  * **Test (~15-30%, habitualmente 20%):** Datos completamente ajenos al entrenamiento que se reservan para evaluar cómo generaliza el modelo ante casos nuevos.

---

## 4. Overfitting, Underfitting y el Equilibrio Bias-Variance

### Overfitting (Sobreajuste)
* **Qué es:** Ocurre cuando un modelo es excesivamente complejo y "memoriza" los datos de entrenamiento en lugar de aprender los patrones generales. Como resultado, falla al recibir datos nuevos.
* **Soluciones:**
  1. Utilizar más datos en el entrenamiento (*training*).
  2. Reducir la complejidad del modelo (aplicar regularización).
  3. Utilizar **Cross Validation**.

### Underfitting (Subajuste)
* **Qué es:** Ocurre cuando el modelo es demasiado simple o se entrena con muy pocos datos, por lo que no logra identificar los patrones subyacentes y tampoco generaliza bien.

### Bias vs. Variance (Sesgo vs. Varianza)
* **Bias (Sesgo):** Capacidad del modelo para ajustarse a los datos de entrenamiento. Un *bias* alto provoca *Underfitting*.
* **Variance (Varianza):** Sensibilidad del modelo ante pequeñas variaciones en los datos. Si se ajusta en exceso al entrenamiento, tendrá una *variance* alta y provocará *Overfitting*.
* **Meta:** Encontrar el punto de equilibrio reduciendo ambos al máximo para captar el patrón real y generalizar bien a futuro.

---

## 5. Técnicas Avanzadas de Optimización y Validación

### A. Regularization (Regularización)
Consiste en simplificar el modelo (penalizando su complejidad) para mejorar su capacidad de generalización. Con esto se asume un ligero aumento del error en *train* a cambio de disminuir drásticamente el error en *test*, reduciendo la brecha entre ambos.

### B. Cross Validation (Validación Cruzada - K-Folds)
Técnica para evaluar el modelo de forma independiente a cómo se hayan partido los datos:
1. Se mezclan los datos y se dividen en `K` subconjuntos o *folds* (habitualmente **10**, o **5** si el dataset es muy grande).
2. Se entrena el modelo usando `K-1` grupos y se evalúa con el grupo restante.
3. Se repite el proceso rotando el grupo de evaluación hasta pasar por todos.
4. Se calcula la media de los errores obtenidos en todas las iteraciones para conocer el rendimiento real del modelo.

### C. Flujo Completo: Train – Validation – Test
Para evitar hacer trampas y sobreajustar al conjunto de *test* al elegir hiperparámetros, se sigue este flujo:
1. Separar el **Test (~20%)** y guardarlo bajo llave hasta el final.
2. Con el **Train (~80%)** restante, aplicar **Cross Validation** (dividiendo internamente en *Train* y *Validation*) para entrenar y comparar distintos modelos.
3. Elegir el modelo con menor error en validación.
4. Reentrenar ese modelo ganador usando todo el bloque inicial de **Train + Validation (80%)**.
5. Evaluar el modelo final una única vez contra el **Test (20%)** desconocido.

---

## 6. Regla de Oro: Evitar el Data Leakage (Contaminación de Datos)
El conjunto de *test* no debe influir jamás en las transformaciones aplicadas al conjunto de *train*. Todos los cálculos de limpieza y preparación se aprenden únicamente en *train* y luego se aplican sobre *test*:
* **Scalers:** Ajustar el escalador (ej. `StandardScaler`) solo con *train* y transformar *test* con ese mismo objeto.
* **Valores nulos (Missings):** Imputar en *test* usando la media, mediana o moda calculada exclusivamente en *train*.
* **Feature Reduction & Engineering:** Eliminar exactamente las mismas columnas y aplicar las mismas fórmulas creadas en *train*.

---

## 7. Herramientas en Python
No es necesario programar las fórmulas matemáticas desde cero; el ecosistema de Python ofrece librerías especializadas que abstraen esta complejidad:
* **Scikit-learn (`sklearn`):** Estándar para modelos clásicos de Machine Learning y preprocesamiento.
* **TensorFlow + Keras:** Librerías principales para desarrollar modelos de Deep Learning.




<hr>
