# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:47:16 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)



# Título y descripción
st.title("Predicción de Potencial Comercial de Yacimientos Petroleros")
st.write("""
Esta aplicación utiliza Machine Learning para predecir si un yacimiento petrolero es **comercial** o no. 
Ajusta las propiedades del modelo, entrena y evalúa el rendimiento. 🛢️
""")

# Barra lateral con configuración y ayuda
st.sidebar.header("Configuración del modelo")
st.sidebar.write("""
### Ayuda
Esta aplicación permite predecir el **potencial comercial** de yacimientos petroleros utilizando modelos de Machine Learning.

#### Modelos utilizados:
- **Regresión Logística**: Un modelo lineal que estima la probabilidad de que un yacimiento sea comercial o no.
- **Árbol de Decisión**: Un modelo basado en reglas de decisión que divide los datos en función de características.
- **Bosques Aleatorios**: Un conjunto de árboles de decisión que mejora la precisión mediante el voto entre varios árboles.

#### ¿Cómo funciona la aplicación?
1. El usuario puede ajustar la configuración del modelo en la barra lateral.
2. Se entrena el modelo seleccionado con los datos geofísicos simulados.
3. Se presenta la **precisión** del modelo, junto con métricas como la **matriz de confusión** y la **curva ROC**.
4. También puedes explorar gráficos interactivos de las características y la evolución de la precisión del modelo a lo largo del tiempo.

Desarrollado por:  **Javier Horacio Pérez Ricárdez**.
""")

# Simulación de datos geofísicos
np.random.seed(42)
n_samples = 500
data = {
    "Porosidad (%)": np.random.uniform(5, 30, n_samples),
    "Permeabilidad (mD)": np.random.uniform(10, 500, n_samples),
    "Saturación de Agua (%)": np.random.uniform(10, 80, n_samples),
    "Profundidad (m)": np.random.uniform(1000, 4000, n_samples),
    "Potencial Comercial": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
}
df = pd.DataFrame(data)

# Visualización inicial del dataset
if st.checkbox("Mostrar dataset simulado"):
    st.write(df)

# División de datos
X = df[["Porosidad (%)", "Permeabilidad (mD)", "Saturación de Agua (%)", "Profundidad (m)"]]
y = df["Potencial Comercial"]
test_size = st.sidebar.slider("Tamaño del conjunto de prueba (%)", 10, 50, 30, step=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# Selección de modelo
model_option = st.sidebar.selectbox(
    "Selecciona un modelo:",
    ("Regresión Logística", "Árbol de Decisión", "Bosques Aleatorios")
)

# Configuración de hiperparámetros
if model_option == "Regresión Logística":
    c_value = st.sidebar.slider("C (Regularización)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=c_value, max_iter=200)
elif model_option == "Árbol de Decisión":
    max_depth = st.sidebar.slider("Máxima profundidad", 1, 20, 5)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
else:  # Bosques Aleatorios
    n_estimators = st.sidebar.slider("Número de árboles", 10, 200, 100, step=10)
    max_depth = st.sidebar.slider("Máxima profundidad", 1, 20, 10)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

# Entrenamiento
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Rendimiento del Modelo")
st.write(f"Precisión del modelo: **{accuracy:.2f}**")

# Matriz de confusión con etiquetas descriptivas
conf_matrix = confusion_matrix(y_test, y_pred)

# Crear un DataFrame con etiquetas y valores
conf_matrix_with_labels = pd.DataFrame(
    [[f"TN: {conf_matrix[0, 0]}", f"FP: {conf_matrix[0, 1]}"],
     [f"FN: {conf_matrix[1, 0]}", f"TP: {conf_matrix[1, 1]}"]], 
    index=["Real: No Comercial", "Real: Comercial"],
    columns=["Predicción: No Comercial", "Predicción: Comercial"]
)

# Crear el heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=conf_matrix_with_labels.values, fmt="", cmap="Blues", cbar=False, annot_kws={"size": 12})
ax.set_xlabel("Predicción", fontsize=14)
ax.set_ylabel("Real", fontsize=14)
ax.set_title("Matriz de Confusión con Etiquetas", fontsize=16)
st.pyplot(fig)

# Añadir explicación en texto
st.write("""
### Desglose de la Matriz de Confusión:
- **TN (Verdaderos Negativos):** Yacimientos correctamente clasificados como **No Comercial**.
- **FP (Falsos Positivos):** Yacimientos clasificados como **Comercial**, pero no lo son.
- **FN (Falsos Negativos):** Yacimientos clasificados como **No Comercial**, pero son comerciales.
- **TP (Verdaderos Positivos):** Yacimientos correctamente clasificados como **Comercial**.
""")

# Gráfico de características
if st.checkbox("Mostrar análisis exploratorio"):
    feature = st.selectbox("Selecciona una propiedad para analizar:", X.columns)
    chart_type = st.radio("Selecciona el tipo de gráfico:", ["Histograma", "Dispersión (Scatter)"])

    if chart_type == "Histograma":
        fig = px.histogram(
            df,
            x=feature,
            color="Potencial Comercial",
            nbins=30,
            title=f"Distribución de {feature} por Potencial Comercial",
            color_discrete_sequence=px.colors.qualitative.Set2,
            barmode="overlay",
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig)

    elif chart_type == "Dispersión (Scatter)":
        y_axis = st.selectbox("Selecciona una propiedad para el eje Y:", X.columns)
        fig = px.scatter(
            df,
            x=feature,
            y=y_axis,
            color="Potencial Comercial",
            title=f"Relación entre {feature} y {y_axis}",
            color_discrete_sequence=px.colors.qualitative.Set2,
            size="Porosidad (%)",  # Opcional para tamaño de punto.
            hover_data=["Porosidad (%)", "Permeabilidad (mD)"],
        )
        st.plotly_chart(fig)

# Gráfico de ROC interactivo
if hasattr(model, "predict_proba"):
    y_probs = model.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    st.subheader("Curva ROC - Interactiva")
    # Gráfico interactivo con Plotly
    roc_curve_fig = go.Figure()

    # Curva ROC
    roc_curve_fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines', name=f"AUC = {roc_auc:.2f}",
        line=dict(color="darkorange", width=2)
    ))

    # Línea diagonal de referencia con color más visible (azul claro)
    roc_curve_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name="Línea aleatoria",
        line=dict(color="blue", width=3, dash="dash")  # Color más visible
    ))

    # Configuración de diseño
    roc_curve_fig.update_layout(
        title="Curva ROC",
        xaxis_title="Tasa de Falsos Positivos (FPR)",
        yaxis_title="Tasa de Verdaderos Positivos (TPR)",
        showlegend=True,
        template="plotly_dark"
    )

    # Mostrar gráfico interactivo
    st.plotly_chart(roc_curve_fig)

# Simulación de evolución de la precisión en el tiempo (Gráfico animado)
# Crear un DataFrame con resultados a lo largo del tiempo
time_steps = 50  # Número de pasos en el tiempo
results_df = pd.DataFrame({
    'Time': np.linspace(0, 10, time_steps),
    'Accuracy': np.random.uniform(0.6, 1.0, time_steps),  # Simulamos una precisión
    'Model': np.random.choice(['Modelo 1', 'Modelo 2'], time_steps),  # Simulamos dos modelos
})

# Crear el gráfico animado
fig_anim = px.scatter(
    results_df.sort_values('Time'),
    x='Time', y='Accuracy', color='Model',
    animation_frame='Time',  # Animar con el tiempo
    title="Evolución de la precisión en función del tiempo",
    labels={'Time': 'Tiempo (s)', 'Accuracy': 'Precisión'}
)

# Ajustar la animación para que los puntos se acumulen y no desaparezcan
fig_anim.update_layout(
    updatemenus=[dict(
        type="buttons", showactive=False, x=0.1, xanchor="right",
        y=0, yanchor="bottom",
        buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')])]
    )]
)

fig_anim.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000  # Ajustar la duración del frame
fig_anim.update_traces(marker=dict(size=8, opacity=0.7))
st.plotly_chart(fig_anim)
