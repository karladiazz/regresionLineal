import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Regresión Lineal Interactiva", layout="wide")

st.title("📊 Regresión Lineal Simple Interactiva")
st.write("""
Esta aplicación permite entrenar un modelo de regresión lineal simple y hacer predicciones.
Puedes subir un archivo CSV o ingresar valores manualmente para ver los resultados en tiempo real.
""")

# Subir CSV
uploaded_file = st.file_uploader("📁 Sube tu archivo CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔍 Vista previa de los datos")
    st.dataframe(data.head())
    st.write("### 📊 Estadísticas del dataset")
    st.dataframe(data.describe())

    # Selección de variables
    x_column = st.selectbox("Variable independiente (X)", data.columns)
    y_column = st.selectbox("Variable dependiente (Y)", data.columns)

    X = data[[x_column]].values
    y = data[y_column].values

    # Entrenar modelo
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Resultados
    st.subheader("📈 Resultados del modelo")
    st.write(f"**R² (bondad de ajuste):** {r2_score(y, y_pred):.4f}")
    st.write(f"**Coeficiente (pendiente):** {model.coef_[0]:.4f}")
    st.write(f"**Intercepto:** {model.intercept_:.4f}")

    # Gráfica interactiva
    st.subheader("📉 Visualización del modelo")
    color_datos = st.color_picker("Color de los puntos de datos", "#1f77b4")
    color_linea = st.color_picker("Color de la línea de regresión", "#ff0000")

    fig, ax = plt.subplots()
    ax.scatter(X, y, color=color_datos, label="Datos reales")
    ax.plot(X, y_pred, color=color_linea, linewidth=2, label="Línea de regresión")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.legend()
    st.pyplot(fig)

    # Predicciones manuales múltiples
    st.subheader("🖊 Predicciones manuales")
    valores_x = st.text_area(
        f"Ingrese valores de {x_column} separados por comas",
        "10, 20, 30"
    )
    if st.button("Calcular predicciones"):
        try:
            valores_x = [float(x.strip()) for x in valores_x.split(",")]
            predicciones = model.predict(np.array(valores_x).reshape(-1, 1))
            resultados = pd.DataFrame({
                x_column: valores_x,
                f"Predicción de {y_column}": predicciones
            })
            st.dataframe(resultados)
        except Exception as e:
            st.error(f"Error: {e}")

# Predicción rápida sin CSV
st.subheader("🔮 Predicción rápida")
x_input = st.number_input("Ingrese un valor de X para predecir Y", value=0.0)
if st.button("Predecir rápido"):
    try:
        prediction = model.predict([[x_input]])
        st.success(f"Predicción: Y = {prediction[0]:.4f}")
    except NameError:
        st.warning("Primero debes subir un CSV y entrenar el modelo.")
