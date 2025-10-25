import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.title("Regresión Lineal Simple")
st.write("""
Esta aplicación permite realizar regresión lineal simple entre dos variables cuantitativas.
Puedes subir un archivo CSV con tus datos y obtener predicciones y el valor de R².
""")

# Subir archivo
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.dataframe(data.head())

    # Seleccionar columnas
    col1, col2 = st.columns(2)
    with col1:
        x_column = st.selectbox("Selecciona la variable independiente (X)", data.columns)
    with col2:
        y_column = st.selectbox("Selecciona la variable dependiente (Y)", data.columns)

    X = data[[x_column]].values
    y = data[y_column].values

    # Crear modelo
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Mostrar resultados
    st.write(f"**R² (bondad de ajuste):** {r2_score(y, y_pred):.4f}")
    st.write(f"**Coeficiente (pendiente):** {model.coef_[0]:.4f}")
    st.write(f"**Intercepto:** {model.intercept_:.4f}")

    # Graficar
    fig, ax = plt.subplots()
    ax.scatter(X, y, color="blue", label="Datos reales")
    ax.plot(X, y_pred, color="red", label="Línea de regresión")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.legend()
    st.pyplot(fig)

    # Predicción de nuevos valores
    new_value = st.number_input(f"Ingrese un valor de {x_column} para predecir {y_column}")
    if st.button("Predecir"):
        prediction = model.predict([[new_value]])
        st.write(f"Predicción para {x_column} = {new_value}: {prediction[0]:.4f}")
