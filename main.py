import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Lectura de los datos desde el archivo CSV
data = pd.read_csv('altura_peso.csv')

# se separan columnas en variables independientes (x) y dependientes (y)
x = data['Altura'].values.reshape(-1, 1)  
y = data['Peso'].values  

# Se dividen los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal con Keras
model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(x_train, y_train, epochs=2000, verbose=1)

# Evaluar el modelo
loss = model.evaluate(x_test, y_test)
print(f'Pérdida en el conjunto de prueba: {loss}')

# Visualizar los resultados
y_pred = model.predict(x_test)
plt.scatter(x_test, y_test, color='blue', label='Datos Reales')
plt.plot(x_test, y_pred, color='red', label='Predicciones')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.legend()
plt.show()

# se realizan predicciones con nuevas alturas
new_heights = np.array([160, 170, 180]).reshape(-1, 1)
predicted_weights = model.predict(new_heights)
print(f'Alturas: {new_heights.flatten()}')
print(f'Predicciones de peso: {predicted_weights.flatten()}')
