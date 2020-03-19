import pandas
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

datos = pandas.read_excel('data/PRÁCTICAS VALORACIÓN AGRARIA_SINTETICOS_REGRESIÓN_AULA.xlsx', sheet_name='EJERCICIO 1')
df = DataFrame(datos)

x = df[['Precio medio uva (€/kg)']]
y = df['VALOR MERCADO (€/ha)']
precio_uva = 0.63

poly = PolynomialFeatures(degree=6)
x_ = poly.fit_transform(x)

lineal = linear_model.LinearRegression()
lineal.fit(x_, y)
y_poly_pred = lineal.predict(x_)

predict_ = poly.fit_transform([[precio_uva]])

rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
r2 = r2_score(y, y_poly_pred)

print(f'Resultado regresión: {round(lineal.predict(predict_)[0], 2)}')
print(f'R²: {round(r2, 3)}')
print(f'Error Esperado: ±{round(rmse, 3)}')
