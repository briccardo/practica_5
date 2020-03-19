import pandas
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

datos = pandas.read_excel('data/PRÁCTICAS VALORACIÓN AGRARIA_SINTETICOS_REGRESIÓN_AULA.xlsx', sheet_name='EJERCICIO 2')
df = DataFrame(datos)

x = df[['Salinidad (s)', 'Acceso (a)']]  # independiente
y = df['Precio (€/ha) (V)']  # dependiente

salinidad = 4
acceso = 5

# regresion
poly = PolynomialFeatures(degree=6)
x_ = poly.fit_transform(x)

lineal = linear_model.LinearRegression()
lineal.fit(x_, y)
y_poly_pred = lineal.predict(x_)

predict_ = poly.fit_transform([[salinidad, acceso]])

print(f'Resultado regresión: {round(lineal.predict(predict_)[0], 2)}')