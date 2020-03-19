import pandas
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

datos = pandas.read_excel('data/PRÁCTICAS VALORACIÓN AGRARIA_SINTETICOS_REGRESIÓN_AULA.xlsx', sheet_name='EJERCICIO 3')
df = DataFrame(datos)

x = df[['PRODUCCIÓN BRUTA (€/kg)', 'EDAD PLANTACIÓN (años)', 'DISTANCIA AL NÚCLEO URBANO (km)',
        'RIESGO DE HELADA (%)']]  # independiente
y = df['PRECIO COMPRA-VENTA (€/ha)']  # dependiente

prod_bruta = 30
edad_plantacion = 15
distancia_no = 3
riesgo_helada = 11

# regresion
poly = PolynomialFeatures(degree=6)
x_ = poly.fit_transform(x)

lineal = LinearRegression()
lineal.fit(x_, y)
y_poly_pred = lineal.predict(x_)
predict_ = poly.fit_transform([[prod_bruta, edad_plantacion, distancia_no, riesgo_helada]])

print(f'Resultado regresión: {round(lineal.predict(predict_)[0], 2)}')
