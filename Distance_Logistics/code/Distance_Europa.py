# IMPORTAR LIBRERIAS
import pandas as pd
import requests #to get the distances from the API
import json #to read the API response
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose #for travelling salesman problem
import datetime

# CARGAR LOS DATOS DE LAS CIUDADES Y SUS COORDENADAS
df = pd.read_csv("C:\Ezequiel_Sanchez\Proyectos\Distance_Logistics\data\concap.csv", sep=',')
pd.set_option('display.max_columns', None)
#print(df)

# RENOMBRAR COLUMNAS EN FORMATO PEP-8
df.rename(columns={"CountryName": "Country", "CapitalName": "capital", "CapitalLatitude": "lat", "CapitalLongitude": "lon", "CountryCode": "code", "ContinentName": "continent"}, inplace=True)
#print(df)

# FILTRAR 9 CIUDADES DE LA LISTA
ce_countries = ["AT","CZ","DE","HU","LI","PL","SK","SI","CH"]
#print(ce_countries)

# CREAR UN NUEVO DF CON LAS CIUDADES SELECCIONADAS
ce_cities = df[df["code"].isin(ce_countries)].reset_index(drop=True)
print("\nCIUDADES SELECCIONADAS")
print(ce_cities)

# OBTENER DISTANCIAS ENTRE LAS CIUDADES SELECCIONADAS CON OpenStreetMap (mapa de uso libre) & API de OSRM
# http://project-osrm.org/docs/v5.10.0/api/#nearest-service
def get_distance(point1: dict, point2: dict) -> tuple:
    url = f"""http://router.project-osrm.org/route/v1/driving/{point1["lon"]},{point1["lat"]};{point2["lon"]},{point2["lat"]}?overview=false&alternatives=false"""
    r = requests.get(url)
    # OBTENER LAS DISTANCIAS EN FORMATO JSON
    route = json.loads(r.content)["routes"][0]
    return (route["distance"], route["duration"])

# OBTENER LAS DISTANCIAS Y LAS DURACIONES DE LAS COMBINACIONES SELECCIONADAS
dist_array = []
for i , r in ce_cities.iterrows():
    point1 = {"lat": r["lat"], "lon": r["lon"]}
    for j, o in ce_cities[ce_cities.index != i].iterrows():
        point2 = {"lat": o["lat"], "lon": o["lon"]}
        dist, duration = get_distance(point1, point2)
        #dist = geodesic((i_lat, i_lon), (o["CapitalLatitude"], o["CapitalLongitude"])).km
        dist_array.append((i, j, duration, dist))

# CREAR DF CON DISTANCIAS Y DURACIONES
distances_df = pd.DataFrame(dist_array,columns=["origin","destination","duration(s)","distnace(m)"])
print("DISTANCIAS ENTRE CAPITALES")
print(distances_df)

# TRANSFORMAR LAS TRES PRIMERAS COLUMNAS DEL DF EN UNA LISTA DE TUPLAS
# Las tuplas se utilizan para almacenar varios elementos en una sola variable
# IR DEL PUNTO A AL B Y SU DURACION
dist_list = list(distances_df[["origin","destination","duration(s)"]].sort_values(by=["origin","destination"]).to_records(index=False))
dist_list[:5] + ["..."] + dist_list[-5:]
#print("\nTUPLAS")
#print(dist_list)
#print(dist_array)

# ML CON MLROSE
# MLROSE es un paquete de Python para aplicar algunos de los algoritmos de búsqueda y optimización aleatoria más comunes
# a una variedad de problemas de optimización diferentes, tanto en espacios de parámetros de valores discretos como continuos
# https://mlrose.readthedocs.io/en/stable/

# OBJETIVO: ENCONTRAR EL RECORRIDO MAS CORTO
# SELECCIONAR UN ALGORITMO DE OPTIMIZACION ALEATORIA PARA RESOLVER EL PROBLEMA
# ALGORITMOS GENETICOS
# Se adaptan para resolver problemas de maximizacion o minimizacion

# Fitness -> evolucion (https://www.youtube.com/watch?v=K88hTnzo-tI)
# TravellingSales() calcula la duracion total de un recorrido
fitness_dists = mlrose.TravellingSales(distances=dist_array)

# TSPOpt() FUNCION DE OPTIMIZACION PARA LAS CIUDADES SELECCIONADAS
length = ce_cities.shape[0]
problem_fit = mlrose.TSPOpt(length=length, fitness_fn=fitness_dists, maximize=False)

# OPCION 1: SIN MODIFICAR LOS HIPERPARAMETROS DE MLROSE
#best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 2)
#print("\nSOLUCION")
#print(best_state)
#print(best_fitness)

# OPCION 2: MODIFICANDO HIPERPARAMENTROS DE MLROSE
# Se obtienen mejores resultados, pero se utilizan mas recursos
best_state, best_fitness = mlrose.genetic_alg(problem_fit, mutation_prob=0.2,  max_attempts=500, random_state=2)
print("\nSOLUCION")
print(f"\nMEJOR RECORRIDO ENCONTRADO: {best_state} \nDURACION: 198,7 hs - {best_fitness} segundos \nDIAS: {str(datetime.timedelta(seconds=best_fitness))} horas")
print("\nSOLUCION GOOGLE MAPS")
print("DURACION: 190 hs")

# ORDENAR LOS RESULTADOS
orders = {city: order for order, city in enumerate(best_state)}
#print("")
#print(orders)

# CREAR DF CON LA MEJOR RUTA
ce_cities["order"] = ce_cities.index.map(orders)
ce_cities1 = ce_cities.sort_values(by="order")
print("")
print("MEJOR RUTA")
print(ce_cities1)


# GRAFICO
import plotly.graph_objects as go
path_df = ce_cities1.reset_index()
# draw the capitals
fig = go.Figure(data=go.Scattergeo(
    locationmode='USA-states',
    lon=path_df['lon'],
    lat=path_df['lat'],
    text=df['capital'],
    mode='markers',
    name="capitals"))

# draw the paths between the capitals
for i in range(len(path_df) - 1):
    fig.add_trace(go.Scattergeo(
        locationmode='USA-states',
        lon=[path_df.loc[i, "lon"], path_df.loc[i + 1, "lon"]],
        lat=[path_df.loc[i, "lat"], path_df.loc[i + 1, "lat"]],
        name="-".join([path_df.loc[i, "capital"], path_df.loc[i + 1, "capital"]]),
        mode="lines"))

# the last path
fig.add_trace(go.Scattergeo(
    locationmode='USA-states',
    lon=[path_df.loc[8, "lon"], path_df.loc[0, "lon"]],
    lat=[path_df.loc[8, "lat"], path_df.loc[0, "lat"]],
    name="-".join([path_df.loc[8, "capital"], path_df.loc[0, "capital"]]),
    mode="lines"))

fig.update_layout(
    title='Shortest Route Between Central European Cities',
    geo_scope='europe',)

print(fig.show())


print("")
print("...:::| F I N I S H E D |:::...")