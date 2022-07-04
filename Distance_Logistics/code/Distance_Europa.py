from typing import List

import pandas as pd
import requests #to get the distances from the API
import json #to read the API response
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose #for travelling salesman problem
import datetime

# https://github.com/vaclavdekanovsky/data-analysis-in-examples/blob/master/Maps/Driving%20Distance/Driving%20Distance%20between%20two%20places.ipynb


# load the dataframe with capitals
df = pd.read_csv("C:\Ezequiel_Sanchez\Proyectos\S&T_Distance_Logistics\data\concap.csv")
pd.set_option('display.max_columns', None)
# print(df)
# rename so that the column names are shorter and comply with PEP-8
df.rename(columns={"CountryName": "Country", "CapitalName": "capital", "CapitalLatitude": "lat", "CapitalLongitude": "lon", "CountryCode": "code", "ContinentName": "continent"}, inplace=True)
# print(df)

# to start with let's filter only 2 capitals. Rome and Paris.
# ropa = df[df["capital"].isin(["Rome","Paris"])].reset_index()
# cities = ropa.copy()
# print(cities)

# filter only the capitals of the Central Europe
ce_countries = ["AT","CZ","DE","HU","LI","PL","SK","SI","CH"]
ce_cities = df[df["code"].isin(ce_countries)].reset_index(drop=True)
print(ce_countries)
print(ce_cities)


def get_distance(point1: dict, point2: dict) -> tuple:
    # Gets distance between two points en route using http://project-osrm.org/docs/v5.10.0/api/#nearest-service
    url = f"""http://router.project-osrm.org/route/v1/driving/{point1["lon"]},{point1["lat"]};{point2["lon"]},{point2["lat"]}?overview=false&alternatives=false"""
    r = requests.get(url)
    # get the distance from the returned values
    route = json.loads(r.content)["routes"][0]
    return (route["distance"], route["duration"])


# get the distances and durations
dist_array = []
for i , r in ce_cities.iterrows():
    point1 = {"lat": r["lat"], "lon": r["lon"]}
    for j, o in ce_cities[ce_cities.index != i].iterrows():
        point2 = {"lat": o["lat"], "lon": o["lon"]}
        dist, duration = get_distance(point1, point2)
        #dist = geodesic((i_lat, i_lon), (o["CapitalLatitude"], o["CapitalLongitude"])).km
        dist_array.append((i, j, duration, dist))


distances_df = pd.DataFrame(dist_array,columns=["origin","destination","duration(s)","distnace(m)"])
print("DISTANCIAS ENTRE CAPITALES")
print(distances_df)

# turn the first three columns of the dataframe into the list of tuples
dist_list = list(distances_df[["origin","destination","duration(s)"]].sort_values(by=["origin","destination"]).to_records(index=False))
dist_list[:5] + ["..."] + dist_list[-5:]
print("")
print(dist_list)


# we plan to use the list of distnaces (durations in our case), that's why we initialize with `distances = dist_list` param.
fitness_dists = mlrose.TravellingSales(distances = dist_array)

# we plan to visit 9 cities
length = ce_cities.shape[0]
problem_fit = mlrose.TSPOpt(length = length, fitness_fn = fitness_dists, maximize=False)

# non-ideal solution, without specifying mlrose optimization
mlrose.genetic_alg(problem_fit, random_state = 2)

# better but more resource intensive solutions
best_state, best_fitness = mlrose.genetic_alg(problem_fit, mutation_prob = 0.2,  max_attempts = 500, random_state = 2)
print("")
print(f"The best state found is: {best_state}, taking 43hs ({best_fitness} o {str(datetime.timedelta(seconds=best_fitness))})")


# turn the results to an ordered dict
orders = {city: order for order, city in enumerate(best_state)}
print("")
print(orders)

# apply this order to the dataframe with the cities
ce_cities["order"] = ce_cities.index.map(orders)
ce_cities = ce_cities.sort_values(by="order")
print("")
print("SHORTEST ROUTE")
print(ce_cities)




# GRAFICO
import plotly.graph_objects as go
path_df = ce_cities
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