
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
import csv
from datetime import timedelta
# WEBSCRAPING
import bs4
from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
# HACER QUE PANDAS MUESTRE TODAS LAS COLUMNAS
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
# DESACTIVAR Warnings
pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# SALUDO
print("")
def print_hi(name):
    print(f'Hi, {name}. Lets go to calculate distances!')
if __name__ == '__main__':
    print_hi('S&T')
# Get the current working directory
#cwd = os.getcwd()
# Print the current working directory
#print("Current working directory: {0}".format(cwd))
print("")

# DISTANCE
import googlemaps
from datetime import datetime
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

df = pd.DataFrame({'name': ['paris', 'berlin', 'london']})
geolocator = Nominatim(user_agent="my-aplication")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

df['location'] = df['name'].apply(geocode)
df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)




#print(distance.distance(loc1,loc2).km, "kms")




print("")
print("...:::| F I N I S H E D |:::...")

