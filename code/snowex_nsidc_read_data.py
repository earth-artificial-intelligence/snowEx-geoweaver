import os
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from shapely.geometry.polygon import orient
import pandas as pd 
import requests
import json
import pprint
import getpass
import netrc
from platform import system
from getpass import getpass
from urllib import request
from http.cookiejar import CookieJar
from os.path import join, expanduser
import requests
from xml.etree import ElementTree as ET
import time
import zipfile
import io
import shutil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import requests
import os

homedir = os.path.expanduser('~')
datadir = f"{homedir}/snowx/"

snowex_path = f'{datadir}/SnowEx17_GPR_Version2_Week1.csv' # Define local filepath
df = pd.read_csv(snowex_path, sep='\t') 
print(df.head())

# extract date columns

df['date'] = df.collection.str.rsplit('_').str[-1].astype(str)
df.date = pd.to_datetime(df.date, format="%m%d%y")
df = df.sort_values(['date'])
print(df.head())

# Convert to Geopandas dataframe to provide point geometry

gdf_utm= gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs='EPSG:32612')
print(gdf_utm.head())

homedir = os.path.expanduser('~')
datadir = f"{homedir}/snowx/"

exampledataurl = "https://raw.githubusercontent.com/snowex-hackweek/website/main/book/tutorials/machine-learning/data/snow_depth_data.csv"

r = requests.get(exampledataurl, allow_redirects=True)

with open(f"{datadir}/snow_depth_data.csv", 'wb') as datafile:
  datafile.write(r.content)

