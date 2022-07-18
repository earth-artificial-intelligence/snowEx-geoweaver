# Write first python in Geoweaver
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

homedir = os.path.expanduser('~') 
datadir = f"{homedir}/snowx/"

polygon_filepath = str(f'{homedir}/snowx/nsidc-polygon.json') # Note: A shapefile or other vector-based spatial data format could be substituted here.

gdf = gpd.read_file(polygon_filepath) #Return a GeoDataFrame object

# Simplify polygon for complex shapes in order to pass a reasonable request length to CMR. The larger the tolerance value, the more simplified the polygon.
# Orient counter-clockwise: CMR polygon points need to be provided in counter-clockwise order. The last point should match the first point to close the polygon.
poly = orient(gdf.simplify(0.05, preserve_topology=False).loc[0],sign=1.0)

#Format dictionary to polygon coordinate pairs for CMR polygon filtering
polygon = ','.join([str(c) for xy in zip(*poly.exterior.coords.xy) for c in xy])
print('Polygon coordinates to be used in search:', polygon)

temporal = '2017-01-01T00:00:00Z,2017-12-31T23:59:59Z' # Set temporal range

param_dict = {
    'short_name': 'SNEX17_GPR',
    'version': '2',
    'polygon': polygon,
#     'bounding_box': bounding_box, #optional alternative to polygon search parameter; if using, remove or comment out polygon search parameter
    'temporal':temporal,
}

def search_granules(search_parameters, geojson=None, output_format="json"):
    """
    Performs a granule search with token authentication for restricted results
    
    :search_parameters: dictionary of CMR search parameters
    :token: CMR token needed for restricted search
    :geojson: filepath to GeoJSON file for spatial search
    :output_format: select format for results https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html#supported-result-formats
    
    :returns: if hits is greater than 0, search results are returned in chosen output_format, otherwise returns None.
    """
    search_url = "https://cmr.earthdata.nasa.gov/search/granules"

    
    if geojson:
        files = {"shapefile": (geojson, open(geojson, "r"), "application/geo+json")}
    else:
        files = None
    
    
    parameters = {
        "scroll": "true",
        "page_size": 100,
    }
    
    try:
        response = requests.post(f"{search_url}.{output_format}", params=parameters, data=search_parameters, files=files)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP Error: {http_error}")
    except Exception as err:
        print(f"Error: {err}")
    
    hits = int(response.headers['CMR-Hits'])
    if hits > 0:
        print(f"Found {hits} files")
        results = json.loads(response.content)
        granules = []
        granules.extend(results['feed']['entry'])
        granule_sizes = [float(granule['granule_size']) for granule in granules]
        print(f"The total size of all files is {sum(granule_sizes):.2f} MB")
        return response.json()
    else:
        print("Found no hits")
        return

print(search_granules(param_dict))

