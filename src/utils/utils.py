import os
import numpy as np
import pandas as pd
import geopandas as gpd
import requests

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import sys
import zipfile
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError, HTTPError

### DESCARGA DE DATOS ###

def download_data(urls, data_directory):
    response = ''
    while response != 'S' and response != 'n':
        response = input('¿Estás seguro de que quieres descargar los datos? (S/n)')
    if response == 'S':
        for name, url in urls.items():
            print(f'Obteniendo {name}:')
            download(name, url, data_directory)
            print()

def download(name, url, directory):
    def show_progress(block_num, block_size, total_size):
        """
        Callback para urlretrieve que muestra el progreso de la descarga.
        """
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, int(downloaded * 100 / total_size))
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\rDescargando: {percent}% [{downloaded_mb:.2f}MB / {total_mb:.2f}MB]")
            sys.stdout.flush()
        else:
            downloaded_mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\rDescargando: {downloaded_mb:.2f}MB descargados (tamaño total desconocido)")
            sys.stdout.flush()

    data_zip = os.path.join(directory, f"{name}.zip")

    # Intentamos obtener el tamaño total del archivo antes de descargarlo
    try:
        with urlopen(url) as response:
            total_size = int(response.getheader('Content-Length'))
            total_mb = total_size / (1024 * 1024)
            print(f"Tamaño del archivo a descargar: {total_mb:.2f} MB")
    except (URLError, HTTPError, TypeError):
        print("No se pudo determinar el tamaño del archivo antes de la descarga.")

    # Descargamos el archivo
    urlretrieve(url, data_zip, reporthook=show_progress)

    # Mostramos el tamaño del archivo descargado
    if os.path.exists(data_zip):
        file_size = os.path.getsize(data_zip) / (1024 * 1024)
        print("\nDescarga completa ({file_size:.2f} MB)")

    # Descomprimimos
    with zipfile.ZipFile(data_zip, 'r') as zip:
        zip.extractall(directory)
        print(f"Archivo descomprimido en: {directory}")

    # Eliminamos el archivo zip
    os.remove(data_zip)
    print("Archivo ZIP eliminado.")

### FILTRO DE PAÍSES ###

def filtro_paises(df_paises:pd.DataFrame, paises:list):
    lista_codes = []
    for pais in paises:
        resultado = df_paises.loc[df_paises.NAME_ENGL == pais, 'CNTR_ID']
        code = resultado.values[0] if len(resultado) > 0 else print(f'No se encuentra {pais}')
        lista_codes.append(code)
    lista_codes.append('KS') # Código para Kosovo, podemos comprobarlo visualmente abriendo los shapes en un SIG

    return lista_codes

def nombre_pais(df_paises:pd.DataFrame, country_codes:list):
    lista_paises = []
    for code in country_codes:
        resultado = df_paises.loc[df_paises.CNTR_ID == code, 'NAME_ENGL']
        pais = resultado.values[0] if len(resultado) > 0 else 'Kosovo'
        lista_paises.append(pais)

    return lista_paises

eu_countries = [
    "Germany",
    "Austria",
    "Belgium",
    "Bulgaria",
    "Czechia",
    # "Cyprus",
    "Croatia",
    "Denmark",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Estonia",
    "Finland",
    "France",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Sweden",
    'Switzerland',
    'Norway','United Kingdom',
    'North Macedonia', 'Serbia', 'Albania', 'Bosnia and Herzegovina', 'Montenegro', 'Kosovo'
]

### FILTRO DE COORDENADAS ###

def nominatim(location:str) -> list:
    url = 'https://nominatim.openstreetmap.org/search?'
    headers = {"User-Agent": f'Spotweather_{location}'}
    params = {
        'format':'geocodejson',
        'namedetails':1,
        'addressdetails':1,
        'accept-language':'en-US,en;q=0.8,es-ES,es;q=0.9',
        'q':location
    }
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None

    results = response.json()['features']
    for index, result in enumerate(results):
        print(index, result['properties']['geocoding']['label'])
    return results

def info_ubicacion(location:dict) -> dict:
    properties = {}
    geocoding = location['properties']['geocoding']
    for feature in ['name', 'type', 'city','county','state', 'country','country_code']:
        properties[feature] = geocoding.get(feature, None) if feature != 'county' else geocoding['admin'].get('level6', None)
        print(f'{feature}: {properties[feature]}')

    name = properties['name']
    lon, lat = location['geometry']['coordinates']
    
    map = f'https://www.google.com/maps/@?api=1&map_action=map&basemap=satellite&center={lat}%2C{lon}'
    # URLs Google Maps  ->  https://developers.google.com/maps/documentation/urls/get-started?hl=es-419
    print(map)

    return {'name':name, 'lat':lat, 'lon':lon, 'properties':properties}

### EVALUACIÓN DE CLÚSTERES ###

def num_clusters(model, X):
    values, counts = np.unique(model.labels_, return_counts=True)
    num_clusters = max(values)+1
    perc_anomalias = round(counts[0]/len(X)*100, 2)

    print(f'Clusteres: {num_clusters}')
    print(f'Anomalías: {perc_anomalias} %\n')

def persistance(model):
    '''
    Crea un dataframe a partir de cluster_persistence_
    y retorna los clusteres con persistencia superior al 0.1
    '''

    df = pd.DataFrame(model.cluster_persistence_)
    df = df.reset_index().rename(columns={'index':'Clusters',0:'Persistance'})
    df =  df.sort_values('Persistance', ascending=False)
    return df.loc[df.Persistance > 0.1].reset_index().drop(columns='index')

def clusters_anomalias(datos, countries=None):
    fig, ax = plt.subplots(1,2, figsize=(15, 10))

    clusters = datos.loc[datos.CLUSTER != -1]
    anomalias= datos.loc[datos.CLUSTER == -1]

    sns.scatterplot(clusters, x='LON', y='LAT', hue='CLUSTER', edgecolor='none', palette='Spectral', ax=ax[0], legend=False)
    sns.scatterplot(anomalias, x='LON', y='LAT', hue='CLUSTER', edgecolor='none', palette='Spectral', ax=ax[1], legend=False)

    ax[0].set_title(f'Modelo')
    ax[1].set_title(f'Anomalías')

    for n in [0,1]:
        ax[n].set_ylim(34, 75)
        ax[n].set_xlim(-15, 35)
        ax[n].set_ylabel('')
        ax[n].set_xlabel('')

        # Agregamos el shapefile directamente desde geopandas
        if countries is not None:
            countries.plot(ax=ax[n], color='none', edgecolor='k', alpha=0.5)
        
    fig.tight_layout;

def detalle_cluster(model, datos, cluster, countries=None):
    datos = datos.loc[datos.CLUSTER == cluster]

    # Grid con 2 columnas: izquierda (mapa) y derecha (boxplots)
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)

    # Scatterplot en la mitad izquierda
    ax0 = fig.add_subplot(gs[0])
    sns.scatterplot(data=datos, x='LON', y='LAT', hue='CLUSTER', edgecolor='none',
                    palette='Spectral', ax=ax0, legend=False)
    ax0.set_ylim(34, 75)
    ax0.set_xlim(-15, 35)
    ax0.set_ylabel('')
    ax0.set_xlabel('')
    ax0.grid()
    for spine in ax0.spines.values():
        spine.set_visible(False)    
        if countries is not None:
            countries.plot(ax=ax0, color='none', edgecolor='k', alpha=0.5)

    # Subgrid con 3 filas para los boxplots horizontales
    gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1])

    features = ['AREA_HA', 'DAY_YEAR', 'YEAR']
    names = ['Area (ha)', 'Mes', 'Año']

    for i, (feature_name, name) in enumerate(zip(features, names)):
        ax_i = fig.add_subplot(gs_right[i])
        
        if feature_name == 'DAY_YEAR':
            feature = datos['DAY_YEAR'] / 30
        else:
            feature = datos[feature_name]

        sns.boxplot(data=datos, y='CLUSTER', x=feature, orient='h',
                    whis=0, flierprops={'marker': '|'}, ax=ax_i)

        # Etiquetas personalizadas para meses
        if feature_name == 'DAY_YEAR':
            ax_i.set_xticks(range(13))
            ax_i.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                                  'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic', ''])

        ax_i.set_xlabel(name)
        ax_i.set_ylabel('')
        ax_i.grid()
        for spine in ax_i.spines.values():
            spine.set_visible(False)
    
    # Título
    features_type = ['BROADLEA', 'CONIFER', 'SCLEROPH', 'AGRIAREAS']
    veg = [features_type[index] for index, x in enumerate(list(datos[features_type].describe().T['max'])) if x > 0]

    fig.suptitle(f'Cluster {cluster}  -  Persistencia: {model.cluster_persistence_[cluster]}\n{", ".join(veg)}',  fontsize=12, fontweight='bold')
    fig.tight_layout()