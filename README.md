# Wildfire Clustering in Europe
*Clusterización de incendios forestales en Europa*

---

## Description

An unsupervised machine learning model that classifies wildfires in Europe (2016–2024) based on location, seasonality, burned area, and vegetation type. Implemented in a reproducible pipeline using EFFIS data and HDBSCAN to identify wildfire patterns and typologies.

*Modelo de machine learning no supervisado que clasifica incendios forestales en Europa (2016–2024) según ubicación, estacionalidad, superficie quemada y tipo de vegetación. Implementado en un pipeline reproducible que usa datos de EFFIS y HDBSCAN para identificar patrones y tipologías de incendios.*

---

## Main Dependencies

- Python 3.10+
- pandas, geopandas
- hdbscan
- scikit-learn
- joblib

---

## Installation

1. Clone repository:

   ```bash
   git clone https://github.com/tu_usuario/ML_EFFIS.git
   cd ML_EFFIS

2. (Optional) Virtual environment and dependencies::
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt

## Quick Usage

A pretrained version of the model is available at:
   ```bash
   /src/models/EFFIS_SCAN.joblib
   ```
You can load it directly for testing without running the entire pipeline from scratch:
   ```python
   from joblib import load
   
   # Cargar el modelo preentrenado / Load pretrained model
   EFFIS_SCAN = load("src/models/EFFIS_SCAN.joblib")
   
   # Modificar parámetros clave del pipeline con kw_args: / Modify key pipeline parameters with kw_args:
   EFFIS_SCAN.set_params(Preprocess_shp__Filter_area__kw_args={'bbox': nuevo_bbox})                # Area of interest
   EFFIS_SCAN.set_params(Filter_transform__Feature_select__kw_args={'columns': cols_to_filter})    # Dataset column filter
   EFFIS_SCAN.set_params(Filter_transform__Area_filter__kw_args={'ha': new_area})                  # Minimum fire area to consider
   EFFIS_SCAN.set_params(Filter_transform__One-Hot__kw_args={'features': new_features_list})       # Vegetation type variables
   ```

This allows easy reuse of the pipeline for new geographic areas, custom filters, or analysis configurations.

👉 To run the full project workflow, follow the steps described in main.ipynb (in Spanish). 

## Credits
Developed by por Víctor Riera  
Source: European Forest Fire Information System (EFFIS)
