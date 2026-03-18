# CDS_Group10

```
CDS_Group10
├─ README.md
├─ data
│  ├─ interim
│  │  └─ comcat
│  │     ├─ comcat_clean_events.parquet
│  │     └─ mainshock_target_dataset.parquet
│  ├─ processed
│  │  ├─ .DS_Store
│  │  ├─ comcat
│  │  │  ├─ baseline_mainshock_model_dataset.parquet
│  │  │  └─ mainshock_model_dataset.parquet
│  │  └─ integrated
│  └─ raw
│     └─ comcat
│        ├─ 2025
│        │  ├─ comcat_2025_01.geojson
│        │  ├─ comcat_2025_02.geojson
│        │  ├─ comcat_2025_03.geojson
│        │  ├─ comcat_2025_04.geojson
│        │  ├─ comcat_2025_05.geojson
│        │  ├─ comcat_2025_06.geojson
│        │  ├─ comcat_2025_07.geojson
│        │  ├─ comcat_2025_08.geojson
│        │  ├─ comcat_2025_09.geojson
│        │  ├─ comcat_2025_10.geojson
│        │  ├─ comcat_2025_11.geojson
│        │  └─ comcat_2025_12.geojson
│        └─ comcat_2015_2024.parquet
├─ notebook
│  └─ ComCat.ipynb
└─ src
   ├─ app
   │  └─ streamlit_app.py
   ├─ data
   │  ├─ make_features.py
   │  ├─ make_splits.py
   │  └─ validate_dataset.py
   ├─ features
   │  └─ preprocess.py
   ├─ models
   │  ├─ evaluate.py
   │  ├─ predict.py
   │  ├─ registry.py
   │  └─ train.py
   └─ utils
      ├─ io.py
      └─ paths.py

```