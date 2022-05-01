FROM python:3.8.13-slim

# 👇 Installation de dépendances sur le système
RUN apt-get update \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 👇 Packaging des sources python
COPY requirements.txt setup.py /app/
COPY servier/ /app/servier/
# COPY config/ /app/config/

# 👇 Packaging pour la web app streamlit
# COPY webapp/ /app/webapp/
# COPY .streamlit/ /app/.streamlit/
# COPY .mapbox_token /app/

# 👇 Packaging des données
# COPY data/processed/stations /app/data/processed/stations
# COPY data/processed/general_info_stations.parquet /app/data/processed/general_info_stations.parquet

# 👇 Génération et installation de dépendances Python via wheel
WORKDIR /app
RUN pip install --user -U pip && python setup.py bdist_wheel
RUN pip install dist/servier-0.1-py3-none-any.whl

# 👇 Exposition de l'application avec Streamlit via le port 8501
# EXPOSE 8501
# CMD streamlit run webapp/dashboard.py
