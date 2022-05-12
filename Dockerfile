FROM python:3.8.13-slim

# 👇 Installation de dépendances sur le système
RUN apt-get update \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 👇 Packaging des sources python
# COPY requirements_test.txt setup.py /app/
# COPY models/ /app/servier/
# COPY main.py /app/
# COPY app.py /app/
COPY . /app

# 👇 Packaging des données
# COPY data/processed/stations /app/data/processed/stations
# COPY data/processed/general_info_stations.parquet /app/data/processed/general_info_stations.parquet

# 👇 Génération et installation de dépendances Python via wheel
WORKDIR /app
RUN pip install --user -U pip && python setup.py bdist_wheel
RUN pip install dist/servier-0.2-py3-none-any.whl
RUN pip install gunicorn flask

# 👇 Exposition de l'application via le port 5002
EXPOSE 5002
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app"]
