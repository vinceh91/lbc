servier
==============================

# Web app

flask run 
or
python app.py
http://127.0.0.1:5000/predict?q=Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C

## Docker Container for the web app

curl --location --request POST 'http://localhost:5002/predict?q=Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C' 
```
