priceless
==============================

MS Dynamic Pricing

# Web app

## Docker Container for the web app

```bash
docker build -t pricelessapp:v1 -f Dockerfile .
```

```bash
docker run -p 8501:8501 pricelessapp:v1
```

## Push docker to Azure VM

```bash
docker save pricelessapp:v1 | bzip2 | pv | ssh -i priceless-vm_key.pem kaboudan@20.86.120.64 sudo docker load
```

## Streamlit Web App with Azure App service

Create docker registry

```bash
az acr create --name PricelessAppRegistry --resource-group azrgd-dtst-01 --sku basic --admin-enabled true
```

Build the docker image and save to Azure Container Registry

```bash
az acr build --registry PricelessAppRegistry --resource-group azrgd-dtst-01 --image priceless-app .
```

Deploy a Web App from a Container Image

NO RIGHT !!!

```bash
az appservice plan create -g PricelessApp -n PricelessAppServicePlan -l westeurope --is-linux --sku B1
```

B1 — the first tier above free

```bash
az webapp create -g PricelessApp -p PricelessAppServicePlan -n priceless-web-app -i pricelessappregistry.azurecr.io/priceless-app:latest
```
