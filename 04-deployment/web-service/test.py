import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://localhost:9696/predict' #specify endpoint predict or get 404
response = requests.post(url, json=ride)
print(response.json())
