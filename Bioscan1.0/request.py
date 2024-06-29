import requests

url = 'http://127.0.0.1:5000/predict'
file_path = 'C:\\Big_Projects\\test6.png'

with open(file_path, 'rb') as img:
    files = {'file': img}
    response = requests.post(url, files=files)
    print(response.json())
