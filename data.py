import requests, json
parameters={"limit":1000,"days":5000}
response=requests.get('https://eonet.sci.gsfc.nasa.gov/api/v2/events?status=closed&source=EO')
data = response.json()

output_file = 'json_response.json'

with open(output_file, 'w') as f:
    json.dump(data, f)