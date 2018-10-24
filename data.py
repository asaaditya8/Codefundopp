import requests, json, urllib, urllib.request
from bs4 import BeautifulSoup

def obtain_json():
    parameters={"limit":1000,"days":5000}
    response=requests.get('https://eonet.sci.gsfc.nasa.gov/api/v2/events?status=closed&source=EO')
    data = response.json()

    output_file = 'json_response.json'

    with open(output_file, 'w') as f:
        json.dump(data, f)

def make_soup(url):
    thepage = urllib.request.urlopen(url)
    soupdata = BeautifulSoup(thepage, "html.parser")
    return soupdata


if __name__ == '__main__':
    soup = make_soup("https://disasters.nasa.gov/hurricane-lane-2018")

    for img in soup.find_all('img'):
        temp = img.get('src')
        if temp[:1] == '/':
            image = "https://disasters.nasa.gov" + temp
        else:
            image = temp

        if image.split('/')[7] != 'sidebar_thumb':
            print(image)