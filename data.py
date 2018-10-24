import requests, json, urllib, urllib.request, numpy as np, pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


def obtain_json():
    parameters={"limit":1000,"days":5000}
    response=requests.get('https://eonet.sci.gsfc.nasa.gov/api/v2/events?status=closed')
    data = response.json()

    output_file = 'json_response.json'

    with open(output_file, 'w') as f:
        json.dump(data, f)


def obtain_categories(fpath=None):
    # with open(fpath, 'r') as f:
    #     data = json.load(f)
    #     data = data['events']
    #
    # categories = set()
    # for temp in data:
    #     for t in temp['categories']:
    #         categories.add(t['title'])
    # categories = sorted(categories)
    categories = ('Drought', 'Dust and Haze', 'Earthquakes', 'Floods', 'Landslides', 'Manmade', 'Sea and Lake Ice', 'Severe Storms', 'Snow', 'Temperature Extremes', 'Volcanoes', 'Water Color', 'Wildfires')
    return categories


def make_soup(url):
    thepage = urllib.request.urlopen(url)
    soupdata = BeautifulSoup(thepage, "html.parser")
    return soupdata


def obtain_image_url(url, id):
    if id not in ['NASA_DISP', 'EO']:
        return []

    try:
        soup = make_soup(url)
    except:
        return []

    if soup is None:
        return []

    image_urls = []
    for img in soup.find_all('img'):
        temp = img.get('src')
        if temp[:1] == '/':
            image = None
            # image = "https://disasters.nasa.gov" + temp
        else:
            image = temp

        if id == 'NASA_DISP' and image:
            if image.split('/')[7] != 'sidebar_thumb':
                image_urls.append(image)

        elif id == 'EO' and image:
            image_urls.append(image)

    return image_urls


def obtain_url_csv():
    json_file_path = 'json_response.json'
    csv_path = 'eo_nasa_urls.csv'

    with open(json_file_path, 'r') as f:
        data = json.load(f)
        data = data['events']

    all_cats = obtain_categories()
    n = len(all_cats)
    cat_dict = {k: i for i, k in enumerate(all_cats)}

    csv_data = []

    for temp in tqdm(data):
        y = np.zeros((n,), dtype='int')
        img_urls = []

        for c in temp['categories']:
            y[cat_dict[c['title']]] = 1

        for s in temp['sources']:
            img_urls.extend(obtain_image_url(s['url'], s['id']))

        if img_urls:
            csv_data.extend([[i] + list(y) for i in img_urls])

    df = pd.DataFrame(csv_data, columns=['url'] + list(all_cats))
    df.to_csv(csv_path)


if __name__ == '__main__':
    obtain_url_csv()

