import pandas as pd
import requests
from bs4 import BeautifulSoup

url = 'http://espn.go.com/nba/teams'
r = requests.get(url)

soup = BeautifulSoup(r.text)
tables = soup.find_all('ul', class_='Nav__Dropdown__Group__Section__List')

teams = []
prefix_1 = []
prefix_2 = []
teams_urls = []
for table in tables:
    lis = table.find_all('li')
    for li in lis:
        info = li.a
        url = info['href']
        if (url not in teams_urls) and ('nba' in url):
            teams.append(info.text)
            teams_urls.append(url)
            prefix_1.append(url.split('/')[-2])
            prefix_2.append(url.split('/')[-1])


dic = {'url': teams_urls, 'prefix_2': prefix_2, 'prefix_1': prefix_1}
teams = pd.DataFrame(dic, index=teams)
teams.index.name = 'team'

import pickle
pickle.dump(teams,open('teams.pickle','wb'))


