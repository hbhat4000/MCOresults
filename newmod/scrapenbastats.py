import requests
from bs4 import BeautifulSoup
import datetime
import json
import re
import pickle
import pandas as pd

"""
from slimit import ast
from slimit.parser import Parser
from slimit.visitors import nodevisitor

# this function assumes that gameid is passed in as a string
def getGameDate(gameid):
    # make sure gameid is zero-padded
    if len(gameid) == 8:
        gid = "00" + gameid
    else:
        gid = gameid

    baseurl = "https://stats.nba.com/game/"
    requrl = baseurl + gid + "/"
    tmp = requests.get(requrl)
    soup = BeautifulSoup(tmp.text, 'html.parser')
    x = soup.contents[164].contents[3]

    parser = Parser()
    tree = parser.parse(x.text)
    fields = {getattr(node.left, 'value', ''): getattr(node.right, 'value', '')
              for node in nodevisitor.visit(tree)
              if isinstance(node, ast.Assign)}

    y = fields['"GAME_DATE"']
    datetime_object = datetime.strptime(y[1:len(y)-1],'%A, %B %d, %Y')
    return datetime_object
"""

def getGameInfo(gameid):
    # make sure gameid is zero-padded
    if len(gameid) == 8:
        gid = "00" + gameid
    else:
        gid = gameid

    baseurl = "https://stats.nba.com/game/"
    requrl = baseurl + gid + "/"
    tmp = requests.get(requrl)
    soup = BeautifulSoup(tmp.text, 'html.parser')
    x = str(soup.contents[164].contents[3].text)
    leftbrak = [m.start() for m in re.finditer('\[',x)]
    rightbrak = [m.start() for m in re.finditer('\]',x)]
    twodicts = json.loads(x[leftbrak[1]:rightbrak[1]+1])

    x2 = soup.contents[164].contents[5]
    newdict = json.loads(str(x2.text))

    nickname = pickle.load(open('nickname.pickle','rb'))
    hometeam = nickname[newdict['homeTeam']]
    awayteam = nickname[newdict['awayTeam']]
    gamedate = datetime.datetime.strptime(newdict['startDate'],'%Y-%m-%d')

    team1 = twodicts[0]['TEAM_NICKNAME']
    if team1 == hometeam:
        homescore = twodicts[0]['PTS']
        awayscore = twodicts[1]['PTS']
    else:
        awayscore = twodicts[0]['PTS']
        homescore = twodicts[1]['PTS']
    
    return gameid, awayteam, hometeam, awayscore, homescore, gamedate



# this function expects year to be an integer
def getAllGames(year):
    gameids = ['002'+str(year)[2:]+'{:0>5}'.format(str(j)) for j in range(1,1231)]
    emptystrlist = ['' for i in range(1230)]
    emptyintlist = [0 for i in range(1230)]
    emptydatelist = [datetime.datetime(year=year,month=1,day=1) for i in range(1230)]
    
    games = pd.DataFrame({'gameids': gameids,
                          'away': emptystrlist,
                          'home': emptystrlist,
                          'awayscore': emptyintlist,
                          'homescore': emptyintlist,
                          'dates': emptydatelist})

    for i in range(1230):
        gid = gameids[i]
        info = getGameInfo(gid)
        print(i, gid, info[5])
        for j in range(1,6):
            games.loc[i,j] = info[j]


    pickle.dump(games, open('./'+str(year)+'/games.pickle','wb'))


