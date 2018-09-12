import requests
from bs4 import BeautifulSoup

from slimit import ast
from slimit.parser import Parser
from slimit.visitors import nodevisitor
from datetime import datetime

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


