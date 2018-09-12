import scrapenbastats as sns
import pandas as pd
import pickle

# games = pd.read_csv('games.csv')
# dates = [[]]*1230
games = pickle.load(open('games.pickle','rb'))

for i in range(452,1230):
    g = games['gameids'][i]
    gamedate = sns.getGameDate(str(g))
    print(i, g, gamedate)
    # dates[i] = gamedate
    games['dates'][i] = gamedate
    pickle.dump(games, open('games.pickle','wb'))


