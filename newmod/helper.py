import numpy as np
import pandas as pd
import pickle

# helper functions

# load player list
pl = pd.read_csv('playerlist.csv')
playerdict = {}
for i in range(pl.shape[0]):
    playerdict[pl.loc[i]['PERSON_ID']] = pl.loc[i]['DISPLAY_FIRST_LAST']

# load important dictionaries
u2i = pickle.load(open('u2i.pickle','rb'))
i2u = pickle.load(open('i2u.pickle','rb'))

def humanlineup(lineup):
    lineuptuple = i2u[lineup]
    humanlineup = []
    for j in lineuptuple:
        humanlineup.append(playerdict[j])

    return humanlineup


