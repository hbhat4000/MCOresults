import pandas as pd
import pickle

names = pd.read_csv('teamnames.csv', header=None)
nickname = {}
for i in range(names.shape[0]):
    nickname[names.loc[i][2]] = names.loc[i][0]

pickle.dump(nickname, open('nickname.pickle','wb'))

