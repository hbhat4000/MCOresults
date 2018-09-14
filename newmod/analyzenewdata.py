import pandas as pd
import pickle
import numpy as np

year = 2016
csvfname = 'events_'+str(year)+'-'+str(year+1)+'_pbp.csv'

# load regular season data
df = pd.read_csv(csvfname,
                 usecols=[4,7,8,34,35,36,37,38,
                          91,92,93,94,95,
                         96,97,98,99,100])


colnames = pd.read_csv('colnames.csv',header=None)

"""

# make a dictionary of all 5-person units
u2i = {}
i2u = {}
idctr = 0
for i in range(df.shape[0]):
    homekey = tuple(sorted(df.loc[i]['HOME_PLAYER_ID_1':'HOME_PLAYER_ID_5']))
    viskey = tuple(sorted(df.loc[i]['AWAY_PLAYER_ID_1':'AWAY_PLAYER_ID_5']))
    if homekey not in u2i:
        u2i[homekey] = idctr
        i2u[idctr] = homekey
        idctr += 1
    
    if viskey not in u2i:
        u2i[viskey] = idctr
        i2u[idctr] = viskey
        idctr += 1
    



pickle.dump(u2i, open('./'+str(year)+'/u2i.pickle', 'wb'))
pickle.dump(i2u, open('./'+str(year)+'/i2u.pickle', 'wb'))

"""

u2i = pickle.load(open('./'+str(year)+'/u2i.pickle','rb'))
i2u = pickle.load(open('./'+str(year)+'/i2u.pickle','rb'))

# u2i[tuple(sorted([203926, 101127, 201584, 203124, 204001]))]
# df.loc[480]


uniqteamnames = sorted(list(set(df['AWAY_TEAM'].append(df['HOME_TEAM']))))

# print(uniqteamnames)
# print(len(uniqteamnames))



def myprocess(mat):
    selection = np.ones(mat.shape[0],dtype='int32')
    for i in range(1,mat.shape[0]-1):
        if mat[i,0] == mat[i+1,0]:
            selection[i] = 0
    
    newmat = np.copy(mat)[np.where(selection)[0],:]
    return newmat


gametraj = {}
for tn in uniqteamnames:
    gametraj[tn] = {}


# need this because the event....csv files do not have correct home/away teams
games = pickle.load(open('./'+str(year)+'/games.pickle','rb'))


curgameid = -999999
for i in range(df.shape[0]):
    # check if current row is first row of new game
    if curgameid != df.loc[i]['GAME_ID']:
        curgameid = df.loc[i]['GAME_ID']
        tn = games[games['gameids']==curgameid]['home'].values[0]
        # tn = df.loc[i]['HOME_TEAM']
        times = []
        units = []
        teamscores = []
        opposcores = []
        print("Working on game ",curgameid)
        lastunit = -1
    
    # only append if there has been a substitution
    # or if it is the last row of this game
    homekey = tuple(sorted(df.loc[i]['HOME_PLAYER_ID_1':'HOME_PLAYER_ID_5']))
    curunit = u2i[homekey]
    curtime = df.loc[i]['TIME']
    lastrow = (i == (df.shape[0]-1)) or (curgameid != df.loc[i+1]['GAME_ID'])
    if (lastunit != curunit) or lastrow:
        units.append(curunit)
        times.append(curtime)
        teamscores.append(df.loc[i]['HOME_SCORE'])
        opposcores.append(df.loc[i]['AWAY_SCORE'])
        lastunit = curunit    
    
    # save trajectory matrix if current row is last row of this game
    if lastrow:
        # create matrix from game that was just finished
        trajmat = np.vstack([times, units, teamscores, opposcores]).T
        gametraj[tn][curgameid] = myprocess(trajmat)
        




pickle.dump(gametraj, open('./'+str(year)+'/gametrajhalf.pickle', 'wb'))




curgameid = -999999
for i in range(df.shape[0]):
    # check if current row is first row of new game
    if curgameid != df.loc[i]['GAME_ID']:
        curgameid = df.loc[i]['GAME_ID']
        tn = games[games['gameids']==curgameid]['away'].values[0]
        times = []
        units = []
        teamscores = []
        opposcores = []
        print("Working on game ",curgameid)
        lastunit = -1
    
    # only append if there has been a substitution
    # or if it is the last row of this game
    viskey = tuple(sorted(df.loc[i]['AWAY_PLAYER_ID_1':'AWAY_PLAYER_ID_5']))
    curunit = u2i[viskey]
    curtime = df.loc[i]['TIME']
    lastrow = (i == (df.shape[0]-1)) or (curgameid != df.loc[i+1]['GAME_ID'])
    if (lastunit != curunit) or lastrow:
        units.append(curunit)
        times.append(curtime)
        teamscores.append(df.loc[i]['AWAY_SCORE'])
        opposcores.append(df.loc[i]['HOME_SCORE'])
        lastunit = curunit
    
    # save trajectory matrix if current row is last row of this game
    if lastrow:
        # create matrix from game that was just finished
        trajmat = np.vstack([times, units, teamscores, opposcores]).T
        gametraj[tn][curgameid] = myprocess(trajmat)
        




pickle.dump(gametraj, open('./'+str(year)+'/gametraj.pickle', 'wb'))




# remaining steps
# map out states+score vs time for all games
# consider cross-product space
# compute MLEs
# compute accuracies etc
# regularize

