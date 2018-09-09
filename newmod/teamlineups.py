import numpy as np
import pickle

# load game trajectories
gametraj = pickle.load(open('gametraj.pickle','rb'))

# team names
teamnames = gametraj.keys()

# form team lineups
teamlineups = {}  # for each team, this gives a list of lineups
                  # can also think of this as the reverse of teamforlookup
teamforlookup = {}   # for each team and each lineup, gives us an integer index
lineupgamesplayed = {}  # for each team & lineup, number of games played
lineupplayingtimes = {} # for each team & lineup, number of seconds played

for tn in teamnames:
    teamlineups[tn] = []
    lineupgamesplayed[tn] = {}
    for i in gametraj[tn]:
        thislist = gametraj[tn][i][:,1].tolist()
        for j in thislist:
            teamlineups[tn].append(j)

        uniquelist = list(set(thislist))
        for j in uniquelist:
            if j in lineupgamesplayed[tn]:
                lineupgamesplayed[tn][j] += 1
            else:
                lineupgamesplayed[tn][j] = 1


for tn in teamnames:
    teamlineups[tn] = list(set(teamlineups[tn]))

for tn in teamnames:
    teamforlookup[tn] = {}
    for i in teamlineups[tn]:
        teamforlookup[tn][i] = teamlineups[tn].index(i)

for tn in teamnames:
    lineupplayingtimes[tn] = {}
    for i in gametraj[tn]:
        thisgame = gametraj[tn][i]
        for j in range(1,thisgame.shape[0]):
            timeplayed = thisgame[j,0] - thisgame[j-1,0]
            if thisgame[j-1,1] in lineupplayingtimes[tn]:
                lineupplayingtimes[tn][thisgame[j-1,1]] += timeplayed
            else:
                lineupplayingtimes[tn][thisgame[j-1,1]] = timeplayed

pickle.dump(teamlineups, open('teamlineups.pickle','wb'))
pickle.dump(teamforlookup, open('teamforlookup.pickle','wb'))
pickle.dump(lineupgamesplayed, open('lineupgamesplayed.pickle','wb'))
pickle.dump(lineupplayingtimes, open('lineupplayingtimes.pickle','wb'))


