import numpy as np
import pandas as pd
import pickle
import mcoLH as mco

# load games and game trajectories
games = pickle.load(open('games.pickle','rb'))
gametraj = pickle.load(open('gametraj.pickle','rb'))

# team names
teamnames = list(gametraj.keys())

# pick a team, just for now
curteam = teamnames[14]

# set number of games to train on
traingames = 40

# figure out which gameids are in the training and test sets
hg = games['home'] == curteam
ag = games['away'] == curteam
curteamgames = games[hg | ag].sort_values(by=['dates'])
traingids = curteamgames.iloc[:traingames]['gameids'].tolist()
testgids = curteamgames.iloc[traingames:traingames+1]['gameids'].tolist()

# create list of all unique lineups encountered in training set
trainlineups = []
for i in traingids:
    thislist = gametraj[curteam][i][:,1].tolist()
    for j in thislist:
        trainlineups.append(j)

# need this to standardize the state space
trainlineups = list(set(trainlineups))
trainlookup = {}
for i in trainlineups:
    trainlookup[i] = trainlineups.index(i)

# scan through the gametraj matrices and produce two ingredients
# 1) a list of time sequences
# 2) a list of discrete-state time series on a standardized state space
alltseq = [[]]*traingames
alldsts = [[]]*traingames
for i in range(traingames):
    ts = gametraj[curteam][traingids[i]][:,0].tolist()
    ds = list(map(lambda x: trainlookup[x], gametraj[curteam][traingids[i]][:,1]))
    ctr = 0
    while ds[len(ds)-ctr-2] == ds[len(ds)-1]:
        ctr += 1

    alltseq[i] = ts[0:len(ds)-ctr]
    alldsts[i] = ds[0:len(ds)-ctr]

# train naive CTMC
phat, absstates, stt = mco.trainCTMCm(alldsts, alltseq, len(trainlineups))

# raw equilibrium
raweq = mco.equilib(phat, 'CTMC')

# compute empirical fraction of time in each state
wvec = stt/np.sum(stt)

# apply MCO to retrain CTMC
eps, constrviol = mco.fixCTMC(phat, wvec, forcePos = True)

# form new transition rate matrix
phatfix = mco.addPert(phat, eps, len(trainlineups), 'CTMC')

# new equilibrium
fixeq = mco.equilib(phatfix, 'CTMC')

# evaluate model on test set
# for each game, figure out how long it is
# extrapolate our raweq and fixeq to the game length to get
# prediction of how much each lineup plays
# (remember to convert back to original lineup number)
# then compare with reality
testgames = len(testgids)
rawtesterrors = np.zeros(testgames)
fixtesterrors = np.zeros(testgames)

for i in range(testgames):
    thisgame = gametraj[curteam][testgids[i]]
    gamelen = thisgame[thisgame.shape[0]-1,0]
    rawpred = raweq*gamelen
    fixpred = fixeq*gamelen
    testresults = {}
    for j in range(thisgame.shape[0]-1):
        testlineup = thisgame[j,1]
        secondsplayed = thisgame[j+1,0] - thisgame[j,0]
        if testlineup not in testresults:
            testresults[testlineup] = secondsplayed
        else:
            testresults[testlineup] += secondsplayed

    for testlineup in testresults:
        if testlineup not in trainlineups:
            rawtesterrors[i] += testresults[testlineup]
            fixtesterrors[i] += testresults[testlineup]
        else:
            rawtesterrors[i] += np.abs(testresults[testlineup] - rawpred[trainlookup[testlineup]])
            fixtesterrors[i] += np.abs(testresults[testlineup] - fixpred[trainlookup[testlineup]])

# to do:
# -- compare lineup distribution across test set
# -- convert to player times and compare!



