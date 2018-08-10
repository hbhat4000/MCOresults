import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = [[]]*4
data[0] = np.load('ctmcresultsPT1.npz')
data[1] = np.load('ctmcresultsPT2.npz')
data[2] = np.load('ctmcresultsPT3.npz')
data[3] = np.load('ctmcresultsPT4.npz')

combineddata = {}
datakeys = sorted(data[0].keys())
for k in datakeys:
    combineddata[k] = np.mean(np.vstack(list(map(lambda x : x[k], data))),axis=-1)

numstates = 2**np.arange(2,12)

regr = linear_model.LinearRegression()
regkeys = ['epsnorms','nnz','traintimeF','traintimeN']
for k in regkeys:
    regr.fit(np.log(numstates).reshape(-1, 1), np.log(combineddata[k]))
    print([k, regr.coef_])

otherkeys = ['LTerrN_test', 'LTerrN_train', 'STerrN', 'LTerrF_test', 'LTerrF_train', 'STerrF', 'constrviol']
for k in otherkeys:
    print([k, np.mean(combineddata[k])])

print(combineddata['LTerrF_test'])


