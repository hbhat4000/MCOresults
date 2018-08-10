import numpy as np
import scipy.linalg as sl
import mcoLH as mco

# number of states
N = 5

# number of series to generate
ns = 50000

M, w = mco.createrandomCTMC(N)

dsts = [[]]*ns
tseq = [[]]*ns
for i in range(ns):
    dsts[i], tseq[i] = mco.sampleCTMC(M, 50, 0)

X, w = mco.trainCTMCm(dsts, tseq, N)

print(sl.norm(X-M))

