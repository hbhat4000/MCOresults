import numpy as np
import mcoLH as mco
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# specify the states
states = np.array([2**x for x in range(2,3)])
lstates = len(states)

# keep track of how many simulations 
numberofsims = np.zeros(lstates)

# training and test size, i.e., length of time series to request from sampler
L = 100

# number of series to generate
nslist = np.array(np.round(np.sqrt(states)*500/2),dtype='int32')

# number of simulations, i.e., how many times we repeat the whole process
N = 20

# initialize all the arrays used to store results
LTerrN_train = np.zeros((lstates, N))
LTerrN_test = np.zeros((lstates, N))
STerrN = np.zeros((lstates, N))
traintimeN = np.zeros((lstates, N))

LTerrF_train = np.zeros((lstates, N))
LTerrF_test = np.zeros((lstates, N))
STerrF = np.zeros((lstates, N))
traintimeF = np.zeros((lstates, N))
solFound = np.zeros((lstates, N))
constrviol = np.zeros((lstates, N))
nnz = np.zeros((lstates, N))
epsnorms = np.zeros((lstates, 3, N))

baseline_train = np.zeros((4, lstates, N))
baseline_test = np.zeros((4, lstates, N))
baseline_ST = np.zeros((4, lstates, N))
traintimeB = np.zeros((4, lstates, N))

for whichstate in range(lstates):
    for simctr in range(N):
        print((whichstate,states[whichstate],simctr))
        M, w, lastrow = mco.createrandomCTMC(states[whichstate])
        ns = nslist[whichstate]
        ts = [[]]*ns
        tseq = [[]]*ns
        ts_test = [[]]*ns
        tseq_test = [[]]*ns
        for nsnum in range(ns):
            ts[nsnum], tseq[nsnum] = mco.sampleCTMC(M,L,0)
            ts_test[nsnum], tseq_test[nsnum] = mco.sampleCTMC(M,L,0)
        
        # naive test block
        o1,o2,o3,o4 = mco.test_CTMC(ts,tseq,ts_test,tseq_test)
        LTerrN_train[whichstate, simctr] = o1
        LTerrN_test[whichstate, simctr] = o2
        STerrN[whichstate, simctr] = o3
        traintimeN[whichstate, simctr] = o4
        
        # fixed test block
        o1,o2,o3,o4,o5 = mco.test_CTMC(ts,tseq,ts_test,tseq_test,md='fixed')
        LTerrF_train[whichstate, simctr] = o1
        LTerrF_test[whichstate, simctr] = o2
        STerrF[whichstate, simctr] = o3
        traintimeF[whichstate, simctr] = o4
        solFound[whichstate, simctr] = o5[0]
        constrviol[whichstate, simctr] = o5[1]
        nnz[whichstate, simctr] = o5[2]
        epsnorms[whichstate, :, simctr] = np.array(o5[3])

        # baseline test block
        for whichbase in range(4):
            mymode = 'baseline' + str(whichbase+1)
            o1,o2,o3,o4 = mco.test_CTMC(ts,tseq,ts_test,tseq_test,md=mymode)
            baseline_train[whichbase, whichstate, simctr] = o1
            baseline_test[whichbase, whichstate, simctr] = o2
            baseline_ST[whichbase, whichstate, simctr] = o3
            traintimeB[whichbase, whichstate, simctr] = o4

    outname = "ctmcresults"+str(states[whichstate])+".npz"
    np.savez(outname, LTerrN_train = LTerrN_train,
                       LTerrN_test = LTerrN_test,
                            STerrN = STerrN,
                        traintimeN = traintimeN,
                      LTerrF_train = LTerrF_train,
                       LTerrF_test = LTerrF_test,
                            STerrF = STerrF,
                        traintimeF = traintimeF,
                          solFound = solFound,
                        constrviol = constrviol,
                               nnz = nnz,
                          epsnorms = epsnorms,
                    baseline_train = baseline_train,
                     baseline_test = baseline_test,
                       baseline_ST = baseline_ST,
                        traintimeB = traintimeB)

