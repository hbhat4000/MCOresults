from mcoLH import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as io
# Specify the states
states = [2**x for x in range(2,9)]

# number of states
r = len(states)

# training size
L = 10000
# test size 
T = [500,10**3,10**4,10**5,10**6]
b = len(T)
# number of simulation
naive_avgerr = []
fixed_avgerr = []
constviol = []
sparsity_ratio = []
numsim = np.zeros(r)
# Record LT test errors
err = np.zeros((r,b))
err_fixed = np.zeros((r,b))
# Record LT training errors
train_err = np.zeros((r,b))
train_err_fixed = np.zeros((r,b))
l = 0
for i in states:
    k = 0
    naive_vec = np.zeros(b)
    fixed_vec = np.zeros(b)
    train_naive = np.zeros(b)
    train_fixed = np.zeros(b)
    for j in T:
        M, _ =createrandomCTMC(i-1,broken=False)
        ts, tseq = sampleCTMC(M,L,1)
        ts_test, tseq_test = sampleCTMC(M,j,1)
        ts[L] = i-1
        ts_test[j] = i - 1
        LTerr, LTerr_test,_ = test_naiveCTMC(ts,tseq,ts_test,tseq_test,wantST=False)
        LTerr_fixed, LTerr_test_fixed,constrviol,_,_,sparsity= test_fixedCTMC(ts,tseq,ts_test,tseq_test,wantST=False)
        
        train_naive[k] = LTerr
        train_fixed[k] = LTerr_fixed
        naive_vec[k] = LTerr_test
        fixed_vec[k] = LTerr_test_fixed
        k += 1
    train_err[l,] = train_naive
    train_err_fixed[l,] = train_fixed
    err[l,] = naive_vec
    err_fixed[l,] = fixed_vec
    l += 1


ind = np.array([500,1000,10000,100000,1000000]) 
ind1 = ind - 1 

train = pd.DataFrame(train_err,index=states,columns=ind)
train = train.rename_axis('# of states').rename_axis('CTMC simulation',axis='columns')
fixed = pd.DataFrame(train_err_fixed,index=states,columns=ind)
fixed = fixed.rename_axis('# of states').rename_axis('CTMC simulation',axis='columns')
dat = pd.DataFrame(err,index=states,columns=ind)
dat = dat.rename_axis('# of states').rename_axis('CTMC simulation',axis='columns')
dat_fixed = pd.DataFrame(err_fixed,index=states,columns=ind)
dat_fixed = dat_fixed.rename_axis('# of states').rename_axis('CTMC simulation',axis='columns')

print(train.to_latex(column_format='lccccc'))
print(fixed.to_latex(column_format='lccccc'))
print(dat.to_latex(column_format='lccccc'))
print(dat_fixed.to_latex(column_format='lccccc'))

naive_ctmc = {}
fixed_ctmc = {}
naive_ctmc['training_err'] = train_err
naive_ctmc['test_err'] = err
fixed_ctmc['training_err'] = train_err_fixed
fixed_ctmc['test_err'] = err_fixed
io.savemat('lts_naive_ctmc',naive_ctmc)
io.savemat('lts_fixed_ctmc',fixed_ctmc)
