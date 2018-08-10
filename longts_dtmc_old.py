from mcoLH import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
constviol = []
sparsity_ratio = []
numsim = np.zeros(r)
err = np.zeros((r,b))
err_fixed = np.zeros((r,b))
l = 0
for i in states:
    k = 0
    naive_vec = np.zeros(b)
    fixed_vec = np.zeros(b)
    for j in T:
        M, _ =createrandomDTMC(i-1,broken=False)
        ts = sampleDTMC(M,L,1)
        ts_test = sampleDTMC(M,j,1)
        ts[L] = i-1
        ts_test[j] = i - 1
        _, LTerr_test,_ = test_naiveDTMC(ts,ts_test,wantST=False)
        _, LTerr_test_fixed,constrviol,_,sparsity= test_fixedDTMC(ts,ts_test,wantST=False)
        
        naive_vec[k] = LTerr_test
        fixed_vec[k] = LTerr_test_fixed
        k += 1
    err[l,] = naive_vec
    err_fixed[l,] = fixed_vec
    l += 1

ind = np.array([500,1000,10000,100000,1000000]) 
ind1 = ind - 1 

dat = pd.DataFrame(err,index=states,columns=ind)
dat = dat.rename_axis('# of states').rename_axis('DTMC simulation',axis='columns')
dat_fixed = pd.DataFrame(err_fixed,index=states,columns=ind)
dat_fixed = dat_fixed.rename_axis('# of states').rename_axis('DTMC simulation',axis='columns')
print(dat.to_latex(column_format='lccccc'))
print(dat_fixed.to_latex(column_format='lccccc'))



