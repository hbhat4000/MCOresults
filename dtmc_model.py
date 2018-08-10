from mcoLH import *
import pandas as pd

# Specify the states
states = [2**x for x in range(2,9)]

# number of states
r = len(states)

# training size
L = 1000
# test size 
T = 3*L

# number of simulation
N = 1000
naive_avgerr = []
fixed_avgerr = []
numsim = np.zeros(r)
k = 0
for i in states:
    naive_vec = np.zeros(N)
    fixed_vec = np.zeros(N)
    for j in range(N):
        M, _ =createrandomDTMC(i-1,broken=False)
        ts = sampleDTMC(M,L,1)
        ts_test = sampleDTMC(M,T,1)
        ts[L] = i-1
        ts_test[T] = i - 1
        LTerr_training, LTerr_test,_,STerr,_ = test_naiveDTMC(ts,ts_test)
        LTerr_training_fixed, LTerr_test_fixed,_,_,STerr_fixed, constrviol,_,_= test_fixedDTMC(ts,ts_test)
        naive_vec[j] = STerr
        fixed_vec[j] = STerr_fixed
    naive_avgerr.append(np.cumsum(naive_vec)/np.arange(1,L+1))   
    fixed_avgerr.append(np.cumsum(fixed_vec)/np.arange(1,L+1))
    avg_fixed_diff = np.abs(fixed_avgerr - np.mean(fixed_vec))
    avg_naive_diff = np.abs(naive_avgerr - np.mean(naive_vec))
    numsim[k] = np.where(avg_fixed_diff < 0.001)[0][-1]
    k += 1
print(numsim)
#print(naive_avgerr)
#print(fixed_avgerr)
ind = np.array([50,250,500,1000])
ind1 = ind - 1
err = np.zeros((r,len(ind)))
err_fixed = err.copy()
for i in range(r):
    err[i,] = naive_avgerr[i][ind1]
    err_fixed[i,] = fixed_avgerr[i][ind1]
dat = pd.DataFrame(err,index=states,columns=ind)
dat = dat.rename_axis('# of states').rename_axis('DTMC simulation',axis='columns')
dat_fixed = pd.DataFrame(err_fixed,index=states,columns=ind)
dat_fixed = dat_fixed.rename_axis('# of states').rename_axis('DTMC simulation',axis='columns')
print(dat.to_latex(column_format='lcccc'))
print(dat_fixed.to_latex(column_format='lcccc'))
