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
L = 1000
#L = 1000
# test size 
T = 3*L

# number of simulation
N = 1000
naive_avgerr = []
fixed_avgerr = []
constviol = []
sparsity_ratio = []
numsim = np.zeros(r)
k = 0
for i in states:
    naive_vec = np.zeros(N)
    fixed_vec = np.zeros(N)
    constr_vec = np.zeros(N)
    sparse_vec = np.zeros(N)
    for j in range(N):
        M, _ =createrandomCTMC(i-1,broken=False)
        ts, tseq = sampleCTMC(M,L,1)
        ts_test, tseq_test = sampleCTMC(M,T,1)
        ts[L] = i-1
        ts_test[T] = i - 1
        LTerr_training, LTerr_test,_,STerr,_ = test_naiveCTMC(ts,tseq,ts_test,tseq_test)
        LTerr_training_fixed, LTerr_test_fixed,_,STerr_fixed, constrviol,_,_,sparsity= test_fixedCTMC(ts,tseq,ts_test,tseq_test)
        naive_vec[j] = STerr
        fixed_vec[j] = STerr_fixed
        constr_vec[j] = constrviol
        sparse_vec[j] = sparsity
    naive_avgerr.append(np.cumsum(naive_vec)/np.arange(1,N+1))   
    fixed_avgerr.append(np.cumsum(fixed_vec)/np.arange(1,N+1))
    #plt.scatter(np.arange(1,N+1),naive_avgerr,c='blue')
    #plt.axhline(np.mean(naive_vec))
    #plt.savefig('%dstates_naive.eps'%i)
    avg_fixed_diff = np.abs(fixed_avgerr - np.mean(fixed_vec))
    avg_naive_diff = np.abs(naive_avgerr - np.mean(naive_vec))
    constviol.append(np.cumsum(constr_vec)/np.arange(1,N+1))
    sparsity_ratio.append(np.cumsum(sparse_vec)/np.arange(1,N+1))
    numsim[k] = np.where(avg_fixed_diff < 0.001)[0][-1] 
    #numsim_naive = np.where(avg_naive_diff < 0.01)[0][-1]       
    k += 1
print(numsim)
#print(naive_avgerr)
#print(fixed_avgerr)
#ind = np.array([50,250,500,1000])

#ind = np.array([10,20,30])
ind = np.array([50,250,500,1000]) 
ind1 = ind - 1 
err = np.zeros((r,len(ind)))
err_fixed = err.copy()
for i in range(r):
    err[i,] = naive_avgerr[i][ind1]
    err_fixed[i,] = fixed_avgerr[i][ind1]
dat = pd.DataFrame(err,index=states,columns=ind)
dat = dat.rename_axis('# of states').rename_axis('CTMC simulation',axis='columns')
dat_fixed = pd.DataFrame(err_fixed,index=states,columns=ind)
dat_fixed = dat_fixed.rename_axis('# of states').rename_axis('CTMC simulation',axis='columns')
print(dat.to_latex(column_format='lcccc'))
print(dat_fixed.to_latex(column_format='lcccc'))



