import numpy as np
import scipy.linalg
import scipy.interpolate
import cvxopt
import mosek
import time
# from cvxopt import matrix, spmatrix, solvers
# from cvxopt.modeling import op

def counttrans(ts, N=0):
    """ counts transitions in a discrete state time series.
    
    Parameters
    ----------
    :param ts: array of integers

        This array must be a standardized discrete state time series.  The states (i.e., the integers in the array) must belong to the set :math:`\\{0, 1, \\ldots, N-1\\}`.  Passing any array that allows indexing using square brackets should be fine; this includes built-in Python arrays and NumPy arrays.
    :param N: integer, optional

        Total number of states in the Markov chain.  Note that it is possible to have an observed time series `\\{0, 1, 0, 1\\}` with a total number of `N = 3` states.  In this example, the state `2` is unobserved.  If the user does not set `N`, it will be set to max(ts)+1.
        
    Returns
    -------
    :returns: np.array

        Returns an :math:`N \\times N` matrix where the `(i,j)` entry records the number of times we observe the transition `i` to `j` in the time series.

    """
    if N > 0:
        assert (max(ts)+1) <= N
    else:
        N = max(ts) + 1
    transmat = np.zeros((N, N), dtype='int32')
    numts = len(ts)
    for i in range(1, numts):
        transmat[ts[i-1], ts[i]] += 1
    return transmat

# function to standardize a discrete state time series;
# note that the ordering of states is arbitrary.
# e.g.: you start with ['a','t','g','c',...]
# you get as output [1, 3, 2, 0, ...]
# (let's also return as output the mapping)
def standardize(tsin):
    states = list(set(tsin))
    myd = {z : states.index(z) for z in states}
    tsout = list(map(lambda x: myd[x], tsin))
    return tsout, myd

# function to tally time spent in each state
# inputs need to be:
# -- sequence of states (a standardized discrete state time series)
# -- sequence of times
# -- N=total number of states
# assumptions:
# -- both sequences have the same length
# -- timeseq[0] is the initial time
# -- timeseq[i] is the time at which we jump from state dsts[i-1] to dsts[i]
def statetimetally(dsts, timeseq, N):
    assert len(dsts) == len(timeseq)
    assert (max(dsts)+1) <= N
    sttally = np.zeros(N)
    diffseq = np.diff(timeseq)
    numts = len(dsts)
    for i in range(numts-1):
        sttally[dsts[i]] += diffseq[i]
    return sttally

# just like statetimetally except that instead of summing the total time
# spent in state i, we create a list of all the times spent in state i
def statetimegather(dsts, timeseq, N):
    assert len(dsts) == len(timeseq)
    assert (max(dsts)+1) <= N
    stgather = [[]]*N
    diffseq = np.diff(timeseq)
    numts = len(dsts)
    for i in range(numts-1):
        if not stgather[dsts[i]]:
            stgather[dsts[i]] = [diffseq[i]]
        else:
            stgather[dsts[i]].append(diffseq[i])
    return stgather

# dtmc training (raw MLE)
# we should check whether diagonal is zero for semi-Markov
# returns normalized Markov transition matrix
def trainDTMC(dsts, N=0):
    rawcounts = counttrans(dsts, N)
    transmat = 1.0*rawcounts
    deadendstates = []
    for i in range(transmat.shape[0]):
        if np.sum(rawcounts[i,:]) == 0:
            transmat[i,i] = 1.0
            deadendstates.append(i)
        else:
            transmat[i,] /= np.sum(transmat[i,])
    return transmat, deadendstates

# ctmc training (raw MLE)
def trainCTMC(dsts, timeseq, N):
    transratemat = counttrans(dsts, N)
    for i in range(N):
        assert transratemat[i, i] == 0
    deadendstates = []
    for i in range(transratemat.shape[0]):
        if np.sum(transratemat[i,:]) == 0:
            deadendstates.append(i)
    # make sure diagonal counts are zero
    # and divide each row by the time spent in that state
    # and then set diagonal entry
    transratemat = 1.0*transratemat
    sttally = statetimetally(dsts, timeseq, N)
    for i in range(N):
        if np.sum(transratemat[i,:]) != 0:
            transratemat[i,] /= sttally[i]
            rowsum = np.sum(transratemat[i,])
            transratemat[i, i] = -rowsum
    return transratemat, deadendstates

# dtmc/ctmc equilibrium vector
# we ***know*** the eigenvalue is either 1 or 0
# we seek the left eigenvector with this eigenvalue
# simple implementation of inverse power iteration,
# http://bit.ly/2sVNaXo
def equilib(m, whichone, x = None):
    n = m.shape[0]
    mT = m.T
    if whichone == "DTMC":
        p, l, u = scipy.linalg.lu(mT - np.identity(n))
        eigval = 1.0
    elif whichone == "CTMC":
        p, l, u = scipy.linalg.lu(mT)
        eigval = 0.0
    else:
        raise ValueError('whichone must either be "DTMC" or "CTMC"')

    mnorm = scipy.linalg.norm(mT)
    #print("Equilib says the norm is " + str(mnorm))
    machineeps = np.finfo(float).eps
    mythresh = mnorm * machineeps
    for i in range(n):
        if abs(u[i, i]) < mythresh:
            u[i, i] = mythresh

    # x is the initial guess for the eigenvector solve
    # if x is not specified, we compute a cheap guess
    if x is None:
        # x = np.linalg.solve(u, np.ones(n))
        # x /= np.sum(np.abs(x))
        x = np.ones(n) / n
        for i in range(25):
            x = np.dot(mT, x)
    done = False
    iters = 0
    maxiters = 10
    while not done:
        xnew = np.linalg.solve(u, np.linalg.solve(l, np.dot(p, x)))
        xnew = np.abs(xnew)
        xnew /= np.sum(xnew)
        if np.sum(np.abs(xnew - x)) < mythresh:
            done = True
        if np.sum(np.abs(np.dot(xnew, m) - eigval*xnew)) < mythresh:
            done = True
        x = xnew
        iters += 1
        if iters > maxiters:
            return None
    return xnew

# helper function for Li's exponential KDE
# this is the "kernel" function, i.e., PDF of Exp(1) R.V.:
# equals exp(-u) only for nonnegative elements of u, 0 otherwise
def fix_element(u):
    k = np.zeros(len(u))
    k[u >= 0] = np.exp(-u[u >= 0])
    return k

# Li's exponential KDE
# also returns CDF corresponding to the KDE PDF
# warning: need O(100) points for len(u),
# before the KDE passes the "eye test"
def expkde(u, npts, r0=-1, r1=-1):
    if (r0 == -1) or (r1 == -1) or (r0 >= r1):
        r0 = min(u)
        r1 = max(u)*1.3

    t = np.linspace(r0, r1, npts)
    n = len(u)

    # initialize KDE and CDF to be vectors of 0's
    kde = np.zeros(npts)
    cdf = np.zeros(npts)

    # set bandwidth using exp-optimal formula in paper
    h = (1/4)**(1/5)*np.mean(u)*n**(-1/5)

    # add one data point at a time
    for j in range(n):
        block = fix_element((t - u[j])/h)
        myeval = block/h
        kde += myeval
        cdf += (t > u[j])*(1.0 - block)

    return (t, kde/n, cdf/n)

# semimarkov training (DTMC MLE + exponential KDE)
def trainSMC(dsts, timeseq, N):
    dtmc, _ = trainDTMC(dsts, N)
    # note that we do not check whether
    stg = statetimegather(dsts, timeseq, N)
    densities = []
    for u in stg:
        densities.append(expkde(u, 100))
    return dtmc, densities

# remaining steps:
# if there are N states, need to build N functions of the form:
# f = interp1d(cdf, grid)
# f(np.random.rand())
# this will tell you how long to spend in a given state before transitioning
# need to track the cumulative times and return those together with 
# the discrete state time series
def sampleSMC(M, dens, n, init):
    outts = np.zeros(n, dtype='int32')
    sojourns = np.zeros(n)
    cdfmat = np.zeros(M.shape)
    cdfmat[:,0] = M[:,0]
    for i in range(M.shape[1] - 1):
        cdfmat[:,(i+1)] = cdfmat[:,i] + M[:,(i+1)]
    sojourner = []
    for den in dens:
        sojourner.append(scipy.interpolate.interp1d(den[2], den[0], copy=False, bounds_error=False, fill_value=(0.,1.), assume_sorted=True))       
    curstate = init
    for j in range(n):
        sojourns[j] = sojourner[curstate](np.random.rand())
        currow = cdfmat[curstate,:]
        u = np.random.rand(1)
        outts[j] = list(currow > u).index(True)
        curstate = outts[j]
    tseq = np.cumsum(sojourns)
    return outts, tseq
 
def createDTMC(t,n):
    Malt = np.zeros((n,n))
    for i in range(n):
        if (np.sum(t[i,])!=0):
            Malt[i,] = t[i,]/np.sum(t[i,])
        else:
            Malt[i,i] = 1.0
    return Malt
        

# given t, an n x n matrix of transition counts, and
# given w, an n x 1 vector of time spent in each state,
# this function returns a CTMC transition rate matrix
def createCTMC(t, w, n):
    Malt = np.zeros((n, n))
    for i in range(n):
        if (np.sum(t[i,])!= 0):
            Malt[i,] = t[i,]/w[i]
            rowsum = np.sum(Malt[i,])
            Malt[i, i] = -rowsum
        else:
            Malt[i,] = t[i,]
    return Malt

# drop the diagonal elements from M, and
# reshape into vector with n*(n-1) entries
def mShape(M, n):
    Mnew = np.zeros(n*(n-1))
    for i in range(n):
        thisRange = np.concatenate((range(0, i), range(i+1, n)), axis=0).astype(int)
        Mnew[(i*(n-1)):((i+1)*(n-1))] = M[i, thisRange]
    return Mnew

def ctmcG(n, forcePos = False):
    nb = 4 
    eye = np.repeat(1.0, n*(n-1))
    y = np.tile(np.arange(2*n*(n-1)), nb)
    x = np.tile(np.arange(n*(n-1)), nb*2)
    x += np.repeat(np.arange(nb)*n*(n-1), 2*n*(n-1))
    if forcePos:
        ws = np.concatenate([eye, -eye, -eye, -eye, -eye, 0*eye, 0*eye, -eye, -eye]) 
        y = np.concatenate([y, np.arange(n*(n-1))])
        x = np.concatenate([x, nb*n*(n-1) + np.repeat(np.arange(n),n-1)])
        G = cvxopt.spmatrix(ws, x, y, ((nb*n*(n-1)) + n, 2*n*(n-1)))
    else:
        ws = np.concatenate([eye, -eye, -eye, -eye, -eye, 0*eye, 0*eye, -eye])
        G = cvxopt.spmatrix(ws, x, y, (nb*n*(n-1), 2*n*(n-1)))
    return G

def ctmcA(n, wvec):
    x1 = []
    ws1 = []
    for i in range(n):
        for j in range(n):
            if j != i:
                x1.append(j)
        for jj in range(n-1):
            ws1.append(wvec[i])
    x1 = np.array(x1)
    ws1 = np.array(ws1)
    y1 = np.arange(n*(n-1))
    x2 = np.repeat(np.arange(n), n-1) 
    y2 = np.arange(n*(n-1))
    ws2 = np.repeat(-wvec, n-1)
    ws = np.concatenate([ws1, ws2])
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    A = cvxopt.spmatrix(ws, x, y, (n, 2*n*(n-1)))
    return A

def ctmcB(w, M):
    b = -np.dot(M.T,w)
    b = cvxopt.matrix(b.T)
    return b

def ctmcH(n, M, forcePos = False):
    M_new = mShape(M, n)
    l = n*(n-1)
    myzeros = np.zeros(l)
    h = np.concatenate((myzeros, myzeros, M_new, myzeros)) #, axis=1)
    if forcePos:
        delta = 1.0e-6
        hextra = np.zeros(n)
        for i in range(n):
            hextra[i] = -delta + np.sum(M_new[(i*(n-1)):((i+1)*(n-1))])
        h = np.concatenate([h, hextra])
    h = np.expand_dims(h,1)
    return cvxopt.matrix(h)

def createC(n):
    l = n*(n-1)
    c = np.concatenate((np.zeros(l), np.ones(l)), axis=0)
    c = cvxopt.matrix(c)
    return c

def createrandomCTMC(n, broken=True):
    w = (np.random.randint(50, 501, n)).astype(float)
    transitions = np.reshape(np.random.randint(1, 501, n*n), (n, n))
    for i in range(n):
        transitions[i, i] = 0.0
    if broken:
        transitions[(n-1), :] = 0.0
    M = createCTMC(transitions, w, n)
    w = w/np.sum(w)
    return M, w

def createrandomDTMC(n, broken=True):
    w = None
    keepGoing = True
    #iternum = 0
    while keepGoing:
        transitions = np.reshape(np.random.randint(1, 501, n*n).astype(float), (n, n))
        for i in range(n):
            transitions[i, :] /= np.sum(transitions[i, :])
        w = equilib(transitions, "DTMC")
        if w is not None:
            keepGoing = False
        #iternum += 1

    if broken:
        transitions[(n-1), :] = 0.0
        transitions[(n-1), (n-1)] = 1.0
    return transitions, w

# pass in a transition rate matrix M with at least one dead-end state
# also pass in w, the desired equilibrium vector
# fixCTMC runs the optimization problem described in paper1.pdf
# if it finds an optimal solution, it returns it.  this optimal solution
# consists of the perturbation \varepsilon which, if added to M in the right
# way, will yield a continuous-time transition rate matrix that achieves the
# equilibrium w.  The perturbation \varepsilon should be sparse, i.e., the
# optimizer will try to maximize the number of entries that are zero.
def fixCTMC(M, w, forcePos = False):
    n = len(w)
    G = ctmcG(n, forcePos)
    A = ctmcA(n, w)
    b = ctmcB(w, M)
    h = ctmcH(n, M, forcePos)
    c = createC(n)
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['mosek'] = {mosek.iparam.log: 0}
    sol = cvxopt.solvers.lp(c, G, h, A, b, solver='mosek')
    if sol['status'] == "optimal":
        return sol['x'], sol['primal infeasibility']
    return None

def addPert(M, eps, n, whichone):
    if whichone == "CTMC":
        preadd = 0.0
    elif whichone == "DTMC":
        preadd = 1.0
    else:
        raise ValueError('whichone must either be "DTMC" or "CTMC"')

    Mout = np.copy(M)
    counter = 0
    for row in range(n):
        for col in range(n):
            if row == col:
                continue
            else:
                Mout[row, col] = M[row, col] + eps[counter]
                counter += 1
    for i in range(n):
        Mout[i, i] = preadd - np.sum(Mout[i, 0:i]) - np.sum(Mout[i, (i+1):n])
    return Mout

def dtmcA(n, wvec):
    x1 = []
    ws1 = []
    for i in range(n):
        for j in range(n):
            if j != i:
                x1.append(j)
        for jj in range(n-1):
            ws1.append(wvec[i])
    x1 = np.array(x1)
    ws1 = np.array(ws1)
    y1 = np.arange(n*(n-1))
# NEW ADDITION, BECAUSE WE NEED THE DIAGONAL ENTRIES
    x2 = np.repeat(np.arange(n), n-1)
    y2 = np.arange(n*(n-1))
    ws2 = np.repeat(-wvec, n-1)
    ws = np.concatenate([ws1, ws2])
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    A = cvxopt.spmatrix(ws, x, y, (n, 2*n*(n-1)))
    return A

def dtmcG(n):
    nb = 4 # number of blocks, used to be 4
    eye = np.repeat(1.0, n*(n-1))
    ws1 = np.concatenate([eye, -eye, -eye, -eye, eye, 0*eye, -eye, 0*eye]) # ,0*eye,-eye])
    y1 = np.tile(np.arange(2*n*(n-1)), nb)
    x1 = np.tile(np.arange(n*(n-1)), nb*2)
    x1 += np.repeat(np.arange(nb)*n*(n-1), 2*n*(n-1))
# NEW CONSTRAINTS:
    x2 = np.repeat(np.arange(n), n-1) + nb*n*(n-1)
    y2 = np.arange(n*(n-1))
    x3 = x2 + n
    ws = np.concatenate([ws1, -eye, eye])
    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y2])
    G = cvxopt.spmatrix(ws, x, y, ((nb*n*(n-1) + 2*n), 2*n*(n-1)))
    return G

def dtmcH(n, w, M):
    zero = np.repeat(0.0, n*(n-1))
    mvec = mShape(M, n)
    h = np.concatenate([zero, zero, 1.0 - mvec, mvec, (w - np.diag(M)), np.diag(M)]) 
    return cvxopt.matrix(h)

def dtmcB(w, M):
    check = np.array(w - np.dot(M.T, w)) # + np.diag(M)*w)
    if len(check.shape) > 1:
        b1 = check.reshape(M.shape[0])
    else:
        b1 = check
    return cvxopt.matrix(b1)

# pass in a transition matrix M with at least one dead-end state
# also pass in w, the desired equilibrium vector
# fixDTMC runs the optimization problem described in paper1.pdf
# if it finds an optimal solution, it returns it.  this optimal solution
# consists of the perturbation \varepsilon which, if added to M in the right
# way, will yield a discrete-time transition matrix that achieves the
# equilibrium w.  The perturbation \varepsilon should be sparse, i.e., the
# optimizer will try to maximize the number of entries that are zero.
def fixDTMC(M, w):
    n = len(w)
    G = dtmcG(n)
    h = dtmcH(n, w, M)
    A = dtmcA(n, w)
    b = dtmcB(w, M)
    c = createC(n)
    cvxopt.solvers.options['show_progress']= False
    cvxopt.solvers.options['mosek'] = {mosek.iparam.log: 0}
    sol = cvxopt.solvers.lp(c, G, h, A, b, solver='mosek')
    if sol['status'] == "optimal":
        return sol['x'], sol['primal infeasibility']
    return None

# sample from a given DTMC 
# need M, the transition matrix
# need n, the total length of desired output time series
# need init, the initial state
# returns NumPy array (of integer type) of length n+1,
# where the 0th entry is the initial state

def createCDFmat(M):
    cdfmat = np.zeros(M.shape)
    for i in range(M.shape[1] - 1):
        cdfmat[:,(i+1)] = cdfmat[:,i] + M[:,(i+1)]
    return cdfmat

def sampleDTMC1(cdfmat,n,init):
    outts = np.zeros(n+1, dtype = 'int32')
    outts[0] = init
    for j in range(n):
        currow = cdfmat[outts[j],:]
        u = np.random.rand(1)
        outts[j+1] = np.argmax(currow > u)
    return outts
 
def sampleDTMC(M, n, init):
    outts = np.zeros(n+1, dtype='int32')
    outts[0] = init
    cdfmat = np.zeros(M.shape)
    cdfmat[:,0] = M[:,0]
    for i in range(M.shape[1] - 1):
        cdfmat[:,(i+1)] = cdfmat[:,i] + M[:,(i+1)]
    for j in range(n):
        currow = cdfmat[outts[j],:]
        u = np.random.rand(1)
        #outts[j+1] = list(currow > u).index(True)
        outts[j+1] = np.where(currow > u)[0][0]
    return outts

# introduce dead-end states into a given DTMC
# at the moment, we trust that the entries of deadlist
# correspond to actual rows of M --> need some bounds checking later
def breakDTMC(M, deadlist):
    Mout = M.copy()
    for i in deadlist:
        Mout[i,:] = 0.
        Mout[i,i] = 1.
    return Mout

# given a time series of total length n,
# take an input parameter m <= (n-1).
# for each of the subsets of time series consisting of the first
# m, m+1, ..., n-1 points, train a DTMC and then do both 1-step prediction
def testnaiveDTMC(ts, m):
    numpredict = len(ts) - m
    STerrors = np.zeros(numpredict, dtype='int32')
    LTerrors = np.zeros(numpredict)
    numstates = max(ts) + 1
    statefreq = np.zeros(numstates, dtype='int32')
    for i in range(m):
        statefreq[ts[i]] += 1
    for i in range(m,len(ts)):
        phat, _ = trainDTMC(ts[0:i], numstates)
        prediction = sampleDTMC(phat, 1, ts[i-1])
        if prediction != ts[i]:
            STerrors[i-m] = 1
        thisequilib = equilib(phat, "DTMC")
        if thisequilib is None:
            LTerrors[i-m] = np.nan
        else:
            statefrac = (1.0*statefreq)/np.sum(statefreq)
            LTerrors[i-m] = np.sum(np.abs(thisequilib - statefrac))
        statefreq[ts[i]] += 1
    
    avgSTerrors = np.mean(STerrors)
    avgLTerrors = np.mean(LTerrors)
    return STerrors, LTerrors, avgSTerrors, avgLTerrors

# given a time series of total length n,
# take an input parameter m <= (n-1).
# for each of the subsets of time series consisting of the first
# m, m+1, ..., n-1 points, train a DTMC and then do both 1-step prediction
def testfixedDTMC(ts, m):
    numpredict = len(ts) - m
    STerrors = np.zeros(numpredict, dtype='int32')
    LTerrors = np.zeros(numpredict)
    numstates = max(ts) + 1
    statefreq = np.zeros(numstates, dtype='int32')
    for i in range(m):
        statefreq[ts[i]] += 1
    for i in range(m,len(ts)):
        phat, des = trainDTMC(ts[0:i], numstates)
        statefrac = (1.0*statefreq)/np.sum(statefreq)
        if len(des) > 0:
            eps, _ = fixDTMC(phat, statefrac)
            if eps is None:
                STerrors[i] = np.nan
                LTerrors[i] = np.nan
            else:
                l = numstates*(numstates - 1)
                phat = addPert(phat,eps[0:l],numstates,"DTMC")
        else:
            print("No Dead End States!")
        prediction = sampleDTMC(phat, 1, ts[i-1])
        if prediction != ts[i]:
            STerrors[i-m] = 1
        thisequilib = equilib(phat, "DTMC")
        if thisequilib is None:
            LTerrors[i-m] = np.nan
        else:
            LTerrors[i-m] = np.sum(np.abs(thisequilib - statefrac))
        statefreq[ts[i]] += 1
    avgSTerrors = np.mean(STerrors)
    avgLTerrors = np.mean(LTerrors)
    return STerrors, LTerrors, avgSTerrors, avgLTerrors

# sample from a given CTMC 
# need M, the transition matrix
# need n, the total length of desired output time series
# returns list of discrete state time series,
#   which has maximum length n+1 and contains integers,
#   with initial state stored in the 0th position
# also returns cumulative sequence of times
def sampleCTMC(M, n, init):
    outts = np.zeros(n+1, dtype='int32')
    touts = np.zeros(n+1)
    outts[0] = init
    touts[0] = 0.0
    # M = np.array(M)
    # for i in range(M.shape[0]):
    # M[i,i] = 0
    reachedDeadEnd = False
    for j in range(n):
        # indices of positive entries
        posind = np.where(M[outts[j],:] > 0)[0]
        # Dead-end state means exit immediately
        if len(posind) == 0:
            reachedDeadEnd = True
            break
        # pick up positive entries of the current row
        posentries = M[outts[j], posind]
        # sample from exponential distirbution with rates 1/posentries
        u = np.random.exponential(1 / posentries, len(posentries))
        tmin = np.min(u)
        # pick up the state index corresponding to the minimum u
        outts[j+1] = posind[np.argmin(u)]
        touts[j+1] = touts[j] + tmin

    fi = j + 2
    if reachedDeadEnd:
        fi -= 1

    return outts[0:fi], touts[0:fi]

# Test naive CTMC
# Need state series, time series, training size m and total # of states
def testnaiveCTMC(ts, timeseq, m):
    if np.any(timeseq == np.Inf):
        cutoff = np.searchsorted(timeseq, np.Inf)
        ts = ts[:cutoff]
        timeseq = timeseq[:cutoff]
    if len(ts) <= m:
        m = len(ts) - 1
    numpredict = len(ts) - m
    STerrors = np.zeros(numpredict, dtype='int32')
    LTerrors = np.zeros(numpredict)
    numstates = max(ts) + 1
    statefreq = np.zeros(numstates, dtype='int32')
    for i in range(m):
        statefreq[ts[i]] += 1
    for i in range(m, len(ts)):
        phat, _ = trainCTMC(ts[0:i], timeseq[0:i], numstates)
        prediction, _ = sampleCTMC(phat, 1, ts[i - 1])
        if prediction != ts[i]:
            STerrors[i - m] = 1
 
        """thisequilib = equilib(phat, "CTMC")
        if thisequilib is None:
            LTerrors[i - m] = np.nan
            print(phat)
        else:"""
        statefrac = (1.0 * statefreq) / np.sum(statefreq)
        LTerrors[i - m] = np.sum(np.abs(np.dot(phat.T,statefrac)))
 
        statefreq[ts[i]] += 1
    
    ave_STerr = np.mean(STerrors)
    ave_LTerr = np.mean(LTerrors)
    return STerrors, LTerrors, ave_STerr, ave_LTerr


# Test fixed CTMC
def testfixedCTMC(ts, timeseq, m):
    if np.any(timeseq == np.Inf):
        cutoff = np.searchsorted(timeseq, np.Inf)
        ts = ts[:cutoff]
        timeseq = timeseq[:cutoff]
    # m must be less than length of ts or timeseq
    numpredict = len(ts) - m
    STerrors = np.zeros(numpredict, dtype='int32')
    LTerrors = np.zeros(numpredict)
    numstates = max(ts) + 1
    statefreq = np.zeros(numstates, dtype='int32')
    for i in range(m):
        statefreq[ts[i]] += 1
    for i in range(m, len(ts)):
        phat, des = trainCTMC(ts[0:i], timeseq[0:i], numstates)
        statefrac = (1.0 * statefreq) / np.sum(statefreq)
        if len(des) > 0:
            eps = fixCTMC(phat, statefrac)
            if eps is None:
                STerrors[i - m] = np.nan
                LTerrors[i - m] = np.nan
            else:
                l = numstates * (numstates - 1)
                phat = addPert(phat, eps[0:l], numstates, 'CTMC')
        else:
            # print('No Dead End States!')
            r = np.sum(phat, axis=1)
            ind = np.argmin(r)
            phat[ind, ] = np.zeros(numstates)
            eps = fixCTMC(phat, statefrac)
            if eps is None:
                STerrors[i - m] = np.nan
                LTerrors[i - m] = np.nan
            else:
                l = numstates * (numstates - 1)
                phat = addPert(phat, eps[0:l], numstates, 'CTMC')
 
        prediction, _ = sampleCTMC(phat, 1, ts[i - 1])
        if prediction != ts[i]:
            STerrors[i - m] = 1
        LTerrors[i - m] = np.sum(np.abs(np.dot(phat.T, statefrac)))
        if LTerrors[i - m] > 10e-02:
            print(i)
            print(LTerrors[i - m])
            # print(thisequilib)
            print(phat)
            print(eps)
        statefreq[ts[i]] += 1
    
    ave_STerr = np.mean(STerrors)
    ave_LTerr = np.mean(LTerrors)
    return STerrors, LTerrors, ave_STerr, ave_LTerr



# New method for testing DTMC and CTMC without minimum training length
# Generate two indepedent time series from the same Markov chains.
# Train on one time series and test on the other.
# DTMC does not require time sequence
# CTMC requires time sequences

# Test naive DTMC (new method)
def test_naiveDTMC(ts, ts_test, wantST = True):
    start = time.time()
    m = len(ts)
    n = len(ts_test)
    numstates = max(max(ts),max(ts_test))+1
    STerrors = np.zeros(n-1, dtype='int32')
    phat, _ = trainDTMC(ts, numstates)
    
    statefreq = np.zeros(numstates, dtype='int32')
    statefreq_test = np.zeros(numstates, dtype = 'int32')
    for i in range(m):
        statefreq[ts[i]] += 1
    for i in range(n):
        statefreq_test[ts_test[i]] +=1

    # compute LT training and test errors
    statefrac = (1.0*statefreq)/np.sum(statefreq)
    pvec = equilib(phat, "DTMC", statefrac)
    train_errLT = np.sum(np.abs(pvec - statefrac))
    training_time = time.time()-start
    statefrac_test = (1.0*statefreq_test)/np.sum(statefreq_test)
    test_errLT = np.sum(np.abs(pvec - statefrac_test))
    
    if wantST:
        # ST test error; testing/predictions on test ts
        # New way: vectorized version: 
        # 1st: sample n-1 points from uniform distribution
        #u = np.random.randn(n-1,1)
        #cdfmat = createCDFmat(phat)
        #predictions = np.argmax(u > cdfmat)
        #STerrors = predictions != ts_test[1:]
        #STerrors.astype(int)
        cdfmat = createCDFmat(phat)        
        for i in range(1,n):
            prediction = sampleDTMC1(cdfmat, 1, ts_test[i-1])
            if prediction[1] != ts_test[i]:
                STerrors[i-1] = 1
        
        avgSTerrors = np.mean(STerrors)
        return train_errLT, test_errLT, STerrors, avgSTerrors, training_time
    else:
        return train_errLT, test_errLT, training_time

# Test fixed DTMC (new method)
def test_fixedDTMC(ts, ts_test, wantST = True):
    start = time.time()
    m = len(ts)
    n = len(ts_test)
    STerrors = np.zeros(n-1, dtype='int32')
    preds = np.zeros(n-1,dtype = 'int32')
    preds[0] = ts_test[0]
    numstates = max(max(ts),max(ts_test)) + 1
    statefreq = np.zeros(numstates, dtype='int32')
    statefreq_test = np.zeros(numstates, dtype='int32')
    for i in range(m):
        statefreq[ts[i]] += 1
    for i in range(n):
        statefreq_test[ts_test[i]] += 1

    phat, des = trainDTMC(ts, numstates)
    statefrac = (1.0*statefreq)/np.sum(statefreq)
    
    if len(des) > 0:
        eps, constrviol = fixDTMC(phat, statefrac)
        if eps is None:
             STerrors = -999
             LTerrors = -999
           
        else:        
             l = numstates*(numstates - 1)
             phat = addPert(phat,eps[0:l],numstates,"DTMC")
             sparsity = np.sum(np.ravel(eps[0:l])!= 0.)/len(np.ravel(eps[0:l]))
    else:
        print("No Dead End States!")
        
    pvec = equilib(phat, "DTMC", statefrac)
    if pvec is None:
        print(phat)
        trainLTerr = np.nan
        testLTerr = np.nan
        STerrors = np.nan
        avgSTerrors = np.nan
        return trainLTerr, testLTerr, STerrors, avgSTerrors, constrviol
      
    # compute LT training and test errors
    trainLTerr = np.sum(np.abs(pvec - statefrac))
    training_time = time.time()-start
    statefrac_test = (1.0*statefreq_test)/np.sum(statefreq_test)
    testLTerr = np.sum(np.abs(pvec - statefrac_test))

    if wantST:
        # compute ST test error
        for i in range(1,n):
            prediction = sampleDTMC(phat, 1, ts_test[i-1])
            preds[i-1] = prediction[1]
            if prediction[1] != ts_test[i]:
                STerrors[i-1] = 1
    
        avgSTerrors = np.mean(STerrors)
        if (len(des) == 0):
            return trainLTerr, testLTerr, preds, STerrors, avgSTerrors,training_time
        else:
            return trainLTerr, testLTerr, preds,STerrors, avgSTerrors, constrviol,training_time, sparsity
    else:
        if (len(des)==0):
            return trainLTerr, testLTerr, training_time
        else:
            return trainLTerr, testLTerr, constrviol, training_time, sparsity

#Test naive CTMC (new method)
def test_naiveCTMC(ts, tseq, ts_test, tseq_test, wantST = True):
    start = time.time()
    numpredict = len(ts_test)
    # prediction errors 
    STerrors = np.zeros(numpredict-1, dtype='int32')

    numstates = max(ts) + 1
    
    phat, des = trainCTMC(ts, tseq, numstates)
    sttally = statetimetally(ts,tseq,numstates)
    statefrac = (sttally) / np.sum(sttally)

    # new way: explicitly compute the equilibrium vector \pi of phat
    #          and calculate || \pi - statefrac || in the L^1 norm
    pivec = equilib(phat, "CTMC")

    # compute LT training and test errors
    LTerror_training = np.sum(np.abs(pivec - statefrac))
    training_time = time.time()-start
    sttally_test = statetimetally(ts_test,tseq_test,numstates)
    statefrac_ts_test = sttally_test/ np.sum(sttally_test)
    LTerror_test = np.sum(np.abs(pivec - statefrac_ts_test))

    if wantST:
        # compute ST test error, i.e., "prediction"
        for i in range(1,len(ts_test)):
            prediction, _ = sampleCTMC(phat, 1, ts_test[i - 1])
            if len(prediction) == 1:
                STerrors[i-1] = 1
            elif prediction[1] != ts_test[i]:
                STerrors[i-1] = 1
    
        ave_STerr = np.mean(STerrors)
        return LTerror_training, LTerror_test, STerrors, ave_STerr, training_time
    else:
        return LTerror_training, LTerror_test, training_time

# Test fixed CTMC (new method)
def test_fixedCTMC(ts, tseq, ts_test, tseq_test, wantST = True):
    start = time.time()
    numpredict = len(ts_test)
    # prediction errors 
    STerrors = np.zeros(numpredict-1, dtype='int32')

    numstates = max(ts) + 1
    #numstates1 = max(ts_test) + 1 
    phat, des = trainCTMC(ts, tseq, numstates)
    sttally = statetimetally(ts,tseq,numstates)
    statefrac = (sttally) / np.sum(sttally)
    #print(des)
    if len(des) > 0:
        eps, constrviol = fixCTMC(phat, statefrac, forcePos = True)
        if eps is None:
            STerror = -999
            LTerror = -999
        else:
            l = numstates * (numstates - 1)
            phat = addPert(phat, eps[0:l], numstates, 'CTMC')
            sparsity = np.sum(np.ravel(eps[0:l])!= 0.)/len(np.ravel(eps[0:l]))
    else:
        print('No Dead End States!')
    
    # new way: see above
    pivec = equilib(phat, "CTMC")

    # compute LT training and test errors
    LTerror_training = np.sum(np.abs(pivec - statefrac))
    training_time = time.time()-start
    sttally_test = statetimetally(ts_test,tseq_test,numstates)
    statefrac_ts_test = sttally_test/ np.sum(sttally_test)
    LTerror_test = np.sum(np.abs(pivec - statefrac_ts_test))

    if wantST:
        # compute ST test error, i.e., "prediction"
        for i in range(1,len(ts_test)):
            prediction, _ = sampleCTMC(phat, 1, ts_test[i - 1])
            #prediction_test[i-1] = prediction
            if len(prediction) == 1:
                STerrors[i-1] = 1
            elif prediction[1] != ts_test[i]:
                STerrors[i-1] = 1
            
        ave_STerr = np.mean(STerrors)
        if (len(des)==0):
            return LTerror_training, LTerror_test, STerrors, ave_STerr, pivec, training_time
        else:
            return LTerror_training, LTerror_test, STerrors, ave_STerr, constrviol, pivec, training_time, sparsity
    else:
        if (len(des)==0):   
            return LTerror_training, LTerror_test, pivec, training_time
        else:
            return LTerror_training, LTerror_test, constrviol, pivec, training_time, sparsity

#Test large states naive CTMC (new method)
def bigstate_naiveCTMC(ts, tseq, ts_test,tseq_test):
   
    numstates_ts = max(ts) + 1
    numstates_ts_test = max(ts_test) + 1
    numstates = max(numstates_ts, numstates_ts_test)
    phat, des = trainCTMC(ts, tseq, numstates)
    sttally = statetimetally(ts,tseq,numstates)
    statefrac = (sttally) / np.sum(sttally)


    # new way: explicitly compute the equilibrium vector \pi of phat
    #          and calculate || \pi - statefrac || in the L^1 norm
    pivec = equilib(phat, "CTMC")
    #print("Naive pivec")
    #print(pivec)
    LTerror_training = np.sum(np.abs(pivec - statefrac))
    
    #Compute long-term test errors
    sttally_test = statetimetally(ts_test,tseq_test,numstates)
    statefrac_ts_test = sttally_test/ np.sum(sttally_test)
    LTerror_test = np.sum(np.abs(pivec - statefrac_ts_test))
    
    return LTerror_training, LTerror_test


# Test fixed CTMC (new method)
def bigstate_fixedCTMC(ts, tseq, ts_test,tseq_test):
    

    numstates_ts = max(ts) + 1
    numstates_ts_test = max(ts_test) + 1
    numstates = max(numstates_ts, numstates_ts_test)
    
    phat, des = trainCTMC(ts, tseq, numstates)
    sttally = statetimetally(ts,tseq,numstates)
    statefrac = (sttally) / np.sum(sttally)
    #print(des)
    if len(des) > 0:
        eps = fixCTMC(phat, statefrac, forcePos = True)
        if eps is None:
            STerror = -999
            LTerror = -999
        else:
            l = numstates * (numstates - 1)
            phat = addPert(phat, eps[0:l], numstates, 'CTMC')
    else:
        print('No Dead End States!')
    
    # old way:
    # LTerror_training = np.sum(np.abs(np.dot(phat.T, statefrac)))

    # new way: see above
    pivec = equilib(phat, "CTMC")
    #print("Fixed pivec")
    #print(pivec)
    LTerror_training = np.sum(np.abs(pivec - statefrac))
    
    #Compute long-term test errors
    sttally_test = statetimetally(ts_test,tseq_test,numstates)
    statefrac_ts_test = sttally_test/ np.sum(sttally_test)

    # new way: see above
    LTerror_test = np.sum(np.abs(pivec - statefrac_ts_test))
    
    return LTerror_training, LTerror_test
