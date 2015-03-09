##SETUP
import os
import sys
import posix
sys.path.append(os.path.join(posix.environ['HOME'], 'mlprojects', 'collab', 'python'))
sys.path.append(os.path.join(posix.environ['HOME'], 'mlprojects', 'mltools', 'python'))
sys.path.append(os.path.join(posix.environ['HOME'], 'mlprojects', 'swig', 'src'))
##ENDSETUP

import pdb
import time
import pyflix.datasets
import numpy as np
import ndlml
import math
import netlab
import optimi
import mltools
from ndlutil import *

def dataDir(dataSetName):
    baseName = os.path.join('/local', 'data', 'datasets')
    if dataSetName == 'netflix':
        return os.path.join(baseName, 'netflix', 'pyflix')
    elif dataSetName == 'movielens100k1':
        return os.path.join(baseName, 'movielens100k', 'pyflix1')
    elif dataSetName == 'movielens100k2':
        return os.path.join(baseName, 'movielens100k', 'pyflix2')
    elif dataSetName == 'movielens100k3':
        return os.path.join(baseName, 'movielens100k', 'pyflix3')
    elif dataSetName == 'movielens100k4':
        return os.path.join(baseName, 'movielens100k', 'pyflix4')
    elif dataSetName == 'movielens100k5':
        return os.path.join(baseName, 'movielens100k', 'pyflix5')

def resultsDir(dataSetName):
    baseName = os.path.join('/local', 'data', 'results')
    if dataSetName == 'netflix':
        return os.path.join(baseName, 'netflix')
    elif dataSetName == 'movielens100k1':
        return os.path.join(baseName, 'movielens100k1')
    elif dataSetName == 'movielens100k2':
        return os.path.join(baseName, 'movielens100k2')
    elif dataSetName == 'movielens100k3':
        return os.path.join(baseName, 'movielens100k3')
    elif dataSetName == 'movielens100k4':
        return os.path.join(baseName, 'movielens100k4')
    elif dataSetName == 'movielens100k5':
        return os.path.join(baseName, 'movielens100k5')



class iterInfo:
    """A class containing information for the current iteration."""
    def __init__(self, tic, tic0, count, numSparse, userOrder, iterNum):
        self.tic = tic
        self.tic0 = tic0
        self.count = count
        self.numSparse = numSparse
        self.userOrder = userOrder
        self.iterNum = iterNum

class dataSet:
    """A class containing information about the data set."""
    def __init__(self, data, numItems, itemMean, itemStd, itemCount, learnRateAdjust):
        self.data = data
        self.numItems = numItems
        self.itemMean = itemMean
        self.itemStd = itemStd
        self.itemCount = itemCount
        self.learnRateAdjust = learnRateAdjust
  

class params:
    """A class containing all the model parameters."""
    def __init__(self, X, X_u, param, lnsigma2, lnbeta, fullKern=None, sparseKern=None, noise=None):
        self.X = X
        self.X_u = X_u
        self.param = param
        self.lnsigma2 = lnsigma2
        self.lnbeta = lnbeta
        self.fullKern = fullKern
        self.sparseKern = sparseKern
        self.noise = noise
        

class options:
    """Class containing the options for the collaborative filtering."""
    # For learning rate anealling.
    lambdaVal = 0.01
    maxLearnRate = 1
    t0 = 400000
    momentum = 0.9
    
    # Initial white noise variance.
    startVariance = 5
    baseKern = 'rbf'
    
    heteroNoise = False # Whether to have different noise for each film.
    # Active set size and maximum size for FTC.
    runSparse = True
    numActive = 100
    maxFTC = 500
    sparseApprox = ndlml.gp.DTCVAR

    seed = 10000

    # How often to print status
    showEvery = 5000

    # How often to save status 
    saveEvery= 20000
    numIters = 10
    resultsBaseDir = None
    def __init__(self, dataSetName = 'netflix'):
        self.resultsBaseDir = resultsDir(dataSetName)

class ppca(mltools.probabilisticmodel):
    """A pppca style model for collaborative filtering, takes advantage of the sparse structure of the data, """
    linVariance = 1.0
    biasVariance = 0.11
    whiteVariance = 5.0
    def __init__(self, latentDim, numData, 
                 linVariance=1.0, biasVariance=0.11, whiteVariance=5.0, heteroNoise=False):
        mltools.probabilisticmodel.__init__(self)
        self.X = np.random.normal(0.0, 1e-6, (numData, latentDim))
        self.latentDim = latentDim
        self.numData = numData
        self.linVariance = linVariance
        self.biasVariance = biasVariance
        self.indices = None
        self.y = None
        self.heteroNoise = heteroNoise   # Are we using separate noise for each item
        if self.heteroNoise:
            self.d = np.asmatrix(np.tile(whiteVariance, (numData, 1)))
        else:
            self.whiteVariance = whiteVariance

    def __str__(self):
        if self.heteroNoise:
            return "Heteroscadastic PPCA model, Lin var " + str(self.linVariance) + ", Bias var " + str(self.biasVariance)
        else:
            return "PPCA model, Lin var " + str(self.linVariance) + ", Bias var " + str(self.biasVariance) + ", noise var " + str(self.whiteVariance)

    def extractParam(self):
        """Get the parameters of the PPCA model."""
        if self.heteroNoise:
            return np.concatenate((np.asarray(self.X[self.indices]).flatten(),
                                  np.asarray([self.linVariance]), 
                                  np.asarray([self.biasVariance]), 
                                  np.asarray(self.d[self.indices, :]).flatten()))
        else:
            return np.concatenate((np.asarray(self.X[self.indices]).flatten(), 
                                   np.asarray([self.linVariance]), 
                                   np.asarray([self.biasVariance]), 
                                   np.asarray([self.whiteVariance])))
                              
    def expandParam(self, parameters):
        """Set the parameters of the PPCA model using a vector of the parameters."""
        N = len(self.indices)
        start = 0
        end = self.latentDim*N
        self.X[self.indices, :] = parameters[start:end].reshape(N,self.latentDim).copy()
        start = end
        end = end + 1
        self.linVariance = float(parameters[start])
        start = end
        end = end + 1
        self.biasVariance = float(parameters[start])
        start = end
        end = end + N
        if self.heteroNoise:
            self.d[self.indices, :] = parameters[start:end].reshape(N, 1).copy()
        else:
            self.whiteVariance = float(parameters[start])
        

    def logLikelihood(self):
        """Return the log likelihood of the model."""
        Cinvy, logdetC = self.invCovProducts()
        N = len(self.indices)
        ym = np.asmatrix(self.y)
        if self.heteroNoise:
            ym = np.multiply(ym, 1/np.sqrt(self.d[self.indices, :]))
        ll = -0.5*(N*math.log(2*math.pi)
                   + logdetC
                   + ym.T*Cinvy)
        return float(ll)
        
    def invCovProducts(self, requireX=False, useSlowMethod=False):
        """For requireX set to False, returns inverse covariance
        multiplied by y and the log determinant of the inverse
        covariance (useful for log likelihood computation). For
        requireX set to True, returns the inverse of the covariance
        matrix multiplied by y, the inverse covariance summed across
        one axis, the inverse covariance multiplied by X and the trace
        of the inverse covariance (useful for gradient
        computations). Calling this function avoids having to compute
        full covariance matrices over the data (leading to O(N^2)
        storage). Instead, here, the storage requirements are
        O(qN)."""
        s_w = self.linVariance
        s_b = self.biasVariance
        if self.heteroNoise:
            s_n = 1.0    
        else:
            s_n = self.whiteVariance
        sigNoiseRatio = s_w/s_n
        q = self.latentDim
        N = len(self.indices)
        Xm = np.asmatrix(self.X[self.indices, :])
        ym = np.asmatrix(self.y)


        if self.heteroNoise:
            # noise values for the observations.
            h = np.asmatrix(1/np.sqrt(self.d[self.indices, :]))
            # If the noise is heteroscadastic scale the Xs and ys
            Xm = np.multiply(Xm, h)  
            ym = np.multiply(ym, h)

        if (N>q and not useSlowMethod) or (N<=q and useSlowMethod): 
            XTX = Xm.T*Xm                        # this is qxq
            B = np.eye(q)+ sigNoiseRatio*XTX     # this is qxq
            Binv, U = pdinv(B)[0:2]
            logdetB = logdet(B, U)[0]
            if requireX:
                AinvX = (Xm - sigNoiseRatio*Xm*(Binv*XTX))/s_n       #this is Nxq
                if self.heteroNoise:
                    # This is actually the diagonal for heteroscadastic
                    AinvTr = (np.ones((N, 1)) - sigNoiseRatio*(np.multiply(Xm*Binv,Xm)).sum(axis=1))
                else:
                    AinvTr = (N - sigNoiseRatio*(np.multiply(Xm*Binv,Xm)).sum())/s_n  # this is 1x1
            Ainvy = (ym - sigNoiseRatio*Xm*(Binv*(Xm.T*ym)))/s_n     # this is Nx1
            if self.heteroNoise:
                sumAinv = (h - sigNoiseRatio*Xm*(Binv*(Xm.T*h)))/s_n 
                sumAinvSum = float(h.T*sumAinv)
            else:
                sumAinv = (np.ones((N, 1)) - sigNoiseRatio*Xm*(Binv*Xm.sum(axis=0).T))/s_n # this is Nx1
                sumAinvSum = sumAinv.sum()
            
            denom = 1.0 + s_b*sumAinvSum
            fact = s_b/denom
            if requireX:
                CinvX = AinvX - fact*sumAinv*(sumAinv.T*Xm)
                CinvSum = sumAinv - fact*sumAinv*sumAinvSum
                if self.heteroNoise:
                    CinvTr = AinvTr - fact*np.multiply(sumAinv,sumAinv)
                else:
                    CinvTr = AinvTr - fact*sumAinv.T*sumAinv

            Cinvy = Ainvy - fact*sumAinv*float(sumAinv.T*ym)
            if not requireX:
                logdetA = N*math.log(s_n) + logdetB
                logdetC = logdetA + math.log(denom)
                if self.heteroNoise:
                    logdetC = logdetC - 2*np.log(h).sum() 
        else:
            C = s_w*Xm*Xm.T 
            if self.heteroNoise:
                C = C + s_b*h*h.T + np.eye(N)
            else:
                C = C + s_b + s_n*np.eye(N)
            Cinv, U = pdinv(C)[0:2]
            Cinvy = Cinv*ym
            if requireX:
                CinvX = Cinv*Xm
                if self.heteroNoise:
                    CinvTr = np.asmatrix(np.diag(Cinv)).T
                else:
                    CinvTr = Cinv.trace()
                if self.heteroNoise:
                    CinvSum = Cinv*h
                else:
                    CinvSum = Cinv.sum(axis=1)
            else:
                logdetC = logdet(C, U)[0]
                if self.heteroNoise:
                    logdetC = logdetC - 2*np.log(h).sum() 
        if requireX:
            return Cinvy, CinvSum, CinvX, CinvTr
        else:
            return Cinvy, logdetC
        



    def inverseCovariance(self, useSlowMethod=False):
        """Computes the inverse of the covariance matrix and the log
        determinant of the covariance matrix for the mode."""
        q = self.latentDim
        N = len(self.indices)

        s_w = self.linVariance
        s_b = self.biasVariance
        if self.heteroNoise:
            s_n = 1.0
        else:
            s_n = self.whiteVariance

        # Latent coordinates.
        Xm = np.asmatrix(self.X[self.indices, :])
        if self.heteroNoise:
            # Define h as 1/sqrt(d)
            h = np.asmatrix(1/np.sqrt(self.d[self.indices, :]))
            # If the noise is heteroscadastic scale the Xs
            Xm = np.multiply(Xm, h)
            
        if (N>q and not useSlowMethod) or (N<=q and useSlowMethod):
            XTX = Xm.T*Xm
            B = np.eye(q)+ s_w/s_n*XTX
            Binv, U = pdinv(B)[0:2]
            logdetB = logdet(B, U)[0]
            Ainv =  (np.eye(N) - s_w/s_n*Xm*Binv*Xm.T)/s_n
            if self.heteroNoise:
                sumAinv = np.multiply(Ainv, h).sum(axis=1)
                sumAinvSum = np.multiply(sumAinv, h).sum()
            else:
                sumAinv = Ainv.sum(axis=1)
                sumAinvSum = sumAinv.sum()


            denom = 1.0 + s_b*sumAinvSum
            Cinv = Ainv - sumAinv*sumAinv.T*s_b/denom
            if self.heteroNoise:
                # pre and post multiply by D^{-1/2}
                Cinv = np.multiply(np.multiply(Cinv, h), h.T)
            if self.heteroNoise:
                logdetA = logdetB
            else:
                logdetA = N*math.log(s_n) + logdetB

            logdetC = logdetA + math.log(denom)
            if self.heteroNoise:
                logdetC = logdetC - 2*np.log(h).sum() 
        else:

            if self.heteroNoise:
                C = s_w*Xm*Xm.T + s_b*h*h.T + np.eye(N)
            else:
                C = s_w*Xm*Xm.T + s_b + s_n*np.eye(N)
            Cinv, U = pdinv(C)[0:2]
            logdetC = logdet(C, U)[0]
            if self.heteroNoise:
                Cinv = np.multiply(np.multiply(Cinv, h), h.T)
                logdetC = logdetC + np.log(self.d[self.indices, :]).sum()
        return Cinv, logdetC

    def logLikeGradient(self, ):
        """Computes the gradient of the log likelihood with respect to
        the parameters."""
        s_w = self.linVariance
        s_b = self.biasVariance
        if self.heteroNoise:
            s_n = 1.0
            sqrtD = np.sqrt(self.d[self.indices, :])
        else:
            s_n = self.whiteVariance
        
        ym = np.asmatrix(self.y)
        Xm = np.asmatrix(self.X[self.indices, :])

        if self.heteroNoise:
            # noise values for the observations.
            h = np.asmatrix(1/np.sqrt(self.d[self.indices, :]))
            # If the noise is heteroscadastic scale the Xs
            Xm = np.multiply(Xm, h)
            ym = np.multiply(ym, h)

        Cinvy, CinvSum, CinvX, CinvTr = self.invCovProducts(True)
        covGradXm = 0.5*(Cinvy*(Cinvy.T*Xm) - CinvX)
        gX = s_w*2.0*covGradXm
        gsigma_w = np.multiply(covGradXm, Xm).sum()
        if self.heteroNoise:
            CinvySum = h.T*Cinvy
            CinvSumSum = h.T*CinvSum
            gsigma_b = 0.5*(CinvySum*CinvySum - CinvSumSum)
            gX = np.multiply(h, gX)
            gd = 0.5*np.multiply(1/self.d[self.indices, :],
                                 (np.multiply(Cinvy, Cinvy) - CinvTr))
        else:
            CinvySum = Cinvy.sum()
            CinvSumSum = CinvSum.sum()
            gsigma_b = 0.5*(CinvySum*CinvySum - CinvSumSum)
            gsigma_n = 0.5*(Cinvy.T*Cinvy - CinvTr)
        if self.heteroNoise:
            return gX, float(gsigma_w), float(gsigma_b), gd
        else:
            return gX, float(gsigma_w), float(gsigma_b), float(gsigma_n)

    def gradients(self, params=None):
        """Gradients of the objective function with respect to the
        parameters of the model. Here it simply returns the negative
        of the log likelihood gradient."""

        if params is not None:
            self.expandParam(params)
        gX, gsigma_w, gsigma_b, gsigma_n = self.logLikeGradient()
        if self.heteroNoise:
            return np.concatenate((-np.asarray(gX).flatten(), 
                                    np.asarray([-gsigma_w]), 
                                    np.asarray([-gsigma_b]), 
                                    -np.asarray(gsigma_n).flatten()))
        else:
            return np.concatenate((-np.asarray(gX).flatten(), 
                                    np.asarray([-gsigma_w]), 
                                    np.asarray([-gsigma_b]), 
                                    np.asarray([-gsigma_n])))

    def objective(self, params=None):
        """Return the objective function for the model --- here it is
        simply the negative log likelihood."""
        if params is not None:
            self.expandParam(params)
        return -self.logLikelihood()

def restartPpca(loadIter, startCount, loadUser, latentDim, dataSetName, experimentNo, options):
    """Restart a collaborative filtering model from a crashed PPCA run."""

    o = options
    np.random.seed(seed=o.seed)

    isNetflix = False
    if dataSetName=="netflix":
        isNetflix = True
    
    d = loadData(dataSetName)
    
    resultsDir = os.path.join(o.resultsBaseDir, dataSetName + str(experimentNo))
    if not os.path.exists(resultsDir):
        os.mkdir(resultsDir)
    loadDir1 = "iter" + str(loadIter)
    userOrder = np.fromfile(file=os.path.join(resultsDir, 
                                              loadDir1, 
                                              "userOrder"),
                            dtype=int)
    
    
    loadDir2 = "count" + str(startCount) + "_user" + str(loadUser)
    
    loadDir = os.path.join(resultsDir, loadDir1, loadDir2)
    
    if o.heteroNoise:
        model, Xchange, paramChange, dchange = loadPpcaResults(loadDir, 
                                                               latentDim, 
                                                               o)
    else:
        model, Xchange, paramChange = loadPpcaResults(loadDir, 
                                                      latentDim, 
                                                      o)

    param = np.log(np.array([model.linVariance,
                             model.biasVariance,
                             model.whiteVariance]))


    numSparse = 0
    print "Restarting PPCA from iteration ", loadIter, " count ", startCount, " ... "
    tic = time.time()
    tic0 = tic

    for iter in range(loadIter, o.numIters):
        
        saveDir = "iter" + str(iter)
        iterDir = os.path.join(resultsDir, saveDir)
        if iter>loadIter:
            # Ensure repeatability
            state = np.random.get_state()
            # Order users randomly
            userOrder = np.random.permutation(d.data.userIDs())

            if not os.path.exists(iterDir):
                os.mkdir(iterDir)

            userOrder.tofile(os.path.join(iterDir, "userOrder" ))
            param.tofile(os.path.join(iterDir, "param" ))
            model.X.tofile(os.path.join(iterDir, "X" ))
            if model.heteroNoise:
                model.d.tofile(os.path.join(iterDir, "d" ))
                dchange.tofile(os.path.join(iterDir, "dchange"))
            Xchange.tofile(os.path.join(iterDir, "Xchange" ))
            paramChange.tofile(os.path.join(iterDir, "paramChange" ))

        info = iterInfo(tic = tic, 
                        tic0 = tic0, 
                        count = startCount, 
                        numSparse = numSparse, 
                        userOrder = userOrder,
                        iterNum = iter)
        runPpcaIter(dataSet = d, 
                    model = model,
                    Xchange = Xchange,
                    paramChange = paramChange, 
                    options = o,
                    iterInfo = info,
                    iterDir = iterDir)

        startCount = info.count
    # Save state for repeatability
    saveDir = "final"
    iterDir = os.path.join(resultsDir, saveDir)
    if not os.path.exists(iterDir):
        os.mkdir(iterDir)
    param.tofile(os.path.join(iterDir, "param" ))
    model.X.tofile(os.path.join(iterDir, "X" ))
    if model.heteroNoise:
        model.d.tofile(os.path.join(iterDir, "d" ))
        dchange.tofile(os.path.join(iterDir, "dchange"))
        
    Xchange.tofile(os.path.join(iterDir, "Xchange" ))
    paramChange.tofile(os.path.join(iterDir, "paramChange" ))



    
        
def runPpca(latentDim, dataSetName, experimentNo, options):
    """Initialize and run the PPCA collaborative filtering model."""

    o = options
    np.random.seed(seed=o.seed)

    isNetflix = False
    if dataSetName=="netflix":
        isNetflix = True
    
    d = loadData(dataSetName)

    resultsDir = os.path.join(o.resultsBaseDir, dataSetName + str(experimentNo))
    if not os.path.exists(o.resultsBaseDir):
        os.mkdir(o.resultsBaseDir)
    if not os.path.exists(resultsDir):
        os.mkdir(resultsDir)


    model = ppca(latentDim, 17770, heteroNoise=o.heteroNoise)

    Xchange = np.zeros(model.X.shape)
    if model.heteroNoise:
        dchange = np.zeros(model.d.shape)
        param = np.log(np.array([model.linVariance,
                                 model.biasVariance]))
        paramChange = np.zeros((1, 2))
    else:        
        dchange = None
        param = np.log(np.array([model.linVariance,
                                 model.biasVariance,
                                 model.whiteVariance]))
        paramChange = np.zeros((1, 3))

    print "Starting PPCA run ..."
    count = 0

    for iter in range(o.numIters):
        tic = time.time()
        tic0 = tic
        # Ensure repeatability
        state = np.random.get_state()
        # Order users randomly
        userOrder = np.random.permutation(d.data.userIDs())

        # Save state for repeatability
        saveDir = "iter" + str(iter)
        iterDir = os.path.join(resultsDir, saveDir)
        if not os.path.exists(iterDir):
            os.mkdir(iterDir)
        userOrder.tofile(os.path.join(iterDir, "userOrder" ))
        param.tofile(os.path.join(iterDir, "param" ))
        model.X.tofile(os.path.join(iterDir, "X" ))
        Xchange.tofile(os.path.join(iterDir, "Xchange" ))
        if model.heteroNoise:
            model.d.tofile(os.path.join(iterDir, "d" ))
            dchange.tofile(os.path.join(iterDir, "dchange" ))
        paramChange.tofile(os.path.join(iterDir, "paramChange" ))

        info = iterInfo(tic = tic, 
                        tic0 = tic0, 
                        count = count, 
                        numSparse = 0, 
                        userOrder = userOrder,
                        iterNum = iter)
        runPpcaIter(dataSet = d, 
                    model = model,
                    Xchange = Xchange,
                    paramChange = paramChange, 
                    dchange = dchange,
                    options = o,
                    iterInfo = info,
                    iterDir = iterDir)

        count = info.count

    # Save final results
    saveDir = "final"
    iterDir = os.path.join(resultsDir, saveDir)
    if not os.path.exists(iterDir):
        os.mkdir(iterDir)
    param.tofile(os.path.join(iterDir, "param" ))
    paramChange.tofile(os.path.join(iterDir, "paramChange" ))
    model.X.tofile(os.path.join(iterDir, "X" ))
    Xchange.tofile(os.path.join(iterDir, "Xchange" ))
    if model.heteroNoise:
        model.d.tofile(os.path.join(iterDir, "d" ))
        dchange.tofile(os.path.join(iterDir, "dchange" ))

def runPpcaIter(dataSet, model, Xchange, paramChange, dchange, options, iterInfo, iterDir):
    o = options
    d = dataSet
    it = iterInfo
    numUsers = len(it.userOrder)
    latentDim = model.X.shape[1]
    countShouldBe = iterInfo.iterNum*numUsers
    startCount = countShouldBe
    for user in it.userOrder:
        if it.count>countShouldBe:
            # for when we are restarting --- get count/user up to right value.
            countShouldBe = countShouldBe + 1
            startCount = countShouldBe
            continue

        countShouldBe = countShouldBe +1

        learnRate = 1.0/(o.lambdaVal*(it.count + o.t0))
        u = d.data.user(user)
        filmRatings = u.ratings()
        filmIndices = u.values()-1
        
        model.indices = filmIndices
        model.y = preprocessRatings(filmRatings, filmIndices, d)

        gX, gs_w, gs_b, gs_n = model.logLikeGradient()
        
        if model.heteroNoise:
            gd = np.multiply(gs_n, model.d[filmIndices, :])
            gparam = np.array([gs_w*model.linVariance, 
                               gs_b*model.biasVariance])
            param = np.log(np.array([model.linVariance,
                                     model.biasVariance]))
        else:
            gparam = np.array([gs_w*model.linVariance, 
                               gs_b*model.biasVariance,
                               gs_n*model.whiteVariance])
            param = np.log(np.array([model.linVariance,
                                     model.biasVariance,
                                     model.whiteVariance]))

        XTempChange = Xchange[filmIndices, :]
        adjustRates = learnRate*10.0
        XTempChange = XTempChange*o.momentum + gX*adjustRates

        Xchange[filmIndices, :] = XTempChange
        model.X[filmIndices, :] = model.X[filmIndices, :] + XTempChange

        if model.heteroNoise:
            dTempChange = dchange[filmIndices, :]
            dTempChange = dTempChange*o.momentum + gd*adjustRates

            dchange[filmIndices, :] = dTempChange
            model.d[filmIndices, :] = np.exp(np.log(model.d[filmIndices, :]) + dTempChange)

        paramChange = paramChange*o.momentum + gparam*learnRate
        param = param + paramChange
        model.linVariance = math.exp(param[0,0])
        model.biasVariance = math.exp(param[0,1])
        if model.heteroNoise:
            pass
        else:
            model.whiteVariance = math.exp(param[0,2])
        
        
        # Finished one iteration
        it.count = it.count + 1
        # Check if it's time to display
        if not np.remainder(it.count, o.showEvery):
            print "Count " + str(it.count)
            toc = time.time()
            eTime = toc - it.tic;
            totTime = toc - it.tic0;
            usersPerSecond = (it.count-startCount)/totTime
            remainUserIters = numUsers*o.numIters - it.count
            remainTime = remainUserIters/usersPerSecond
            it.tic = toc
            print("Remain time (hrs): " + str(remainTime/3600))
            sys.stdout.flush()

        # Check if it's time to save
        if not np.remainder(it.count, o.saveEvery):
            print("Saving file ...")
            toc = time.time()
            eTime = toc - it.tic;
            totTime = toc - it.tic0;
            usersPerSecond = it.count/totTime
            remainUserIters = numUsers*o.numIters - it.count
            remainTime = remainUserIters/usersPerSecond
            it.tic = toc
            print("Remain time (hrs): " + str(remainTime/3600))
            dirName = "count" + str(it.count) + "_user" + str(user)
            saveDir = os.path.join(iterDir, dirName)
            if not os.path.exists(saveDir):
                os.mkdir(saveDir)
            param.tofile(os.path.join(saveDir, "param" ))
            model.X.tofile(os.path.join(saveDir, "X" ))
            if model.heteroNoise:
                model.d.tofile(os.path.join(saveDir, "d" ))
                dchange.tofile(os.path.join(saveDir, "dchange"))
            Xchange.tofile(os.path.join(saveDir, "Xchange" ))
            paramChange.tofile(os.path.join(saveDir, "paramChange" ))
            sys.stdout.flush()




def loadData(dataSetName):
    """Load the given dataset."""
    isNetflix = True
        # load in data netflix.
    baseDir = dataDir(dataSetName)
    print "Loading data ..."
    data = pyflix.datasets.RatedDataset(os.path.join(baseDir, 'training_set'))
    itemIDs = data.movieIDs()
    numItems = data.movieIDs().shape[0]
    userIDs = data.userIDs()
    numUsers = data.userIDs().shape[0]
    if dataSetName[0:-1] == 'movielens100k':
        numItems = 1682
    
    d = dataSet(data = data,
                numItems = numItems,
                itemMean = np.zeros((numItems, 1)),
                itemStd = np.ones((numItems, 1)),
                itemCount = np.zeros((numItems, 1)),
                learnRateAdjust = np.ones((numItems, 1)))
    print "... done"

    # Compute item means and standard deviations.
    print "Computing item means ..."
    for item in itemIDs:
        ratings = d.data.movie(item).ratings()
        d.itemMean[item-1] = ratings.mean()
        d.itemStd[item-1] = ratings.std()
        d.itemCount[item-1] = ratings.shape[0]
        d.learnRateAdjust[item-1] = float(numUsers)/float(d.itemCount[item-1])
    print "done"
    return d



def extractKernType(latentDim, lnsigma2, options):
    """Extract the kernel types from the options and return as a tuple"""
    o = options

    fullKern = ndlml.cmpndKern(latentDim) 
    sparseKern = ndlml.cmpndKern(latentDim) 
    
    kt = type(o.baseKern)
    if kt==str:
        if o.baseKern == 'rbf':
            kern1 = ndlml.rbfKern(latentDim)
        elif o.baseKern == 'mlp':
            kern1 = ndlml.mlpKern(latentDim)
        elif o.baseKern == 'ratquad':
            kern1 = ndlml.ratquadKern(latentDim)
        elif o.baseKern == 'matern32':
            kern1 = ndlml.matern32Kern(latentDim)
        elif o.baseKern == 'matern52':
            kern1 = ndlml.matern52Kern(latentDim)
        elif o.baseKern == 'lin':
            kern1 = ndlml.linKern(latentDim)

    elif kt==ndlml.rbfkern \
            or kt==ndlml.mlpKern \
            or kt==ndlml.ratquadKern \
            or kt==ndlml.matern32Kern \
            or kt==ndlml.matern52Kern \
            or kt==ndlml.linKern \
            or kt==ndlml.polyKern \
            or kt==ndlml.cmpndKern:
        # Kernel has been provided as a class already.
        kern1 = o.baseKern
    
    kern2 = ndlml.biasKern(latentDim)
    kern3 = ndlml.whiteKern(latentDim)
    kern4 = ndlml.whitefixedKern(latentDim)

    kern2.setVariance(0.11)
    kern3.setVariance(math.exp(lnsigma2))
    kern4.setVariance(1e-2)

    fullKern.addKern(kern1)
    fullKern.addKern(kern2)
    fullKern.addKern(kern3)

    sparseKern.addKern(kern1)
    sparseKern.addKern(kern2)
    sparseKern.addKern(kern4)

    return fullKern, sparseKern


def loadResults(loadDir, latentDim, options):
    """Load in results from disk for the GPLVM model. Also load in parameter and X changes."""
    X = np.fromfile(os.path.join(loadDir, "X")).reshape(-1, latentDim)
    Xchange = np.fromfile(os.path.join(loadDir, "Xchange")).reshape(-1, latentDim)
    X_u = np.fromfile(os.path.join(loadDir, "X_u")).reshape(-1, latentDim)
    inducingChange = np.fromfile(os.path.join(loadDir, "inducingChange")).reshape(-1, latentDim)

    betaSigma = np.fromfile(os.path.join(loadDir, "betaSigma")).reshape(1, 4)
    lnsigma2 = betaSigma[0,1]
    lnbeta = betaSigma[0,0]
    lnsigma2Change = betaSigma[0,3]
    lnbetaChange = betaSigma[0,2]

    (fullKern, sparseKern) = extractKernType(latentDim, lnsigma2, options)
    
    numParams = fullKern.getNumParams()
    param = np.fromfile(os.path.join(loadDir, "param")).reshape(1, numParams)
    paramChange = np.fromfile(os.path.join(loadDir, "paramChange")).reshape(1, numParams)


    # Set log sigma2 (variance for FTC) and log beta (precision for sparse)
    # using this dummy y forces mean of Gaussian noise to be zero.
    dummyy = ndlml.matrix(10, 1)
    dummyy.zeros()

    # paramIter.fromarray(param)
    paramIter = ndlml.matrix(param.shape[0], param.shape[1])
    param[0, -1] = lnsigma2
#    paramIter.fromarray(param)
    for i in range(param.shape[1]):
        paramIter.setVal(param[0, i], i)
    fullKern.setTransParams(paramIter)

    paramIter = ndlml.matrix(param.shape[0], param.shape[1]-1)
    param2 = param[0, 0:-1]
#    paramIter.fromarray(param2)
    for i in range(param.shape[1]-1):
         paramIter.setVal(param[0, i], i)
    sparseKern.setTransParams(paramIter)

    # Set up parameters
    p = params(X = X, 
               X_u = X_u, 
               param = param, 
               fullKern = fullKern,
               sparseKern = sparseKern,
               lnsigma2 = lnsigma2, 
               lnbeta = lnbeta, 
               noise =  ndlml.gaussianNoise(dummyy))



    # Set up vectors for storing old changes.
    pc = params(X=Xchange, 
                X_u=inducingChange,
                param=paramChange, 
                lnsigma2=lnsigma2Change, 
                lnbeta=lnbetaChange)
    return p, pc


def loadPpcaResults(loadDir, latentDim, options):
    """Load X values and parameter vector in from files and return a PPCA model returning these values. Also return the Xchange and paramChange matrices from files"""
    X = np.fromfile(os.path.join(loadDir, "X")).reshape(-1, latentDim)
    Xchange = np.fromfile(os.path.join(loadDir, "Xchange")).reshape(-1, latentDim)

    param = np.fromfile(os.path.join(loadDir, "param")).reshape(1, -1)
    paramChange = np.fromfile(os.path.join(loadDir, "paramChange")).reshape(1, -1)
    if options.heteroNoise:
        d = np.fromfile(os.path.join(loadDir, "d")).reshape(-1, 1)
        dchange = np.fromfile(os.path.join(loadDir, "dchange")).reshape(-1, 1)
    model = ppca(latentDim, X.shape[1], heteroNoise = options.heteroNoise)
    model.X = X
    model.linVariance = math.exp(param[0,0])
    model.biasVariance = math.exp(param[0,1])
    if options.heteroNoise:
        model.d = d
        return model, Xchange, paramChange, dchange
    else:
        model.whiteVariance = math.exp(param[0,2])
        return model, Xchange, paramChange

        


def predictPpcaProbe(latentDim, dataSetName, experimentNo, loadIter, loadUser, loadCount, options):
    """Compute the RMSE error for the netflix probe set."""
    resultsDir = os.path.join(options.resultsBaseDir, "netflix" + str(experimentNo))

    loadDir1 = "iter" + str(loadIter)
    loadDir2 = "count" + str(loadCount) + "_user" + str(loadUser)

    loadDir = os.path.join(resultsDir, loadDir1, loadDir2)

    model = loadPpcaResults(loadDir, latentDim, options)[0]

    d = loadData(dataSetName)

    print "Loading netflix probe data ..."
    probe = pyflix.datasets.RatedDataset(os.path.join(dataDir(),'probe_set'))
    
    print "Testing PPCA exp no ", experimentNo, "iter ", loadIter, " count number ", loadCount
    total = 0
    totalSe = 0.0
    userIds = probe.userIDs()
    tic0 = time.time()
    tic = tic0
    numUsers = len(userIds)
    count = 0
    for user in np.sort(userIds):
        u = d.data.user(user)
        testLen = len(u.ratings())

        up = probe.user(user)

        filmRatings = up.ratings()
        filmIndices = up.values()-1
        
        pred = predPpcaVal(user, filmIndices, model, d)
        pred = np.multiply(pred, d.itemStd[filmIndices]) + d.itemMean[filmIndices]
        pred[np.nonzero(pred>5)] = 5
        pred[np.nonzero(pred<1)] = 1
        newSe = np.asarray(pred - filmRatings.reshape(len(filmRatings), -1))**2
        total += len(filmRatings)
        totalSe += newSe.sum()
        rmse = math.sqrt(totalSe/float(total))
        count = count + 1
        if not np.remainder(count, options.showEvery):
            toc = time.time()
            eTime = toc - tic;
            totTime = toc - tic0;
            usersPerSecond = count/totTime
            remainUserIters = numUsers - count
            remainTime = remainUserIters/usersPerSecond
            tic = toc
            print("Remain time (hrs): " + str(remainTime/3600))          
            print "Total Count: ", count, " rmse: ", rmse #, "pred ", pred, "True: ", filmRatings[count], "var: ", var

    print "Total Count: ", count, " rmse: ", rmse #, "pred ", pred, "True: ", filmRatings[count], "var: ", var
            
def predPpcaVal(user, testFilmIds, model, dataSet):
    """Make a prediction for the given user and the given film ID."""

    latentDim = model.X.shape[1]
    u = dataSet.data.user(user)
    filmRatings = u.ratings()
    filmIndices = u.values()-1
    Xtest = np.asmatrix(model.X[testFilmIds, :])
    # Not y --- scaled and centred y!
    model.indices = filmIndices
    Xm = np.asmatrix(model.X[model.indices, :])
    if model.heteroNoise:
        dsqrt = np.sqrt(model.d[model.indices, :])
        h = np.asmatrix(1/dsqrt)
        Xm = np.multiply(Xm, h)
    model.y =  preprocessRatings(filmRatings, filmIndices, dataSet)
    Cinvy, CinvSum, CinvX, CinvTr = model.invCovProducts(True)
    if model.heteroNoise:
        mean = model.linVariance*Xtest*(Xm.T*Cinvy) + h.T*Cinvy*model.biasVariance
    else:
        mean = model.linVariance*Xtest*(Xm.T*Cinvy) + Cinvy.sum()*model.biasVariance
#    varCentre = np.eye(latentDim) - Xm.T*CinvX
 #   var = np.multiply((Xtest*varCentre), Xtest).sum(axis=1)
#    pdb.set_trace()
    return mean


def restart(loadIter, startCount, loadUser, latentDim, dataSetName, experimentNo, options):
    """Restart a collaborative filtering model from a crashed run."""

    o = options
    np.random.seed(seed=o.seed)

    isNetflix = False
    if dataSetName=="netflix":
        isNetflix = True
    
    d = loadData(dataSetName)
    
    resultsDir = os.path.join(o.resultsBaseDir, dataSetName + str(experimentNo))
    if not os.path.exists(resultsDir):
        os.mkdir(resultsDir)
    loadDir1 = "iter" + str(loadIter)
    userOrder = np.fromfile(file=os.path.join(resultsDir, 
                                              loadDir1, 
                                              "userOrder"),
                            dtype=int)
    
    
    loadDir2 = "count" + str(startCount) + "_user" + str(loadUser)
    
    loadDir = os.path.join(resultsDir, loadDir1, loadDir2)
    
    p, pc = loadResults(loadDir, latentDim, o)


    numSparse = 0
    print "Restarting GPLVM from iteration ", loadIter, " count ", startCount, " ... "
    tic = time.time()
    tic0 = tic

    for iter in range(loadIter, o.numIters):
        
        saveDir = "iter" + str(iter)
        iterDir = os.path.join(resultsDir, saveDir)
        if iter>loadIter:
            # Ensure repeatability
            state = np.random.get_state()
            # Order users randomly
            userOrder = np.random.permutation(d.data.userIDs())

            if not os.path.exists(iterDir):
                os.mkdir(iterDir)

            userOrder.tofile(os.path.join(iterDir, "userOrder" ))
            p.param.tofile(os.path.join(iterDir, "param" ))
            p.X.tofile(os.path.join(iterDir, "X" ))
            p.X_u.tofile(os.path.join(iterDir, "X_u" ))
            pc.X.tofile(os.path.join(iterDir, "Xchange" ))
            pc.param.tofile(os.path.join(iterDir, "paramChange" ))
            pc.X_u.tofile(os.path.join(iterDir, "inducingChange" ))

        info = iterInfo(tic = tic, 
                        tic0 = tic0, 
                        count = startCount, 
                        numSparse = numSparse, 
                        userOrder = userOrder,
                        iterNum = iter)
        runIter(dataSet = d, 
                params = p, 
                paramChange = pc, 
                options = o,
                iterInfo = info,
                iterDir = iterDir)

        startCount = info.count
    # Save state for repeatability
    saveDir = "final"
    iterDir = os.path.join(resultsDir, saveDir)
    if not os.path.exists(iterDir):
        os.mkdir(iterDir)
    betaSigma = np.array([p.lnbeta, p.lnsigma2, pc.lnbeta, pc.lnsigma2])
    betaSigma.tofile(os.path.join(iterDir, "betaSigma"))
    p.param.tofile(os.path.join(iterDir, "param" ))
    p.X.tofile(os.path.join(iterDir, "X" ))
    p.X_u.tofile(os.path.join(iterDir, "X_u" ))
    pc.X.tofile(os.path.join(iterDir, "Xchange" ))
    pc.param.tofile(os.path.join(iterDir, "paramChange" ))
    pc.X_u.tofile(os.path.join(iterDir, "inducingChange" ))



    

def run(latentDim, dataSetName, experimentNo, options):
    """Initialize and run a collaborative filtering model."""

    o = options
    np.random.seed(seed=o.seed)

    isNetflix = False
    if dataSetName=="netflix":
        isNetflix = True
    
    d = loadData(dataSetName)

    resultsDir = os.path.join(o.resultsBaseDir, dataSetName + str(experimentNo))
    if not os.path.exists(o.resultsBaseDir):
        os.mkdir(o.resultsBaseDir)
    if not os.path.exists(resultsDir):
        os.mkdir(resultsDir)


    # Set log sigma2 (variance for FTC) and log beta (precision for sparse)
    # using this dummy y forces mean of Gaussian noise to be zero.
    dummyy = ndlml.matrix(10, 1)
    dummyy.zeros()

    (fullKern, sparseKern) = extractKernType(latentDim, math.log(o.startVariance), o)
    # Set up parameters
    p = params(X = np.random.normal(0.0, 1e-6, (d.numItems, latentDim)), 
               X_u = np.random.normal(0.0, 1e-6, (o.numActive, latentDim)), 
               param = None, 
               fullKern = fullKern,
               sparseKern = sparseKern,
               lnsigma2 = math.log(o.startVariance), 
               lnbeta = -math.log(o.startVariance), 
               noise =  ndlml.gaussianNoise(dummyy))


    # Set up parameter vectors
    paramNd = ndlml.matrix(1, p.fullKern.getNumParams())
    p.fullKern.getTransParams(paramNd)
    p.param = paramNd.toarray()

    # Set up vectors for storing old changes.
    pc = params(X=np.zeros((d.numItems, latentDim)), 
                X_u=np.zeros((o.numActive, latentDim)), 
                param=np.zeros((1, p.fullKern.getNumParams())), 
                lnsigma2=0, 
                lnbeta=0)


    numSparse = 0
    print "Starting GPLVM run ..."
    count = 0

    for iter in range(o.numIters):
        tic = time.time()
        tic0 = tic
        # Ensure repeatability
        state = np.random.get_state()
        # Order users randomly
        userOrder = np.random.permutation(d.data.userIDs())

        # Save state for repeatability
        saveDir = "iter" + str(iter)
        iterDir = os.path.join(resultsDir, saveDir)
        if not os.path.exists(iterDir):
            os.mkdir(iterDir)

        userOrder.tofile(os.path.join(iterDir, "userOrder" ))
        p.param.tofile(os.path.join(iterDir, "param" ))
        p.X.tofile(os.path.join(iterDir, "X" ))
        p.X_u.tofile(os.path.join(iterDir, "X_u" ))
        pc.X.tofile(os.path.join(iterDir, "Xchange" ))
        pc.param.tofile(os.path.join(iterDir, "paramChange" ))
        pc.X_u.tofile(os.path.join(iterDir, "inducingChange" ))

        info = iterInfo(tic = tic, 
                        tic0 = tic0, 
                        count = count, 
                        numSparse = numSparse, 
                        userOrder = userOrder,
                        iterNum = iter)
        runIter(dataSet = d, 
                params = p, 
                paramChange = pc, 
                options = o,
                iterInfo = info,
                iterDir = iterDir)

        count = info.count

    # Save state for repeatability
    saveDir = "final"
    iterDir = os.path.join(resultsDir, saveDir)
    if not os.path.exists(iterDir):
        os.mkdir(iterDir)
    betaSigma = np.array([p.lnbeta, p.lnsigma2, pc.lnbeta, pc.lnsigma2])
    betaSigma.tofile(os.path.join(iterDir, "betaSigma"))
    p.param.tofile(os.path.join(iterDir, "param" ))
    p.X.tofile(os.path.join(iterDir, "X" ))
    p.X_u.tofile(os.path.join(iterDir, "X_u" ))
    pc.X.tofile(os.path.join(iterDir, "Xchange" ))
    pc.param.tofile(os.path.join(iterDir, "paramChange" ))
    pc.X_u.tofile(os.path.join(iterDir, "inducingChange" ))



    



def runIter(dataSet, params, paramChange, options, iterInfo, iterDir):
    o = options
    d = dataSet
    p = params
    pc = paramChange
    it = iterInfo
    numUsers = len(it.userOrder)
    latentDim = p.X.shape[1]
    countShouldBe = iterInfo.iterNum*numUsers
    startCount = countShouldBe
    for user in it.userOrder:
        if it.count>countShouldBe:
            # for when we are restarting --- get count/user up to right value.
            countShouldBe = countShouldBe + 1
            startCount = countShouldBe
            continue

        countShouldBe = countShouldBe +1

        learnRate = 1.0/(o.lambdaVal*(it.count + o.t0))
        u = d.data.user(user)
        filmRatings = u.ratings()
        filmIndices = u.values()-1

        # Check whether we need to do sparse approximation.
        sparseApprox = False
        sparseFTC = False
        if len(filmRatings)>o.maxFTC:
            if not o.runSparse:
                continue
            if o.sparseApprox==ndlml.gp.FTC:
                sparseFTC = True # just do FTC multiple times.
            else:
                sparseApprox = True  # do a real sparse approximation.
        
        if sparseFTC:

            pc.param[0, -1] = pc.lnsigma2 
            p.param[0, -1] = p.lnsigma2 
            parts = int(round(float(len(filmRatings))/o.maxFTC + 0.5))
            splitPoint = len(filmRatings)/parts
            startPoint = 0
            for i in range(parts-1):
                if i == parts-1:
                    endPoint = -1
                else:
                    endPoint = startPoint+splitPoint
                Xiter, yiter = \
                convertNlMatrix(filmRatings = \
                                filmRatings[startPoint:endPoint].flatten(),
                                filmIndices = \
                                filmIndices[startPoint:endPoint].flatten(),
                                dataSet = d,
                                parameters = p)

                model = ndlml.gp(latentDim, 1, Xiter, yiter, p.fullKern, p.noise, ndlml.gp.FTC, 0, 3)
                paramIter = ndlml.matrix(p.param.shape[0], p.param.shape[1])
                for i in range(p.param.shape[1]):
                    paramIter.setVal(p.param[0, i], 0, i)
                model.setOptParams(paramIter)
                model.setOptimiseX(True)    

                # Do additional parameter changes
                p, pc = updateParam(filmIndices[startPoint:endPoint].flatten(),
                                    model, p, pc, sparseApprox, 
                                    learnRate, options)

                startPoint = endPoint
                
                
                
            endPoint = -1 
            filmRatings = filmRatings[startPoint:endPoint].flatten()
            filmIndices = filmIndices[startPoint:endPoint].flatten()
            Xiter, yiter = \
            convertNlMatrix(filmRatings = \
                            filmRatings,
                            filmIndices = \
                            filmIndices,
                            dataSet = d,
                            parameters = p)
            sparseApprox = False
        else:
            # Remove item means and divide by item standard deviations.
            Xiter, yiter = convertNlMatrix(filmRatings = filmRatings,
                                           filmIndices = filmIndices,
                                           dataSet = d,
                                           parameters = p)
                

        paramIter = ndlml.matrix()



        if sparseApprox:
            p.param[0, -1] = p.lnbeta
            pc.param[0, -1] = pc.lnbeta 

                

            # Set up GPLVM with sparse approximation.
            model = ndlml.gp(latentDim, 1, Xiter, yiter, p.sparseKern, p.noise, o.sparseApprox, o.numActive, 3)
            paramIter.resize(1, o.numActive*latentDim+p.fullKern.getNumParams())

            # Set the inducing point locations.
            counter2 = 0
            for j in range(latentDim):
                for i in range(o.numActive):
                    paramIter.setVal(p.X_u[i, j], counter2)
                    counter2 += 1

            # Set the parameters.
            for i in range(p.param.shape[1]):
                paramIter.setVal(p.param[0, i], counter2)
                counter2 += 1


            model.setOptParams(paramIter)
            model.setOptimiseX(True)

        else:
            # Set up full GPLVM.
            pc.param[0, -1] = pc.lnsigma2 
            p.param[0, -1] = p.lnsigma2 

            model = ndlml.gp(latentDim, 1, Xiter, yiter, p.fullKern, p.noise, ndlml.gp.FTC, 0, 3)
            paramIter = ndlml.matrix(p.param.shape[0], p.param.shape[1])
            for i in range(p.param.shape[1]):
                paramIter.setVal(p.param[0, i], 0, i)
            model.setOptParams(paramIter)
            model.setOptimiseX(True)    
        

        p, pc = updateParam(filmIndices, model, p, pc, sparseApprox, learnRate, options)

        # extract beta/sigma2 from the model.
        if sparseApprox:
            p.lnbeta = p.param[0, -1]
            pc.lnbeta = pc.param[0, -1]
        else:
            p.lnsigma2 = p.param[0, -1]
            pc.lnsigma2 = pc.param[0, -1]

        # Finished one iteration
        it.count = it.count + 1
        # Check if it's time to display
        if not np.remainder(it.count, o.showEvery):
            print "Count " + str(it.count)
            toc = time.time()
            eTime = toc - it.tic;
            totTime = toc - it.tic0;
            usersPerSecond = (it.count-startCount)/totTime
            remainUserIters = numUsers*o.numIters - it.count
            remainTime = remainUserIters/usersPerSecond
            it.tic = toc
            print("Remain time (hrs): " + str(remainTime/3600))
            sys.stdout.flush()

        # Check if it's time to save
        if not np.remainder(it.count, o.saveEvery):
            print("Saving file ...")
            toc = time.time()
            eTime = toc - it.tic;
            totTime = toc - it.tic0;
            usersPerSecond = it.count/totTime
            remainUserIters = numUsers*o.numIters - it.count
            remainTime = remainUserIters/usersPerSecond
            it.tic = toc
            print("Remain time (hrs): " + str(remainTime/3600))
            dirName = "count" + str(it.count) + "_user" + str(user)
            saveDir = os.path.join(iterDir, dirName)
            if not os.path.exists(saveDir):
                os.mkdir(saveDir)
            betaSigma = np.array([p.lnbeta, p.lnsigma2, pc.lnbeta, pc.lnsigma2])
            betaSigma.tofile(os.path.join(saveDir, "betaSigma"))
            p.param.tofile(os.path.join(saveDir, "param" ))
            model.toFile(os.path.join(saveDir, "gp" ))
            p.X.tofile(os.path.join(saveDir, "X" ))
            p.X_u.tofile(os.path.join(saveDir, "X_u" ))
            pc.X.tofile(os.path.join(saveDir, "Xchange" ))
            pc.param.tofile(os.path.join(saveDir, "paramChange" ))
            pc.X_u.tofile(os.path.join(saveDir, "inducingChange" ))
            sys.stdout.flush()


def updateParam(filmIndices, model, p, pc, sparseApprox, learnRate, o):

    latentDim = p.X.shape[1]
    giter = ndlml.matrix(1, model.getOptNumParams())
    model.computeObjectiveGradParams(giter)
    pdb.set_trace()
    #g = np.zeros((giter.getRows(), giter.getCols()))
    g = giter.toarray().copy()

    ####### Get X Gradients and update #######################
    endPoint = model.getNumData()*latentDim
    # Add "weight decay" term to gradient.
    gX = g[0, 0:endPoint].reshape((latentDim, model.getNumData())).transpose() 

    # find the last changes associated with these indices.
    XTempChange = pc.X[filmIndices, :]

    # momentum times the last change plus gradient times learning
    # rate is new change.
    #adjustRates = d.learnRateAdjust[filmIndices]*learnRate
    ##adjustRates[np.nonzero(adjustRates>o.maxLearnRate)] = o.maxLearnRate
    adjustRates = learnRate*10.0
    XTempChange = XTempChange*o.momentum + gX*adjustRates
    # store new change
    pc.X[filmIndices, :] = XTempChange
    # update X
    p.X[filmIndices, :] = p.X[filmIndices, :] - XTempChange
    # Apply "weight decay" globally to X --- v. sparse stray data
    # points back towards the centre.
    #p.X = p.X - p.X*learnRate 

    if sparseApprox:
    ####### Get inducing variable gradients and update #######
        startPoint = endPoint 
        endPoint = startPoint + o.numActive*latentDim
        gX_u = g[0, startPoint:endPoint].reshape((model.getInputDim(), o.numActive)).transpose()
        
        # update inducingChange
        pc.X_u = pc.X_u*o.momentum + gX_u*learnRate
        
        # update X_u
        p.X_u = p.X_u - pc.X_u

    ####### Get parameter gradients and update ###################
    startPoint = endPoint
    endPoint = startPoint + p.fullKern.getNumParams()

    # update paramChange
    pc.param = pc.param*o.momentum + g[0, startPoint:endPoint]*learnRate
    # update parameters
    p.param = p.param - pc.param
    return p, pc


def predVal(user, testFilmId, parameters, dataSet):
    """Make a prediction for the given user and the given film ID."""

    latentDim = parameters.X.shape[1]
    useForPred = 500
    u = dataSet.data.user(user)
    filmRatings = u.ratings()
    filmIndices = u.values()-1
    d2 = netlab.dist2(parameters.X[filmIndices, :], parameters.X[testFilmId, :].reshape((1, latentDim)))
    if useForPred < len(filmIndices):
        perm = np.argsort(d2, axis=0)
        filmRatings = filmRatings[perm[0:useForPred]].flatten()
        filmIndices = filmIndices[perm[0:useForPred]].flatten()

    xstar = ndlml.matrix(1, latentDim)
    for i in range(latentDim):
        xstar.setVal(parameters.X[testFilmId, i], i)
    
    X, y = convertNlMatrix(filmRatings=filmRatings, \
                               filmIndices=filmIndices, \
                               dataSet=dataSet, \
                               parameters=parameters)
    K = ndlml.matrix(X.getRows(), X.getRows())
    kstar = ndlml.matrix(X.getRows(), 1)
    pred = ndlml.matrix(X.getRows(), 1)

    parameters.fullKern.compute(K, X);
    parameters.fullKern.compute(kstar, X, xstar);
    K.pdinv() # now invK
    pred.gemv(K, kstar, 1.0, 0.0, "n") # K
    mean = pred.dotColCol(0, y, 0)
    diag = parameters.fullKern.diagComputeElement(xstar, 0)
    var = diag - pred.dotColCol(0, kstar, 0)
    return mean, var


def preprocessRatings(filmRatings, filmIndices, dataSet):
    """Run the preprocessing of the film ratings using the mean for
    the item and its standard deviation"""

    y = np.reshape((filmRatings.flatten()-dataSet.itemMean[filmIndices].flatten()) \
                       /dataSet.itemStd[filmIndices].flatten(), \
                       (len(filmIndices), 1))

    return y

def convertNlMatrix(filmRatings, filmIndices, dataSet, parameters):
    """Take in the filmRatings vector and the film indices data along with the data set and the parameters. Return ndlml.matrix for X and y"""
    
    latentDim = parameters.X.shape[1]
    y = preprocessRatings(filmRatings=filmRatings, \
                              filmIndices=filmIndices, \
                              dataSet=dataSet)
    
    # Xiter, yiter, paramIter are the things to be passed to the
    # ndlml.gp model for finding the gradient. They must be in the
    # form of ndlml.matrix().
    Xiter = ndlml.matrix(len(filmIndices), latentDim)
    yiter = ndlml.matrix(len(filmIndices), 1)
    count3 = 0
    for i in filmIndices:
        yiter.setVal(y[count3, 0], count3, 0)
        for j in range(latentDim):
            Xiter.setVal(parameters.X[i, j], count3, j)
        count3=count3 + 1
    return Xiter, yiter


def predictProbe(latentDim, dataSetName, experimentNo, loadIter, loadUser, loadCount, options):

    resultsDir = os.path.join(options.resultsBaseDir, "netflix" + str(experimentNo))

    loadDir1 = "iter" + str(loadIter)
    loadDir2 = "count" + str(loadCount) + "_user" + str(loadUser)

    loadDir = os.path.join(resultsDir, loadDir1, loadDir2)

    p, pc = loadResults(loadDir, latentDim, options)

    d = loadData(dataSetName)

    print "Loading netflix probe data ..."
    probe = pyflix.datasets.RatedDataset(os.path.join(dataDir(),'probe_set'))
    
    print "Testing exp no ", experimentNo, "iter ", loadIter, " count number ", loadCount
    total = 0
    totalSe = 0.0
    userIds = probe.userIDs()
    tic0 = time.time()
    tic = tic0
    numUsers = len(userIds)
    for user in np.sort(userIds):
        up = probe.user(user)

        filmRatings = up.ratings()
        filmIndices = up.values()-1
        
        u = d.data.user(user)
        testLen = len(u.ratings())
        #if testLen > 10:
        #    continue
        count =0
        for film in filmIndices:
            total += 1

            (pred, var) = predVal(user, film, p, d)
            pred = pred*d.itemStd[film] + d.itemMean[film]
            #if pred>5:
            #    pred = 5
            #if pred<1:
            #    pred = 1
            newSe = (pred - filmRatings[count])**2
            totalSe += newSe
            rmse = math.sqrt(totalSe/float(total))
            if not np.remainder(total, options.showEvery):
                toc = time.time()
                eTime = toc - tic;
                totTime = toc - tic0;
                usersPerSecond = total/totTime
                remainUserIters = numUsers - total
                remainTime = remainUserIters/usersPerSecond
                tic = toc
                print("Remain time (hrs): " + str(remainTime/3600))          
                print "Total Count: ", total, " rmse: ", rmse #, "pred ", pred, "True: ", filmRatings[count], "var: ", var
            count += 1
            
