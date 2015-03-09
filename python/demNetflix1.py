#!/usr/bin/env python


# Try collaborative filtering on the netflix data.

import pdb
import os
import sys
import time
import posix
sys.path.append(os.path.join(posix.environ['HOME'], 'mlprojects', 'collab', 'python'))
sys.path.append(os.path.join(posix.environ['HOME'], 'mlprojects', 'swig', 'src'))
import pyflix.datasets
import numpy as np
import ndlml as nl
import math

np.random.seed(seed=10000)

q = 2
lambdaVal = 0.01
maxLearnRate = 1
t0 = 400000

momentum = 0.9
startVar = 5

numActive = 100
maxFTC = 500

showEvery = 5000
saveEvery = 20000

numIters = 10

dataSetName = "netflix"
experimentNo = 1


# load in data.
baseDir = os.path.join('/local', 'data', 'pyflix')
tr = pyflix.datasets.RatedDataset(os.path.join(baseDir, 'training_set'))


resultsDir = os.path.join(".", dataSetName + str(experimentNo))
if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)


# set initial values to small random numbers.
X = np.random.normal(0.0, 1e-6, (len(tr.movieIDs()), q))
X_u = np.random.normal(0.0, 1e-6, (numActive, q))

lnsigma2 = math.log(startVar)
lnbeta = -math.log(startVar)


# Set up kernel functions
kernFtc = nl.cmpndKern(q)
kernSp = nl.cmpndKern(q)

kern1 = nl.rbfKern(q)
kern2 = nl.biasKern(q)
kern3 = nl.whiteKern(q)
kern4 = nl.whitefixedKern(q)

kern2.setVariance(0.11)
kern3.setVariance(math.exp(lnsigma2))
kern4.setVariance(1e-2)

kernFtc.addKern(kern1)
kernFtc.addKern(kern2)
kernFtc.addKern(kern3)

kernSp.addKern(kern1)
kernSp.addKern(kern2)
kernSp.addKern(kern4)

# using this dummy y forces mean of Gaussian noise to be zero.
dummyy = nl.matrix(10, 1)
dummyy.zeros()
noise = nl.gaussianNoise(dummyy)

movieIDs = tr.movieIDs()
numMovies = len(movieIDs)

userIds = tr.userIDs()
numUsers = len(userIds)

# Compute movie means and standard deviations.
movieMean = np.zeros((numMovies, 1))
movieStd = np.zeros((numMovies, 1))
movieCount = np.zeros((numMovies, 1))
learnRateAdjust = np.zeros((numMovies, 1))

for movie in movieIDs:
    m = tr.movie(movie)
    movieMean[movie-1] = m.ratings().mean()
    movieStd[movie-1] = m.ratings().std()
    movieCount[movie-1] = len(m.ratings())
    learnRateAdjust[movie-1] = float(numUsers)/float(len(m.ratings()))

            
# Set up parameter vectors
Xchange = np.zeros((numMovies, q))
paramNd = nl.matrix(1, kernFtc.getNumParams())
kernFtc.getTransParams(paramNd)
param = paramNd.toarray()
paramIter = nl.matrix()



# Set up vectors for storing old changes.
paramChange = np.zeros((1, kernFtc.getNumParams()))
inducingChange = np.zeros((numActive, q))
lnsigma2Change = 0
lnbetaChange = 0


print "Starting ..."
tic = time.time()
tic0 = tic
count = 0

for iter in range(numIters):
    # Ensure repeatability
    state = np.random.get_state()

    # Order users randomly
    userOrder = np.random.permutation(userIds)

    # Save state for repeatability
    saveDir = "iter" + str(iter)
    iterDir = os.path.join(resultsDir, saveDir)
    if not os.path.exists(iterDir):
        os.mkdir(iterDir)
    userOrder.tofile(os.path.join(iterDir, "userOrder" ))
    param.tofile(os.path.join(iterDir, "param" ))
    X.tofile(os.path.join(iterDir, "X" ))
    X_u.tofile(os.path.join(iterDir, "X_u" ))
    Xchange.tofile(os.path.join(iterDir, "Xchange" ))
    paramChange.tofile(os.path.join(iterDir, "paramChange" ))
    inducingChange.tofile(os.path.join(iterDir, "inducingChange" ))

    for user in userOrder:
        learnRate = 1.0/(lambdaVal*(count + t0))
        u = tr.user(user)
        filmRatings = u.ratings()
        filmIndices = u.values()-1

        if len(filmRatings)>maxFTC:
            sparseApprox = True
        else:
            sparseApprox = False
        # Create ndlmatrixmatrix from user preferences.
        Xiter = nl.matrix()
        Xiter.fromarray(X[filmIndices, :])
        # Remove movie means and divide by movie standard deviations.
        y = np.reshape((filmRatings-movieMean[filmIndices].flatten())/movieStd[filmIndices].flatten(), (len(filmIndices), 1))
        yiter = nl.matrix()
        yiter.fromarray(y)
        count = count + 1
            
        
        if sparseApprox:
            param[0, -1] = lnbeta
            paramChange[0, -1] = lnbetaChange 
            model = nl.gp(q, 1, Xiter, yiter, kernSp, noise, nl.gp.DTCVAR, numActive, 3)
            paramIter.resize(1, numActive*q+kernFtc.getNumParams())

            # Set the inducing point locations.
            counter2 = 0
            for j in range(q):
                for i in range(numActive):
                    paramIter.setVal(X_u[i, j], counter2)
                    counter2 += 1

            # Set the parameters.
            for i in range(param.shape[1]):
                paramIter.setVal(param[0, i], counter2)
                counter2 += 1
            
            
            model.setOptParams(paramIter)
            model.setOptimiseX(True)
            
        else:
            paramChange[0, -1] = lnsigma2Change 
            param[0, -1] = lnsigma2 
           
            model = nl.gp(q, 1, Xiter, yiter, kernFtc, noise, nl.gp.FTC, 0, 3)
            paramIter.fromarray(param)
            model.setOptParams(paramIter)
            model.setOptimiseX(True)    

        giter = nl.matrix(1, model.getOptNumParams())
        model.computeObjectiveGradParams(giter)

        g = giter.toarray()

        ####### Get X Gradients and update #######################
        endPoint = model.getNumData()*q
        # Add "weight decay" term to gradient.
        gX = g[0, 0:endPoint].reshape((q, model.getNumData())).transpose() 

        # find the last changes associated with these indices.
        XTempChange = Xchange[filmIndices, :]

        # momentum times the last change plus gradient times learning
        # rate is new change.
        adjustRates = learnRateAdjust[filmIndices]*learnRate
        adjustRates[np.nonzero(adjustRates>maxLearnRate)] = maxLearnRate
        XTempChange = XTempChange*momentum + gX*adjustRates
        # store new change
        Xchange[filmIndices, :] = XTempChange
        # update X
        X[filmIndices, :] = X[filmIndices, :] - XTempChange
        # Apply "weight decay" globally.
        X = X - X*learnRate 
        if sparseApprox:
        ####### Get inducing variagle gradients and update #######
            startPoint = endPoint 
            endPoint = startPoint + numActive*q
            gX_u = g[0, startPoint:endPoint].reshape((model.getInputDim(), numActive)).transpose()

            # update inducingChange
            inducingChange = inducingChange*momentum + gX_u*learnRate

            # update X_u
            X_u = X_u - inducingChange
        
        ####### Get param gradients and update ###################
        startPoint = endPoint
        endPoint = startPoint + kernFtc.getNumParams()

        # update paramChange
        paramChange = paramChange*momentum + g[0, startPoint:endPoint]*learnRate

        # update parameters
        param = param - paramChange

        if sparseApprox:
            lnbeta = param[0, -1]
            lnbetaChange = paramChange[0, -1]
        else:
            lnsigma2 = param[0, -1]
            lnsigma2Change = paramChange[0, -1]


        # Check if it's time to display
        if not np.remainder(count, showEvery):
            print "Count " + str(count)
            toc = time.time()
            eTime = toc - tic;
            totTime = toc - tic0;
            usersPerSecond = count/totTime
            remainUserIters = numUsers*numIters - count
            remainTime = remainUserIters/usersPerSecond
            tic = toc
            print("Remain time (hrs): " + str(remainTime/3600))

        # Check if it's time to save
        if not np.remainder(count, saveEvery):
            print("Saving file ...")
            toc = time.time()
            eTime = toc - tic;
            totTime = toc - tic0;
            usersPerSecond = count/totTime
            remainUserIters = numUsers*numIters - count
            remainTime = remainUserIters/usersPerSecond
            tic = toc
            print("Remain time (hrs): " + str(remainTime/3600))
            dirName = "count" + str(count) + "_user" + str(user)
            saveDir = os.path.join(iterDir, dirName)
            if not os.path.exists(saveDir):
                os.mkdir(saveDir)
            betaSigma = np.array([lnbeta, lnsigma2, lnbetaChange, lnsigma2Change])
            betaSigma.tofile(os.path.join(saveDir, "betaSigma"))
            param.tofile(os.path.join(saveDir, "param" ))
            model.toFile(os.path.join(saveDir, "gp" ))
            X.tofile(os.path.join(saveDir, "X" ))
            X_u.tofile(os.path.join(iterDir, "X_u" ))
            Xchange.tofile(os.path.join(saveDir, "Xchange" ))
            paramChange.tofile(os.path.join(saveDir, "paramChange" ))
            inducingChange.tofile(os.path.join(iterDir, "inducingChange" ))



# Save state for repeatability
saveDir = "final"
iterDir = os.path.join(resultsDir, saveDir)
if not os.path.exists(iterDir):
    os.mkdir(iterDir)
betaSigma.tofile(os.path.join(iterDir, "betaSigma"))
param.tofile(os.path.join(iterDir, "param" ))
X.tofile(os.path.join(iterDir, "X" ))
X_u.tofile(os.path.join(iterDir, "X_u" ))
Xchange.tofile(os.path.join(iterDir, "Xchange" ))
paramChange.tofile(os.path.join(iterDir, "paramChange" ))
inducingChange.tofile(os.path.join(iterDir, "inducingChange" ))
