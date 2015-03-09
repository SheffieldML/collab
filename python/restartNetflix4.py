#!/usr/bin/env python



# Try collaborative filtering on the netflix data.
import collab
import ndlml as nl
opt = collab.options()
opt.resultsBaseDir = "/local/data/results/netflix/"
try: 
    collab.restart(loadIter = 5, 
                   startCount = 2520000, 
                   loadUser = 1776294, 
                   latentDim = 4, 
                   dataSetName = 'netflix', 
                   experimentNo = 4, 
                   options=opt) 
except:
    import pdb, sys
    e, m, tb = sys.exc_info()
    pdb.post_mortem(tb)

