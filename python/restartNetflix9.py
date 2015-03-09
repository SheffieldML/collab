#!/usr/bin/env python



# Try collaborative filtering on the netflix data.
import collab
import ndlml as nl
opt = collab.options()
opt.resultsBaseDir = "/local/data/results/netflix/"
try: 
    collab.restart(loadIter = 2, 
                   startCount = 1440000, 
                   loadUser = 2331578, 
                   latentDim = 9, 
                   dataSetName = 'netflix', 
                   experimentNo = 9, 
                   options=opt) 
except:
    import pdb, sys
    e, m, tb = sys.exc_info()
    pdb.post_mortem(tb)

