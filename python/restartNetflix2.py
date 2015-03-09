#!/usr/bin/env python



# Try collaborative filtering on the netflix data.
import collab
import ndlml as nl
opt = collab.options()
opt.resultsBaseDir = "/local/data/results/netflix/"
try:
    collab.restart(loadIter = 9, 
                   startCount = 4600000, 
                   loadUser = 1288699, 
                   latentDim = 2, 
                   dataSetName = 'netflix', 
                   experimentNo = 2, 
                   options=opt) 
except:
    import pdb, sys
    e, m, tb = sys.exc_info()
    pdb.post_mortem(tb)

