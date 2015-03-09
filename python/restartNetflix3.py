#!/usr/bin/env python



# Try collaborative filtering on the netflix data.
import collab
import ndlml as nl
opt = collab.options()
opt.resultsBaseDir = "/local/data/results/netflix/"
try: 
    collab.restart(loadIter = 5, 
                   startCount = 2660000, 
                   loadUser = 1499180, 
                   latentDim = 3, 
                   dataSetName = 'netflix', 
                   experimentNo = 3, 
                   options=opt) 
except:
    import pdb, sys
    e, m, tb = sys.exc_info()
    pdb.post_mortem(tb)

