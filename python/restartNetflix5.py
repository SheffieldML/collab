#!/usr/bin/env python



# Try collaborative filtering on the netflix data.
import collab
import ndlml as nl
opt = collab.options()
opt.resultsBaseDir = "/local/data/results/netflix/"
try: 
    collab.restart(loadIter = 9, 
                   startCount = 4640000, 
                   loadUser = 1361446, 
                   latentDim = 5, 
                   dataSetName = 'netflix', 
                   experimentNo = 5, 
                   options=opt) 
except:
    import pdb, sys
    e, m, tb = sys.exc_info()
    pdb.post_mortem(tb)

