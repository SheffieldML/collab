#!/usr/bin/env python



# Try collaborative filtering on the netflix data.
import collab
import ndlml as nl
opt = collab.options()
opt.resultsBaseDir = "/local/data/results/netflix/"
try: 
    collab.restart(loadIter = 6, 
                   startCount = 3320000, 
                   loadUser = 2600176,
                   latentDim = 6, 
                   dataSetName = 'netflix', 
                   experimentNo = 6, 
                   options=opt) 
except:
    import pdb, sys
    e, m, tb = sys.exc_info()
    pdb.post_mortem(tb)

