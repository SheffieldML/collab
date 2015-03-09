#!/usr/bin/env python



# Try collaborative filtering on the netflix data.
import collab
import ndlml as nl
opt = collab.options()
opt.resultsBaseDir = "/local/data/results/netflix/"
try: 
    collab.restart(loadIter = 3, 
                   startCount = 1520000, 
                   loadUser = 560608, 
                   latentDim = 8, 
                   dataSetName = 'netflix', 
                   experimentNo = 8, 
                   options=opt) 
except:
    import pdb, sys
    e, m, tb = sys.exc_info()
    pdb.post_mortem(tb)

