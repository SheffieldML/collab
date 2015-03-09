#!/usr/bin/env python



# Try collaborative filtering on the netflix data.
import collab
import ndlml as nl
opt = collab.options()
opt.resultsBaseDir = "/local/data/results/netflix/"
try: 
    collab.restart(loadIter = 5, 
                   startCount = 2620000, 
                   loadUser = 2190625, 
                   latentDim = 7, 
                   dataSetName = 'netflix', 
                   experimentNo = 7, 
                   options=opt) 
except:
    import pdb, sys
    e, m, tb = sys.exc_info()
    pdb.post_mortem(tb)

