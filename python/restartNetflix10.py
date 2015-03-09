#!/usr/bin/env python



# Try collaborative filtering on the netflix data.
import collab
import ndlml as nl
opt = collab.options()
opt.resultsBaseDir = "/local/data/results/netflix/"
try: 
    collab.restart(loadIter = 0, 
                   startCount = 200000, 
                   loadUser = 1786429, 
                   latentDim = 10, 
                   dataSetName = 'netflix', 
                   experimentNo = 10, 
                   options=opt) 
except:
    import pdb, sys
    e, m, tb = sys.exc_info()
    pdb.post_mortem(tb)

