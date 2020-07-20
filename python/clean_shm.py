#!/usr/bin/env python

from nuppi_utils import *
from optparse import OptionParser
import shm_wrapper as shmw




# Set Logging
log = logfile("clean_shm.py")

log.info('Call to clean_shm.py')

usage = "usage: %prog [options]"

par = OptionParser(usage)

par.add_option("-S", "--status", dest="delstatus", help="Delete Status shared memory",
	action="store_true", default=False)

(opt,arg) = par.parse_args()

# If we want to delete the status shm
if opt.delstatus:

    for i in [1, 2]:
	try:
	    b = Status(log, i)
	    b.delete()
	except:
	    log.info("Shm Status %d can't be found or can't be removed"%i)

# Delete the databug shared memory
for i in [1, 2, 5, 6]:
    try:
        b = Databuf(i)
        b.delete()
    except:
        log.info("Shm Databuf %d can't be found or can't be removed"%i)
