#!/usr/bin/env python2.6

from socket import *
import numpy as n
from optparse import OptionParser
import sys
import time
#from pylab import *

# Set the socket parameters
# To modify !!
host = "192.168.1.1"
buffer = 9000

first = 1

full_usage = """
TBD
"""

usage = "usage: %prog [options]"

parser = OptionParser(usage)

parser.add_option("-n", "--nchan", type="int", dest="nchan", default=2, help="Number of channels per 10Gb output [default=2]")
parser.add_option("-s", "--schan", type="int", dest="schan", default=0, help="Select the channel for output [default=0]")
parser.add_option("-v", "--verbose", type="int", dest="verbose", default=0, help="Select verbosity 0->2 [default=0]")
parser.add_option("-p", "--port", type="int", dest="port", default=6000, help="Select port number [default=6000]")
parser.add_option("-r", "--record", action="store_true", dest="record", default=False, help="Display binned total intensity (by factor 8192/nchan*4) to standard output until ^C. Default is to write complete polar for the specific channel for a single packet and exit")
parser.add_option("-P", "--plot", action="store_true", dest="plot", default=False, help="Plot the histogram of the data received by the 10 GbE port")


(opts, args) = parser.parse_args()

verbose = opts.verbose
nchan = opts.nchan
schan = opts.schan
record = opts.record
port = opts.port

addr = (host,port)
# Create socket and bind to address
Sock = socket(AF_INET,SOCK_DGRAM)
Sock.bind(addr)

bigbuf=n.array([])
iter = 0
counter = 50000
#ion()

# Main loop 
while iter<5: 
    data,addr = Sock.recvfrom(buffer)
    if not data:
        print "Oups.. Problem"
	break
    else:
        # Read header
        counter, ip_id = n.fromstring(data[0:16], dtype=n.int64)
	counter = counter.byteswap()
	ip_id = ip_id.byteswap()
	print counter, ip_id, len(data)

        buf = n.fromstring(data[16:28], dtype=n.int8)
	print buf

	print buf[0], buf[1], buf[2], buf[3]
	print buf[0+4], buf[1+4], buf[2+4], buf[3+4]
	print buf[0+8], buf[1+8], buf[2+8], buf[3+8]
	#data[14], data[15], data[16], data[17], data[18]

	sys.exit()

	# Read data in 8 bytes lines
        buf = n.fromstring(data[16:8208], dtype=n.int64)

	# Byteswap each 8 bytes line in-place
	buf.byteswap(True)

	if(verbose>1):
	    print buf
	    print "buf : ", len(buf), type(buf)
	    print 

	# Split 8 bytes lines into 2 * 4 bytes samples 
	buf2 = n.fromstring(buf[0:8192], dtype=n.int8)
	if(verbose>1):
	    print buf2
	    print "buf2 : ", len(buf2), type(buf2)
	    print

	# Transform into a matrix - Lines are X0_r X0_i Y0_r Y0_i  X1_r X1_i Y1_r Y1_i
	matrix = n.reshape(buf2,(nchan*4,8192/(nchan*4)), order='F')
	if(verbose):
	    print matrix
	    print "matrix : ", len(matrix), type(matrix)
	    print

	# Print time serie
	if record:
	    # calc Total intensity for the 'schan' selected channel 
	    I = n.power(matrix[4*schan,:],2) + n.power(matrix[4*schan+1,:],2) + n.power(matrix[4*schan+2,:],2) + n.power(matrix[4*schan+3,:],2)

	    if(first):
	        print "# %s "%time.asctime()
	        fpga_ref = counter
		first = 0

	    print "%d %f"%(((counter-fpga_ref)/4096),sum(I)/(8192./(nchan*4.)))		

	elif opts.plot:
	    bigbuf = n.append(bigbuf, buf2)
	    #print len(bigbuf)
	    if iter < 20:
	        iter += 1
	    else:	
	        a1=subplot(111)
	        a1.clear()
	        hist(bigbuf,bins=100, range=(-150, 150), normed=1)
		draw()
		bigbuf=n.array([])
		iter = 0
		#break

	# Print one packet
	else:    
	    print "fpga counter = %d, ip_id = %d"%(counter, ip_id)
	    print "Channel #%d of %d channels"%(schan,nchan)
	    #print "X_r  X_i  Y_r  Y_i"
	    """
	    for i in range(8192/(nchan*4)):
	        print "%3d  %3d  %3d  %3d"%(matrix[4*schan,i], matrix[4*schan+1,i], matrix[4*schan+2,i], matrix[4*schan+3,i]) 
	    """	
		
	    #break	
	    if counter < 500000:
	        print counter
	        iter += 1
	



# Close socket
Sock.close()
