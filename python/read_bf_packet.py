#!/usr/bin/env python2.6

from socket import *
import numpy as n
from optparse import OptionParser
import sys
import time
#from pylab import *

# Set the socket parameters
# To modify !!
host = "10.20.0.2"
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
parser.add_option("-p", "--port", type="int", dest="port", default=50000, help="Select port number [default=6000]")
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
packet = 0
#ion()

# Main loop 
while packet < 2:
    data,addr = Sock.recvfrom(buffer)
    if not data:
        print "Oups.. Problem"
	break
    else:
        # Read header
        packet_group, packet_version, bits, binary_point_position = n.fromstring(data[0:4], dtype=n.uint8)
	print "packet_group = %d  packet_version = %d  bits = %d  binary_point_position = %d"%(packet_group, packet_version, bits, binary_point_position)

        endian = n.fromstring(data[4:8], dtype=n.uint32)
	print "endian = %d"%endian

        packet_type, n_streams, polar_code, header_length = n.fromstring(data[8:12], dtype=n.uint8)
	print "packet_type = %d  n_streams = %d  polar_code = %d  header_length = %d"%(packet_type, n_streams, polar_code, header_length)

        data_source, channel_number, sequence_number = n.fromstring(data[12:24], dtype=n.uint32)
	print "data_source = %d  channel_number = %d  sequence_number = %d"%(data_source, channel_number, sequence_number)

        frequency, sampling_rate = n.fromstring(data[24:40], dtype=n.float64)
	print "frequency = %lf  sampling_rate = %lf"%(frequency, sampling_rate)

        frac_bandwidth, non_used = n.fromstring(data[40:48], dtype=n.float32)
	print "frac_bandwidth = %f, 4 bytes non_used"%(frac_bandwidth)

        time_int, time_frac, status_flag, data_length = n.fromstring(data[48:64], dtype=n.uint32)
	print "time_int = %d  time_frac = %d  status_flag = %d  data_length = %d"%(time_int, time_frac, status_flag, data_length)




	# Read data in 8 bytes lines
        buf = n.fromstring(data[64:4160], dtype=n.int32)

	"""
	    print "fpga counter = %d, ip_id = %d"%(counter, ip_id)
	    print "Channel #%d of %d channels"%(schan,nchan)
	    print "X_r  X_i  Y_r  Y_i"
	    for i in range(8192/(nchan*4)):
	        print "%3d  %3d  %3d  %3d"%(matrix[4*schan,i], matrix[4*schan+1,i], matrix[4*schan+2,i], matrix[4*schan+3,i]) 
		
	    #break	
	"""   
	
    packet +=1



# Close socket
Sock.close()
