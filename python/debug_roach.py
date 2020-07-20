#!/usr/bin/env python2.6

#
#  Adapted from 'pasp_init.py' by T.Filiba (UC Berkeley)
#  http://casper.berkeley.edu/svn/trunk/projects/pasp/server_software/configure_roach/pasp_init.py
#

import corr, time, struct, sys, logging, socket, re
from numpy import histogram, mean, std
import nrt_scaling_factor




# fft configuration
# shifting schedule
#fft_shift=0xffffffff
fft_shift=0xfffffffe


# bit select configuration from 8 bits from scaled 18 bit fft data
# 0-select bottom 8 bits
# 1-
# 2-
# 3-select top 8 bits
bitselect_pol0=1  # select bottom bits
bitselect_pol1=1  # select bottom bits

# 8 IPs
ip_table=['192.168.1.1','192.168.1.2','192.168.1.3','192.168.1.4','192.168.1.1','192.168.1.2','192.168.1.3','192.168.1.4']
port_table=[6000,6000,6000,6000,6001,6001,6001,6001]

gbe_base='pasp_dist_gbe_ten_GbE'
ip_reg_base='pasp_dist_gbe_ip_ctr_xblock_init_reg_ip'
port_reg_base='pasp_dist_gbe_ip_ctr_xblock_init_reg_port'

mac_base=(2<<40) + (2<<32)

fabric_ip=struct.unpack('!L',socket.inet_aton('192.168.1.10'))[0] #convert ip to long
fabric_port=6000

roach='roach'
katcp_port=7147

class ROACH():    

    def __init__(self):
        try:
            self.fpga = corr.katcp_wrapper.FpgaClient(roach, katcp_port, timeout=10)
            #self.fpga = corr.katcp_wrapper.FpgaClient(roach, katcp_port, timeout=10,logger=self.log)
            time.sleep(1)


        except KeyboardInterrupt:
            self.exit_clean()
        except:
            self.exit_fail()

    def load_design(self, boffile):
	    self.boffile = boffile
            # program the fpga with selected boffile
            self.fpga.progdev('')
            time.sleep(1)
            self.fpga.progdev(self.boffile);
            time.sleep(10)
    
            self.configure_parameters()
            #dump_fft_brams()

            self.init_10gbe_blocks()
            self.init_ip_table()

	    # Wait to finish programming
	    time.sleep(10)

    def rearm(self):	    
            self.fpga.write_int('pasp_reg_arm',0)
            self.fpga.write_int('pasp_reg_arm',1)

    def exit_fail(self):
        try:
            self.fpga.stop()
        except: pass
        raise

    def exit_clean(self):
        try:
            self.fpga.stop()
        except: pass
        #exit()
    
   
    def configure_parameters(self):

        # extract build configuration info from boffile name
        m=re.search('pasp_([\d]+)i([\d]+)c([\d]+)s([\d]+)g',self.boffile)
    
        # get pasp build configuration from boffile name
        self.numips=int(m.group(1))
        self.numchannels=int(m.group(2))
        self.packetsize=int(m.group(3))
        self.numtengbe=int(m.group(4))
        self.lcm_reorder_order=3
	self.ntaps=8

	if self.numchannels==128:
	    self.lcm_reorder_order=30
	elif self.numchannels==1024:
	    self.lcm_reorder_order=126
    
        # calculate the sync period
	# N Taps * LCM(reorder orders)*(FFTSize/simultaneousinputs)*numips*packetsize
        sync_period=self.ntaps*self.lcm_reorder_order*self.numchannels/4*self.numips*self.packetsize


        # initialize the fpga sync
        self.fpga.write_int('pasp_reg_sync_period',sync_period)
    
        # initialize the fft shift schedule
        self.fpga.write_int('pasp_reg_fft_shift',fft_shift)
    
        # initialize the scaling parameters
	# fft scaling configuration
	# interpreted as 18.12 ufix
	"""
	if self.numchannels == 1024:
	    fft_polX_coeffs=nrt_scaling_factor.xscale_factor_1024
	    fft_polY_coeffs=nrt_scaling_factor.yscale_factor_1024
	    ## GD, finally, remove the scaling...
	    fft_polX_coeffs=[4096]*2048    # no scaling (multiply by 1)
	    fft_polY_coeffs=[4096]*2048    # no scaling (multiply by 1)
	    fft_polX_coeffs_string = struct.pack('>1024L',*fft_polX_coeffs)
	    fft_polY_coeffs_string = struct.pack('>1024L',*fft_polY_coeffs)
	else:
	"""
	fft_polX_coeffs=[4096]*2048    # no scaling (multiply by 1)
	fft_polY_coeffs=[4096]*2048    # no scaling (multiply by 1)
	fft_polX_coeffs_string = struct.pack('>2048L',*fft_polX_coeffs)
	fft_polY_coeffs_string = struct.pack('>2048L',*fft_polY_coeffs)

        # initialize the scaling bit selection
        self.fpga.write_int('pasp_rescale_pol0_reg_output_bitselect',bitselect_pol0)
        self.fpga.write_int('pasp_rescale_pol1_reg_output_bitselect',bitselect_pol1)

      
    def init_10gbe_blocks(self):
        # initialize the 10gbe ports
        for i in range(0,self.numtengbe):
	    self.fpga.tap_start('gbe'+str(i),gbe_base+str(i),mac_base+fabric_ip+i,fabric_ip+i,fabric_port)
    
    def init_ip_table(self):
        # initialize the ip table
        for i in range(0,self.numips):
            self.fpga.write_int(ip_reg_base+str(i),struct.unpack('!L',socket.inet_aton(ip_table[i]))[0])
            self.fpga.write_int(port_reg_base+str(i),port_table[i])

        
    def clear_ip_table(self):
        # zero the ip table
        for i in range(0,self.numips):
            self.fpga.write_int(ip_reg_base+str(i),0)

    # functions for dumping data    
    def dump_adc_brams(self):
        totadc1=[]
        totadc2=[]

        for i in range(10):

	    #trigger the adcscope
	    self.fpga.write_int('pasp_reg_adcscope_trigger',1)
	    self.fpga.write_int('pasp_reg_adcscope_trigger',0)

	    #read the bram data
	    adcscope1 = struct.unpack('>2048b',self.fpga.read('pasp_bram_adcscope_1',2048))
	    adcscope2 = struct.unpack('>2048b',self.fpga.read('pasp_bram_adcscope_2',2048))

	    totadc1.append(adcscope1)
	    totadc2.append(adcscope2)

	print mean(totadc1), std(totadc1)
	print mean(totadc2), std(totadc2)
    #print histogram(adcscope1, range=(-128,127), bins=257)
