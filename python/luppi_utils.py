#
# Adapted from GUPPI
#

import shm
import shm_wrapper as shmw
import time , pyfits
#import astropy.io.fits as pyfits
import numpy as n
from astro_utils import *
import pyslalib
import pyslalib.slalib as s
from datetime import datetime
#import slalib as s
import logging as l

DEGTORAD = 0.017453292519943295769236907684
RADTODEG = 57.29577951308232087679815481410

LOG_FILENAME = '/data/data2/nuppi.log'
DATADIR = '/tmp'                         # Directory to put data
DATAPORT = 1490                          # To change
DATAHOST = 'roach'                       # in /etc/fstab


def logfile(process):
    l.basicConfig(level=l.INFO, format="%(asctime)s - %(name)s - %(levelname)s : %(message)s", filename=LOG_FILENAME, )
    return l.getLogger(process)

def get_dedisp_params(b, databuf_mb, small_blocksize=False):
    """
    get_dedisp_params(b, databuf_mb):
      Given the Status shm and the size of databuf shm, 
        returns the dedispersion parameters
    """
    lofreq_ghz = (b['OBSFREQ']-abs(b['OBSBW']/2.0))/1.0e3
    round_fac = b['PKTSIZE'] #7824
    overlap_samp = 8.3 * b['CHAN_BW']**2 / lofreq_ghz**3 * b['DM']
    print abs(b['OBSBW']/2.0)
    overlap_r = round_fac * ((int)(overlap_samp)/round_fac + 1)
    # Rough optimization for fftlen
    #fftlen = 16*1024
    #if overlap_r<=1024: fftlen=32*1024
    #elif overlap_r<=2048: fftlen=64*1024
    #elif overlap_r<=16*1024: fftlen=128*1024
    #elif overlap_r<=64*1024: fftlen=256*1024
    #elif overlap_r<=128*1024: fftlen=512*1024

    fftlen = 262144
    while fftlen<2*overlap_r: fftlen *= 2
    #overlap_r=fftlen/2

    # Force FFT len
    if(small_blocksize):
        fftlen = 2**18
        overlap_r = 0
    else:
        fftlen = 2**22
        overlap_r = fftlen/2
    #if fftlen < 256*1024:
    #    fftlen = 256*1024  # If we want 1M FFT, increase databuf size !
    databuf_mdata=databuf_mb-(8*b['OBSNCHAN']*fftlen/(1024*1024))
    npts_max_per_chan = databuf_mdata*1024*1024
    #npts_max_per_chan = databuf_mb*1024*1024/4/b['OBSNCHAN']
    nfftmax=int(npts_max_per_chan/fftlen/20)
    nfft = 1

    
    #overlap_r = round_fac * ((int)(overlap_samp)/round_fac + 1)


    while ( (nfft+1)*(fftlen - overlap_r) < b['TFOLD']/b['TBIN']):
    	nfft += 1
    	#nfft = (npts_max_per_chan - overlap_r)/(fftlen - overlap_r)
    if (nfftmax<nfft): nfft=nfftmax
    nfft = 1
    npts = nfft*(fftlen-overlap_r) + overlap_r
    
    blocsize = npts*b['OBSNCHAN']*4

    print 'blocsize=',blocsize,'   nfftmax=',nfftmax,'   nfft=',nfft,' --> ',nfft*(fftlen - overlap_r)*b['TBIN'],' sec'
    print 'databuf_mb=',databuf_mb,'   OBSNCHAN', b['OBSNCHAN']
    print 'npts_max_per_chan=',npts_max_per_chan,'   overlap_r=', overlap_r ,'   overlap_samp=', overlap_samp ,'   fftlen=', fftlen
    print 'Shirp GPUmem   = ',8*b['OBSNCHAN']*fftlen/(1024*1024),' MB'
    print 'Databuf GPUmem = ',16*nfft*fftlen/(1024*1024),' MB'
    print 'Overlap GPUmem = ',8*nfft*fftlen/(1024*1024),' MB'
    print 'Total GPUmem = ',20*nfft*fftlen/(1024*1024)+8*b['OBSNCHAN']*fftlen/(1024*1024),' MB'
    return fftlen, nfft, overlap_r, npts, blocsize

def dm_from_parfile(parfile):
    """
    dm_from_parfile(parfile):
        Read DM value out of a parfile and return it.
    """
    pf = open(parfile, 'r')
    for line in pf:
        fields = line.split()
        key = fields[0]
        val = fields[1]
        if key == 'DM':
            pf.close()
            return float(val)
    pf.close()
    return 0.0
    
def rm_from_parfile(parfile):
    """
    fm_from_parfile(parfile):
        Read RM value out of a parfile and return it.
    """
    print(parfile)
    pf = open(parfile, 'r')
    for line in pf:
        fields = line.split()
        key = fields[0]
        val = fields[1]
        if key == 'RM':
            pf.close()
            return float(val)
    pf.close()
    return 0.0

def header_from_string(str):
    """
    header_from_string(str):
        Convert an input string (which should be the ASCII header from
            a FITS HFU) into an instantiation of a pyfits 'Header' class.
    """
    cl = cardlist_from_string(str)
    return pyfits.Header(cl)

def card_from_string(str):
    """
    card_from_string(str):
        Return a pyfits 'Card' based on the input 'str', which should
            be an 80-character 'line' from a FITS header.
    """
    card = pyfits.Card()
    return card.fromstring(str)

def cardlist_from_string(str):
    """
    cardlist_from_string(str):
        Return a list of pyfits 'Cards' from the input string.
            'str' should be the ASCII from a FITS header.
    """
    cardlist = []
    numcards = len(str)/80
    for ii in range(numcards):
        str_part = str[ii*80:(ii+1)*80]
        if str_part.strip()=="END":
            break
        else:
            cardlist.append(card_from_string(str_part))
    return cardlist


STATUS_KEY = 32768 
STATUS_SIZE = 2880*64
STATUS_SEM_KEY = 1234567
STATUS_CARD = 80


class Status:

    def __init__(self, log, shm_id=1):

	# Connect or create shm
        try:
	    self.stat_buf = shmw.SharedMemoryHandle(STATUS_KEY + shm_id - 1)
	    log.info("Connected to status shm key: 0x%08x"%(STATUS_KEY + shm_id - 1))
	    print("Try Connected to status shm key: 0x%08x"%(STATUS_KEY + shm_id - 1))
	except KeyError:
	    self.a = shm.create_memory(STATUS_KEY + shm_id - 1, STATUS_SIZE, 0666)
	    log.info("Created new status shm key: 0x%08x"%(STATUS_KEY + shm_id - 1))
	    print("Created new status shm key: 0x%08x"%(STATUS_KEY + shm_id - 1))
	    #  Init 
	    self.chk_init()
	    self.stat_buf = shmw.SharedMemoryHandle(STATUS_KEY + shm_id - 1)
	    log.info("Connected to status shm key: 0x%08x"%(STATUS_KEY + shm_id - 1))
	    print("Connected to status shm key: 0x%08x"%(STATUS_KEY + shm_id - 1))
	

	# Semaphore for status shm
        try:
	    self.semid = shm.getsemid(STATUS_SEM_KEY + shm_id - 1)
	    self.sem = shm.semaphore(self.semid)
	except KeyError:
	    self.sem = shm.create_semaphore(STATUS_SEM_KEY + shm_id - 1)
	    self.semid = shm.getsemid(STATUS_SEM_KEY + shm_id - 1)

        self.hdr = None
	if (self.sem.val == 3):
	    log.info("WARNING sem 0x%08x is lock"%(STATUS_SEM_KEY + shm_id - 1))
	    print("WARNING sem 0x%08x is lock" % (STATUS_SEM_KEY + shm_id - 1))
	self.read()


    def delete(self):
        # First remove semaphore
	try:
	    shm.remove_semaphore(self.semid)
	except:
	    pass

	# Then remove status shm
	try:
	    self.stat_buf.remove()
	except:
	    pass


    def chk_init(self):
    	self.a.attach()
	self.a.write("END"+77*" ")

    def __getitem__(self, key):
        return self.hdr[key]

    def keys(self):
        return [k for k, v in self.hdr.items()]

    def values(self):
        return [v for k, v in self.hdr.items()]

    def items(self):
        return self.hdr.items()

    def lock(self):
	return self.sem.P()

    def unlock(self):
	return self.sem.V()

    def read(self):
        #self.unlock() #ADD louis (counter vs freez sem)
        self.lock()
        self.hdr = header_from_string(self.stat_buf.read())
        self.unlock()

    def write(self):
        self.lock()
        self.stat_buf.write(repr(self.hdr.ascard)+"END"+" "*77,0)
        self.unlock()

    def update(self, key, value, comment=None):
        self.hdr.update(key, value, comment)

    def show(self):
        for k, v in self.hdr.items():
            print "'%8s' :"%k, v
        print ""

    def update_time(self):
        # Current time
	MJD = current_MJD()
	#print "MJD : ", MJD
	MJDd = int(MJD)
	MJDf = MJD - MJDd
	MJDs = int(MJDf * 86400 + 1e-6)
	offs = (MJD - MJDd - MJDs/86400.0) * 86400.0
	self.update("STT_IMJD", MJDd)
	maintenant = datetime.now()
	date = "D%04u%02u%02uT%02u%02u"%(maintenant.year, maintenant.month, maintenant.day,maintenant.hour, maintenant.minute)
	self.update("STT_JDAY", date)
	self.update("STT_SMJD", MJDs)
	if offs < 2e-6:
	    offs = 0.0
        self.update("STT_OFFS", offs)

    def convert_pos_to_rad(self):
        h, m, s = self.hdr["RA_STR"].split(":")
	ra = hms_to_rad(int(h),int(m),float(s))
	self.update("RA", ra)

	h, m, s = self.hdr["DEC_STR"].split(":")
	dec = hms_to_rad(int(h),int(m),float(s))
	self.update("DEC",dec)

    def update_azza(self):
        """
        update_azza():
            Update the AZ and ZA based on the current time with the nuppi_status instance.
        """
        (iptr, ang, stat) = s.sla_dafin(self['RA_STR'].replace(':', ' '), 1)
        self.update("RA", ang*15.0*RADTODEG)
        (iptr, ang, stat) = s.sla_dafin(self['DEC_STR'].replace(':', ' '), 1)
        self.update("DEC", ang*RADTODEG)
        MJD = current_MJD()
        az, za = radec_to_azza(self['RA'], self['DEC'], MJD, scope='NRT')
        self.update("AZ", az)
        self.update("ZA", za)    
        

    def set_default_params(self):

        self.update("TELESCOP", "none")
        self.update("FRONTEND", "none")
        self.update("BACKEND", "NUPPI")
        self.update("PKTFMT", "none")
        self.update("PKTSIZE", 8816)
        self.update("DATAHOST", "none")
        self.update("DATAPORT", 0)
        self.update("NRCVR", 2)
        self.update("NBITS", 8)
        self.update("NPOL", 4)
	self.update("POL_TYPE", "IQUV")
	self.update("FD_POLN", "LIN")
	self.update("TRK_MODE", "TRACK")
	self.update("N_DS", 1)
	self.update("N_GPU", 1)

	self.update("SRC_NAME", "0000+00")
	self.update("BEAM_NUM", "BEAM0")
	self.update("MJDSTART", -1)
	self.update("RA_STR", "00:00:00.0")
	self.update("DEC_STR", "+00:00:00.0")
	self.update("DM", 0.0)
	self.update("RM", 0.0)
	self.update("OVERLAP", 0)
	self.update("PARFILE", "none")

	#
	self.update("ONLY_I", 0)
        self.update("DS_TIME", 1)
	self.update("DS_FREQ", 1)
	self.update("CHAN_DM", 0.0)

	#
	self.update("OBS_MODE", "SEARCH")
	self.update("TFOLD", 0.0)
	self.update("OBS_LEN", 0.0)
	self.update("SCANLEN", 0) #7200
	self.update("TOTNCHAN", 128)
	self.update("OBSBW", 100.0)
	self.update("OBSNCHAN", 100.0)
	self.update("LOWCHAN", 0)
	self.update("HIGHCHAN", 0.0)
	self.update("OBSERVER", "none")
	self.update("PROJID", "ES03")
	#self.update("SCANNUM", 0)

	# Calibration
	self.update("CAL_MODE", "OFF")
	self.update("CAL_FREQ", 0.0)
	self.update("CAL_DCYC", 0.0)
	self.update("CAL_PHS", 0.0)

	# Current time
	self.update_time()

	self.update("OFFSET0", 0.0)
	self.update("SCALE0", 1.0)
	self.update("OFFSET1", 0.0)
	self.update("SCALE1", 1.0)
	self.update("OFFSET2", 0.0)
	self.update("SCALE2", 1.0)
	self.update("OFFSET3", 0.0)
	self.update("SCALE3", 1.0)

	self.update("ACC_LEN", 0)
	self.update("NBITSADC", 8)
	self.update("PFB_OVER", 4)


	self.update("AZ", 0.0)
	self.update("ZA", 0.0)
	self.update("BMAJ", 0.0)
	self.update("BMIN", 0.0)
	self.update("LST", 0)

	self.update("TSUBINT", 0.0)
	self.update("OFFS_SUB", 0.0)
	self.update("NPOLYCO", 0)
	self.update("NPKT", 0)

DATABUF_KEY = 12987498
DATABUF_HEADER_SIZE = STATUS_SIZE
DATABUF_STRUCT_SIZE = 8192
DATABUF_SIZE_LIMIT = 1024 * 1024 * 1024

class Databuf:

    def __init__(self, databuf_id=1):
	self.buf = shmw.SharedMemoryHandle(DATABUF_KEY+databuf_id - 1)
	self.data_type = self.buf.read(NumberOfBytes=64, offset=0)
	packed = self.buf.read(NumberOfBytes=3*8+3*4, offset=64)
	self.struct_size, self.block_size, self.header_size = \
		    n.fromstring(packed[0:24], dtype=n.int64)
	self.shmid, self.semid, self.n_block= \
		    n.fromstring(packed[24:36], dtype=n.int32)
	self.header_offset = self.struct_size 
	self.data_offset = self.struct_size + self.n_block*self.header_size
	self.dtype = n.int8
	self.read_size = self.block_size
	self.read_all_hdr()
	# Try to grab semid 
	self.semid = shm.getsemid(DATABUF_KEY + databuf_id - 1)

    def delete(self):

	# Remove status shm
	try:
	    shm.remove_semaphore(self.semid)
	        
	except:
	    pass

	# Then remove databuf shm
	try:
	    self.buf.remove()
	except:
	    pass

    def read_hdr(self,block):
        if (block<0 or block>=self.n_block):
            raise IndexError, "block %d out of range (n_block=%d)" \
                    % (block, self.n_block)
        self.hdr[block] = header_from_string(self.buf.read(self.header_size,\
                self.header_offset + block*self.header_size))

    def read_all_hdr(self):
        self.hdr = []
        for i in range(self.n_block):
            self.hdr.append(header_from_string(self.buf.read(self.header_size,\
                self.header_offset + i*self.header_size)))

    def data(self,block):
        if (block<0 or block>=self.n_block):
            raise IndexError, "block %d out of range (n_block=%d)" \
                    % (block, self.n_block)
        self.read_hdr(block) 

        try:
	    # RAW data
            if (self.hdr[block]["DATATYPE"] == "RAW"):
		self.dtype = n.int8
		raw = n.fromstring(self.buf.read(self.block_size, self.data_offset + block*self.block_size), dtype=self.dtype)
                return raw

	    # FOLDED data	
            elif(self.hdr[block]["DATATYPE"] == "FOLDED"):

		self.dtype = n.float
                nbin = self.hdr[block]["NBIN"]
		npol = self.hdr[block]["NPOL"]
		nchan = self.hdr[block]["OBSNCHAN"]

		size = nbin * npol * nchan * 4 # Float is 4 bytes 
		offset = block * size 

		print "nbin = %d  nchan = %d  npol = %d size = %d"%(nbin, nchan, npol, size)

		raw = n.fromstring(self.buf.read(size, offset), dtype=self.dtype)

		print "Len", len(raw)
		raw.shape = (nchan, npol, nbin)
		return raw

        except KeyError:
	    raise KeyError, "Error reading key parameters..."
            pass
        return raw
        
if __name__=="__main__":
    g = Status()
    g.show()

    print 
    print 'keys:', g.keys()
    print 
    print 'values:', g.values()
    print 
    print 'items:', g.items()
    print 

    #g.update_with_gbtstatus()
    g.write()
    g.show()
