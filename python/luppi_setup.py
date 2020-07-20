from luppi_utils import *
from read_config_xml import *
from set_ata_params import *
import set_nenufar_params
from configure_ROACH import *
from optparse import OptionParser
import argparse
import signal, sys, os, socket #corr

MASTER = 'undysputedbk1'
CONFIG_FILE = '/home/louis/obs/config.xml'
PAR_DIRECTORY = '/ephem'
TELESCOP = 'NC'   # Either set to NC or ATA
N_GPU = 1
'''
if socket.gethostname()== MASTER:
    try:
        import valon_synth
    except ImportError:
        print 'Could not import valon_synth module'
'''

# List of boffiles
#BOF_128CHAN = 'pasp_8i128c1024s4g.bof-11282010'
BOF_2048CHAN = ''
BOF_1024CHAN = 'pasp_8i1024c1024s4g-512m.bof'
BOF_256CHAN  = 'pasp_8i256c1024s4g-512m.bof'
BOF_128CHAN  = 'pasp_8i128c1024s4g-512m.bof'
BOF_32CHAN   = 'pasp_8i32c1024s4g-512m.bof'
BOF_16CHAN   = 'pasp_8i16c1024s4g-512m.bof'

usage = "usage: %prog [options]"


# Parse command line

# Check that something was given on the command line
nargs = len(sys.argv) - 1

# Dict containing list of key/val pairs to update in
# the status shared mem
update_list = {}
def add_param(option, opt_str, val, parser, *args):
    if val!='none':
        update_list[args[0]] = val





CEP_FRAME_HEADER_SIZE = 128    # bits
CEP_BEAMLET_SIZE      = 2*2    # word (2 polars * (Re+Im))
CEP_BM_0_DATA_SIZE    = 16     # bits
CEP_FRAME_MAX_SIZE    = 8952*8 # bits, set in LANEWBA_TOP|i_GEDEK10G_STREAMER_0/1

def CEP_FRAME_SIZE(nof_blocks, nof_beamlets, BM):
  """Returns bit size of such CEP frame"""
  return (CEP_FRAME_HEADER_SIZE + (nof_beamlets * CEP_BEAMLET_SIZE * nof_blocks * (CEP_BM_0_DATA_SIZE>>BM)))

def CEP_FRAME_SIZE_VALID(nof_blocks, nof_beamlets, BM):
  """Return True if such CEP frame is valid"""
  return (  (CEP_FRAME_SIZE(nof_blocks, nof_beamlets, BM) < CEP_FRAME_MAX_SIZE ) \
         and nof_blocks   >    0                          \
         and nof_blocks   <  256                          \
         and nof_beamlets >    0                          \
         and nof_beamlets <= 384                          \
         and (BM == 0 or BM == 1 or BM == 2)              \
         and (  (BM == 0) and ((nof_beamlets % 1) == 0)   \
             or (BM == 1) and ((nof_beamlets % 2) == 0)   \
             or (BM == 2) and ((nof_beamlets % 4) == 0))  \
         )

#
# Function to read a XML Configuration file
# or read command line arguments
# to fill a status shm memory
#
def fill_status_shm(opt, log, b, update_list, gpu_id):
    print "fill_status_shm\n"
    if opt.default:
	b.set_default_params()
    b.update("TELESCOP", TELESCOP)	
    
    # Read config.xml
    if opt.configxml:
	log.info('Read %s'%CONFIG_FILE)
	read_config_xml(b, CONFIG_FILE)
	b.update_time()

    # Set default parameter for specific telescope
    if b['TELESCOP']=='ATA':
	log.info('Set ATA default params')
	set_ata_params(b)
    elif b['TELESCOP']=='NC':
	log.info('Set NenuFAR default params')
	set_nenufar_params.set_nenufar_params(b)

    # If host is ripata, change the port number (data are forwarded)
    if socket.gethostname() in ['ripata']:
	if gpu_id==0:
	    b.update("DATAHOST", "sr2-10gbe")
	    b.update("DATAPORT", 6002 + 10)
	elif gpu_id==1:
	    b.update("DATAHOST", "sr3-10gbe")
	    b.update("DATAPORT", 6003 + 10)
	

    # Update AZ/ZA
    b.update_azza()

    # Apply explicit command line values
    # These will always take precedence over defaults now
    for (k,v) in update_list.items():
	b.update(k,v)

    # Tune the frequency and receiving port for the second GPU
    if socket.gethostname() in ['ripata']:
        if gpu_id == 1:
	    b.update("OBSFREQ", b['OBSFREQ'])
	    #b.update("OBSFREQ", b['OBSFREQ']+b['OBSBW']/(b['N_DS']*b['N_GPU']))
    #else:
        #if gpu_id == 1:
	    #b.update("OBSFREQ", b['OBSFREQ']+b['OBSBW']/2.0)
	    #b.update("DATAPORT", b['DATAPORT']+1)

    # Multicast mode
    if opt.f_gpu0 or gpu_id == 1 and opt.f_gpu1:
        b.update("DST_IP", "ripata-10gbe")
        b.update("RCV_IP", set_nenufar_params.ports_fwd[socket.gethostname()] )
    else:
        b.update("RAWHOST", "None")


    if ((not b['LOWCHAN']) or (not b['HIGHCHAN'])) and ((not b['OBSBW']) or (not b['OBSNCHAN']) or (not b['OBSFREQ'])):
        print 'ERROR information missing about channelization'
        log.warning("ERROR information missing about channelization")
        exit()



    if b['LOWCHAN'] and b['HIGHCHAN']:
	if (b['N_GPU']==1):
	    b.update("OBSNCHAN", b['HIGHCHAN']-b['LOWCHAN'] + 1 )
	    b.update("TOTNCHAN", b['OBSNCHAN'] )
	    b.update("OBSBW", b['OBSNCHAN'] * b['CHAN_BW'])
	    b.update("OBSFREQ", ((b['LOWCHAN'] + b['HIGHCHAN'] )/2.) * b['CHAN_BW'] )
    '''
	if (b['N_GPU']==2):
	    if (gpu_id==1):
	        
	    if (gpu_id==2):
    '''        
    
    
    b.update("SRC_NAME", b['SRC_NAME'].upper())
    # Calc useful values
    b.update("TBIN", abs(b['TOTNCHAN']/b['OBSBW']*1e-6))
    #b.update("OBSNCHAN", b['TOTNCHAN']/(b['N_DS']*b['N_GPU']))
    b.update("OBSNCHAN", b['TOTNCHAN']/(b['N_DS']*b['N_GPU']))
    b.update("OBSBW", b['OBSBW']/(b['N_DS']*b['N_GPU']))
    #b.update("CHAN_BW", b['OBSBW']/b['OBSNCHAN'])

    # Convert RA_STR and DEC_STR in radians
    b.convert_pos_to_rad()

    # Calculate nof_blocks fro PKTSIZE
    if b['TELESCOP']=='nenufar':
        b.update("DATAPORT", b['DATAPORT'])
        nof_blocks=1
        while CEP_FRAME_SIZE_VALID(nof_blocks, b['OBSNCHAN'], 1):
            nof_blocks = 2*nof_blocks
        nof_blocks = nof_blocks/2
        packetsize=b['OBSNCHAN']*nof_blocks*4 + 16
        b.update("PKTSIZE", packetsize)
    #### CORRECTION in FREQUENCY !!!!
    #b.update("OBSFREQ", b['OBSFREQ']-b['CHAN_BW']/2.0)

    if opt.search:
        b.update("OBS_MODE", 'SEARCH')
    elif opt.cal or opt.fcaloff or opt.fcalon:   
        b.update("OBS_MODE", 'CAL')
    elif opt.raw:   
        b.update("OBS_MODE", 'RAW')
    else:
        b.update("OBS_MODE", 'PSR')

    # Set the databuf size in Mb
    databuf_mb = 8192
    
    
    # Params for coherent
    if b['OBS_MODE']=='PSR':

	# First read the DM options from command line, if not try to read a parfile
        if opt.no_dm:
	    b.update("DM", 0.001)
	else:
	    b.update("PARFILE","%s/%s.par"%(PAR_DIRECTORY,b['SRC_NAME']))
	    try:
		b.update("DM", dm_from_parfile(b['PARFILE']))
	    except:
	        b.update("DM", 0.001)
        if opt.defaraday:
            b.update("RM", rm_from_parfile(b['PARFILE']))
        else:
            b.update("RM", 0.0)

	fftlen, nfft, overlap_r, npts, blocsize = get_dedisp_params(b, databuf_mb, opt.small_blocksize)

	b.update("FFTLEN", fftlen)
	b.update("OVERLAP",overlap_r)
	b.update("BLOCSIZE", blocsize)


    # Params for search Mode
    if b['OBS_MODE']=='SEARCH':
    
        if args.nbits:
            b.update("NBITS", int(args.nbits))
        else:
            b.update("NBITS", 32)
        if opt.no_dm:
            b.update("DM", 0.001)
        # First read the DM options from command line, if not try to read a parfile
        if opt.no_dm:
            b.update("DM", 0.001)
        else:
            b.update("PARFILE","%s/%s.par"%(PAR_DIRECTORY,b['SRC_NAME']))
            try:
                b.update("DM", dm_from_parfile(b['PARFILE']))
            except:
                b.update("DM", 0.001)
            if opt.defaraday:
                b.update("RM", rm_from_parfile(b['PARFILE']))
            else:
                b.update("RM", 0.0)

	# Record either Total intensity or full pol
	if opt.onlyI:
	    b.update("ONLY_I", 1)
	else:
	    b.update("ONLY_I", 0)

	# 4bits survey mode with full polarization is not working yet - Force to be 8 bits
	if b['NBITS']==4 and b['ONLY_I']==0:
	    log.warning("4-bit survey mode with full polarization is not working yet - Force to write 8-bit data")
	    b.update("NBITS", 8)
	    
	fftlen, nfft, overlap_r, npts, blocsize = get_dedisp_params(b, databuf_mb, opt.small_blocksize)

	b.update("FFTLEN", fftlen)
	b.update("OVERLAP", overlap_r)
	b.update("BLOCSIZE", blocsize)

	b.update("NBIN", int(0))  # Specific to search mode

        if (b['DS_TIME'] != 1):
            dstime = b['DS_TIME']
            log.info('Should check DS_TIME : check if (fftlen-overlap_r)/32 / DS_TIME is an integer')
            log.info('  dstime= %d'%(dstime))
            num = (fftlen-overlap_r) / 32
            log.info('  (fftlen-overlap_r)/32= = %d   (with fftlen= %d overlap_r= %d)'%(num,fftlen,overlap_r))
            while (num%dstime != 0):
                dstime -= 1
            log.info('  dstime= %d'%(dstime))
            b.update("DS_TIME", dstime)

    # Params for calibration Mode
    if b['OBS_MODE']=='CAL':
        
	b.update("DM", 0.001)
	fftlen, nfft, overlap_r, npts, blocsize = get_dedisp_params(b, databuf_mb, opt.small_blocksize)

	b.update("FFTLEN", fftlen)
	b.update("OVERLAP", overlap_r)
	b.update("BLOCSIZE", blocsize)

	b.update("CAL_MODE", "ON")

	# Flux Calibration   
        if opt.fcaloff:
	    b.update("OBS_MODE", "FOF")
        if opt.fcalon:
	    b.update("OBS_MODE", "FON")

    # Params for Raw data recording Mode
    if b['OBS_MODE']=='RAW':
	npts_max_per_chan = databuf_mb*1024*1024/4/b['OBSNCHAN']
	fftlen = 2**18 #npts_max_per_chan
	nfft = 1
	npts = nfft*fftlen
	blocsize = npts*b['OBSNCHAN']*4

	b.update("FFTLEN", fftlen)
	b.update("OVERLAP", 0)
	b.update("BLOCSIZE", blocsize)

        b.update("DATADIR", "/data2")

    # Log
    log.info("fftlen=%d nfft=%d nchan=%d npts=%d blocsize=(4*npts*nchan)=%d"%(fftlen, nfft, b['OBSNCHAN'], npts, blocsize))

    b.update("CHAN_DM", b['DM'])

    # Scan number
    try:
	scan = n.genfromtxt('/obs/BEAM'+b['BEAM_NUM']+'/scannum.txt')
	scan = int(scan)
	#scan = b['SCANNUM']
        if (opt.inc):
            scan += 1
    except IOError:
	scan = int(int(b['BEAM_NUM'])*250000)
    n.savetxt('/obs/BEAM'+b['BEAM_NUM']+'/scannum.txt', [scan])
    b.update("SCANNUM", scan)
    # Set base filename
    if b['OBS_MODE']=='CAL' or opt.cal:
        base = "%s_%5d_%s_%06d_cal"%(b['BACKEND'].lower(), b['STT_IMJD'], b['SRC_NAME'], b['SCANNUM'])
    else:	
        #base = "%s_%5d_%s_%06d"%(b['BACKEND'].lower(), b['STT_IMJD'], b['SRC_NAME'], b['SCANNUM'])	
        base = "%s_%s_%5d_%06d_%04d_BEAM%s"%(b['SRC_NAME'],b['STT_JDAY'], b['STT_IMJD'], b['SCANNUM'], b['OBSFREQ'], b['BEAM_NUM'])
        #base = "%s_%s_%5d_%04d_BEAM%s"%(b['SRC_NAME'],b['STT_JDAY'], b['STT_IMJD'], b['OBSFREQ'], b['BEAM_NUM'])	
        #base = "%s_D%04u%02u%02uT%02u%02u%02u_%5d_%06d"%(b['SRC_NAME'],maintenant.year, maintenant.month, maintenant.day,maintenant.hour, maintenant.minute, maintenant.second, b['STT_IMJD'], b['SCANNUM'])

    b.update("BASENAME", base)
	

    # Record info to log file
    log.info('   SRC_NAME : %s'%b['SRC_NAME'])
    log.info('  OBS_MODE : %s'%b['OBS_MODE'])
    log.info('   CHAN_DM : %s'%b['CHAN_DM'])
    log.info('   OVERLAP : %s'%b['OVERLAP'])
    log.info('     TFOLD : %s'%b['TFOLD'])

    
    # Write into shm
    b.write()

    # Display
    b.show()


# Func to add a key/val setting option to the command line.
# longopt, short are the command line flags
# name is the shared mem key (ie, SCANNUM, RA_STR, etc)
# type is the value type (string, float, etc)
# help is the help string for -h
par = OptionParser(usage)
def add_param_option(longopt, name, help, type="string", short=None):
    if short!=None:
        par.add_option(short, longopt, help=help, 
                action="callback", callback=add_param, 
                type=type, callback_args=(name,))
    else:
        par.add_option(longopt, help=help, 
                action="callback", callback=add_param, 
                type=type, callback_args=(name,))

# Non-parameter options
par.add_option('-C',"--config", dest="configxml",
        action="store_true",
        help="Config file in xml format")

par.add_option('-S',"--start", dest="start",
        action="store_true",
        help="Send START signal to ROACH")

par.add_option('-X',"--exit", dest="exit",
        action="store_true",
        help="Send EXIT signal to nuppi_daq")

par.add_option('-r',"--reload", dest="reload",
        action="store_true", 
	help="Force to reload the design to ROACH")

par.add_option('-d',"--dumpadc", dest="dumpadc",
        action="store_true", 
	help="Calc stats from ADC Bram ")

par.add_option("-U", "--update", dest="update",
        help="Run in update mode",
        action="store_true", default=False)

par.add_option("-D", "--default", dest="default",
        help="Use all default values",
        action="store_true", default=True)

par.add_option("-f", "--force", dest="force",
        help="Force guppi_set_params to run even if unsafe",
        action="store_true", default=False)

par.add_option("-c", "--cal", dest="cal",
        help="Setup for cal scan (folding mode)",
        action="store_true", default=False)

par.add_option("-i", "--increment_scan", dest="inc",
        help="Increment scan num",
        action="store_true", default=False)

par.add_option("-s", "--search", dest="search",
        help="Don't fold, write (raw or dedispersed) time series",
        action="store_true", default=False)

par.add_option("-R", "--raw", dest="raw",
        help="Write the raw baseband data on disk",
	action="store_true", default=False)

par.add_option("--onlyi", dest="onlyI",
        help="Only record total intensity",
        action="store_true", default=False)

par.add_option("-g", "--f_gpu0", dest="f_gpu0",
        help="Forward data from GPU0",
        action="store_true", default=False)

par.add_option("-G", "--f_gpu1", dest="f_gpu1",
        help="Forward data from GPU1",
        action="store_true", default=False)

par.add_option("--no_dm", dest="no_dm",
        help="Force DM to be 0.001 for search mode",
        action="store_true", default=False)

par.add_option("--defaraday", dest="defaraday",
        help="use RM from param file for coherent defaraday",
        action="store_true", default=False)

par.add_option("--small_blocksize", dest="small_blocksize",
        help="force a small blocksize",
        action="store_true", default=False)

par.add_option("--fcalon", dest="fcalon",
        help="Flux cal ON mode",
        action="store_true", default=False)

par.add_option("--fcaloff", dest="fcaloff",
        help="Flux cal OFF mode",
        action="store_true", default=False)

# Parameter-setting options
add_param_option("--dst_ip",
        name="DST_IP", type="string", 
        help="Use a particular destination IP for the multicast")
add_param_option("--rcv_ip",
        name="RCV_IP", type="string", 
        help="Use a particular receving IP for the multicast")
add_param_option("--scannum", short="-n", 
        name="SCANNUM", type="int",
        help="Set scan number")
add_param_option("--tscan", short="-T",
        name="SCANLEN", type="float",
        help="Scan length (sec)")
add_param_option("--parfile", short="-P",
        name="PARFILE", type="string", 
        help="Use this parfile for folding")
add_param_option("--tfold", short="-t",
        name="TFOLD", type="float",
        help="Fold dump time (sec)")
add_param_option("--bins", short="-b",
        name="NBIN", type="int",
        help="Number of profile bins for folding")
add_param_option("--cal_freq",
        name="CAL_FREQ", type="float",
        help="Frequency of pulsed noise cal (Hz, default 25.0)")
add_param_option("--dstime", 
        name="DS_TIME", type="int",
        help="Downsample in time (int, power of 2)")
add_param_option("--dsfreq", 
        name="DS_FREQ", type="int",
        help="Downsample in freq (int, power of 2)")
add_param_option("--obs", 
        name="OBSERVER", type="string",
        help="Set observers name")
add_param_option("--src", 
        name="SRC_NAME", type="string",
        help="Set observed source name")
add_param_option("--jday", 
        name="STT_JDAY", type="string",
        help="Set julian day (DYYYYMMDDTHHMM)")
add_param_option("--beam", 
        name="BEAM_NUM", type="string",
        help="Set beam number (BEAMX)")
add_param_option("--projid", 
        name="PROJID", type="string",
        help="Set project ID (ES03 default)")
add_param_option("--smjdstart", 
        name="MJDSTART", type="int",
        help="Start seconde programtion (default =-1 start now)")
add_param_option("--dm", 
        name="DM", type="float",
        help="Optimize overlap params using given DM")
add_param_option("--ra", 
        name="RA_STR", type="string",
        help="Set source R.A. (hh:mm:ss.s)")
add_param_option("--dec", 
        name="DEC_STR", type="string",
        help="Set source Dec (+/-dd:mm:ss.s)")
add_param_option("--freq", 
        name="OBSFREQ", type="float",
        help="Set center freq (MHz)")
add_param_option("--bw", 
        name="OBSBW", type="float",
        help="Hardware total bandwidth (MHz)")
add_param_option("--dataport", 
        name="DATAPORT", type="int",
        help="Port number from the beamformer")
add_param_option("--nchan", 
        name="TOTNCHAN", type="int",
        help="Number of hardware channels")
add_param_option("--lowchan", 
        name="LOWCHAN", type="int",
        help="Lower channel")
add_param_option("--highchan", 
        name="HIGHCHAN", type="int",
        help="Higher channel")
add_param_option("--npol", 
        name="NPOL", type="int",
        help="Number of hardware polarizations")
add_param_option("--nbits", 
        name="NBITS", type="int",
        help="Number of bits to use when recording in search mode")
add_param_option("--feed_pol", 
        name="FD_POLN", type="string",
        help="Feed polarization type (LIN/CIRC)")
add_param_option("--acc_len", 
        name="ACC_LEN", type="int",
        help="Hardware accumulation length")
add_param_option("--packets", 
        name="PKTFMT", type="string",
        help="UDP packet format")
add_param_option("--host",
        name="DATAHOST", type="string",
        help="IP or hostname of data source")
add_param_option("--datadir", 
        name="DATADIR", type="string",
        help="Data output directory (default: current dir)")



(opt,arg) = par.parse_args()


# If extra command line stuff, exit
if (len(arg)>0):
    par.print_help()
    print
    print "luppi_setup: Unrecognized command line values", arg
    print
    sys.exit(0)

# If nothing was given on the command line, print help and exit
if (nargs==0):
    par.print_help()
    print
    print "luppi_setup: No command line options were given, exiting."
    print "  Either specifiy some options, or to use all default parameter"
    print "  values, run with the -D flag."
    sys.exit(0)

# Set Logging
log = logfile("luppi_setup.py")

if opt.dumpadc:
    roach = ROACH(l)
    roach.dump_adc_brams()
    sys.exit()
    

# Send START to the ROACH 
if opt.start:
    # Wait for next half second to send START command
    cur_time = time.time()
    cur_time = cur_time - int(cur_time)
    time.sleep(1.5-cur_time)
    roach = ROACH(l)
    log.info('Reset ROACH counter')
    roach.rearm()
    #log.info('Reset ROACH counter')

    # TO DO : Record Start Time 
    # b.update_time() # Is it important ? Don't think so...
    sys.exit()

# Send EXIT signal to nuppi_daq
if opt.exit:
    for i in range(8):
        log.info('Writting RUN = 0 in status shm #%d'%i)
        print('Writting RUN = 0 in status shm #%d'%i)
        b = Status(log, i)
        b.update("RUN", 0)
        b.write()
    sys.exit()


# Check for ongoing observations
if (os.popen("pgrep nuppi_daq").read() != ""):
    if (opt.force):
        print "Warning: Proceeding to set params even though datataking is currently running!"
    else:
	print """
nuppi_setup: A NUPPI datataking process appears to be running, exiting.
  If you really want to change the parameters, run again with the --force 
  option.  Note that this will likely cause problems with the current 
  observation.
	"""
        #sys.exit(1)


if opt.configxml:
    opt.default = 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--beam', dest='beam',
        help="beam number")
parser.add_argument('--nbits', dest='nbits',
        help="nbits")

args, unknown = parser.parse_known_args()
print(args.beam)

# Create and init status shm
log.info('Create status shm BEAM '+ str(args.beam))
print 'Create status shm BEAM '+ str(args.beam)
b = Status(log, int(args.beam))
b.set_default_params()

fill_status_shm(opt, log, b, update_list, int(args.beam))




