from luppi_utils import *

# Set Logging
#log = logfile("set_ata_params.py")

#b = Status(log)


def set_ata_params(b):

    b.update("TELESCOP", "ATA")
    b.update("FRONTEND", "BEAMFORMER")
    b.update("BACKEND", "NUPPI")
    b.update("PKTFMT", "SONATA")
    b.update("PKTSIZE", 4160)

    b.update("DATAHOST", "b16-fpga3-2")
    b.update("DATAPORT", 50000)

    b.update("NRCVR", 1)
    b.update("NBITS", 8)
    b.update("NPOL", 4)
    b.update("POL_TYPE", "IQUV")
    b.update("TRK_MODE", "TRACK")

    b.update("FD_POLN", "LIN")
    b.update("DATADIR", "/export/pgpu1/pulsar_data")

    b.write()
