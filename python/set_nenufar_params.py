ports_fwd = {'sr1': 6001, 'sr2': 6002, 'sr3': 6003, 'sr4': 6004}



def set_nenufar_params(b):

    b.update("TELESCOP", "nenufar")
    b.update("FRONTEND", "LaNewBa")
    b.update("BACKEND", "LUPPI")
    b.update("PKTFMT", "PASP")
    b.update("PKTSIZE", 9999) #8208

    b.update("NRCVR", 2)
    b.update("NBITS", 8)
    b.update("NPOL", 4)
    b.update("POL_TYPE", "IQUV")
    b.update("TRK_MODE", "TRACK")

    # Default params
    b.update("EQUINOX", 2000.0)
    b.update("OBS_MODE", "PSR")
    b.update("OBSFREQ", 0.0000)
    b.update("OBSBW", 0.0000)
    b.update("CHAN_BW", 200./1024.)
    b.update("DM", 0.001)
    b.update("RM", 0.0)
    b.update("CHAN_DM", 0.001)
    b.update("TOTNCHAN", 0)
    b.update("N_DS", 1)
    b.update("N_GPU", 1)
    b.update("NBIN", 2048)
    b.update("TFOLD", 10.737)
    b.update("OBS_LEN", 0.0)
    #b.update("SCANNUM", 0)
    b.update("SCANLEN", 0) #7200
    b.update("OBSERVER", "LB")
    b.update("SNGPULSE", "False")
    b.update("PROJID", "ES03")

    # ROACH infos
    b.update("DATADIR", "/data/")
    b.update("DATAHOST", "1490")
    b.update("RAWHOST", "")
    b.update("DST_IP", "224.2.3.1")
    b.update("RCV_IP", "192.168.5.100")
    b.update("PFB_OVER", 8)
    b.update("NBITSADC", 8)

    b.update("CAL_FREQ", 3.335834)
    b.update("CAL_DCYC", 0.5)
    b.update("CAL_PHS", 0.0)
