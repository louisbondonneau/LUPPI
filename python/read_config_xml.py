#!/usr/bin/env python

#from elementtree import ElementTree as ET
import xml.etree.ElementTree as ET

# DEPRECATED : To update

def read_config_xml(b, CONFIG_FILE):
    """
    Read the config.xml file
    and fill shm in PSRFITS format
    http://www.atnf.csiro.au/research/pulsar/psrfits/fitsdef.html
    """

    doc = ET.parse(CONFIG_FILE)

    iter = doc.getiterator()
    for element in iter:
        if 'telescope' == element.tag:
            b.update("TELESCOP",element.text)
        if 'proj_id' == element.tag:
            b.update("PROJID",element.text)
        if 'observer' == element.tag:
            b.update("OBSERVER",element.text)
        if 'ra_str' == element.tag:
            b.update("RA_STR",element.text)     
        if 'dec_str' == element.tag:
            b.update("DEC_STR",element.text)    
        if 'front_end' == element.tag:
            b.update("FRONTEND",element.tag)
        if 'fd_poln' == element.tag:
            b.update("FD_POLN",element.text)
        if 'equinox' == element.tag:
            b.update("EQUINOX",float(element.text))
        if 'obs_mode' == element.tag:
            b.update("OBS_MODE",element.text)
        if 'obsfreq' == element.tag:
            b.update("OBSFREQ",float(element.text))
        if 'obsbw' == element.tag:
            b.update("OBSBW",float(element.text))
        if 'totnchan' == element.tag:
            b.update("TOTNCHAN",int(element.text))
        if 'n_ds' == element.tag:
            b.update("N_DS",int(element.text))
        if 'n_gpu' == element.tag:
            b.update("N_GPU",int(element.text))
        if 'n_bins' == element.tag:
            b.update("NBIN",int(element.text))
        if 't_dump' == element.tag:
            b.update("TFOLD",float(element.text))
        if 'scan' == element.tag:
            b.update("SCANNUM",int(element.text))
        if 'scanlen' == element.tag:
            b.update("SCANLEN",float(element.text))

        if 'single_pulse' == element.tag:
            b.update("SNGPULSE",element.text)

        if 'nb_psr' == element.tag:
            b.update("NB_PSR",int(element.text))
        # TODO : check MAX PSR    
        if 'NB_PSR' in b.keys():
          for i in range(b['NB_PSR']):
            if 'psr_name_%d'%i == element.tag:
                b.update("SRC_NAME%d"%i,element.text)
            if 'chan_dm_%d'%i == element.tag:
                b.update("DM%d"%i,float(element.text))

