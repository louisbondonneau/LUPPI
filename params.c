/* guppi_params.c
 *
 * Routines to read/write basic system parameters.
 * Use PSRFITS style keywords as much as possible.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "fitshead.h"
#include "psrfits.h"
#include "params.h"
#include "logging.h"

#include "slalib.h"
#include "mjdtime.h"

#ifndef DEGTORAD
#define DEGTORAD 0.017453292519943295769236907684886127134428718885417
#endif
#ifndef RADTODEG
#define RADTODEG 57.29577951308232087679815481410517033240547246656
#endif
#ifndef SOL
#define SOL 299792458.0
#endif

#define DEBUGOUT 0

void get_dbl(char *buf, char *key, double *param, double val) {  
    char logstr[128];
    if (hgetr8(buf, key, param)==0) { 
        *param = val;
        sprintf(logstr,"get_dbl: %s not in status shm! Set to %lf", key, val);
        log_warn("read_obs_params",logstr);     
    }                                          
}

void get_int(char *buf, char *key, int *param, int val) {  
    char logstr[128];
    if (hgeti4(buf, key, param)==0) { 
        *param = val;
        sprintf(logstr,"get_int: %s not in status shm! Set to %d", key, val);
        log_warn("read_obs_params",logstr);     
    }                                          
}

void get_lon(char *buf, char *key, long long *param, long long val) {  
    char logstr[128];
    double dtmp;
    if (hgetr8(buf, key, &dtmp)==0) { 
        sprintf(logstr,"get_lon: %s not in status shm! Set to %lld", key, val);
        log_warn("read_obs_params",logstr);     
	*param = val;
    } else {
        *param = (long long)(llrint(dtmp));
        //param = (long long)(lrint(dtmp));
    }
}

void get_str(char *buf, char *key, char *param, int len) {  
    char logstr[128];
    if (hgets(buf, key, len, param)==0) {     
        sprintf(logstr,"get_str: %s not in status shm!", key);
        log_warn("read_obs_params",logstr);     
    }	
}    

void exit_on_missing_int(char *key, int *param, int *val) {  
    if ((param)==(val)) {            
        char logstr[128];
        sprintf(logstr, "%s is required!\n", key);       
        log_error("read_obs_params", logstr);    
        exit(1);    
    }
}

void exit_on_missing_dbl(char *key, double *param, double *val) {  
    if ((param)==(val)) {            
        char logstr[128];
        sprintf(logstr, "%s is required!\n", key);       
        log_error("read_obs_params", logstr);    
        exit(1);    
    }
}

void exit_on_missing_str(char *key, char *param, char *val) {  
    if ((param)==(val)) {            
        char logstr[128];
        sprintf(logstr, "%s is required!\n", key);       
        log_error("read_obs_params", logstr);    
        exit(1);    
    }
}



// Return the beam FWHM in degrees for obs_freq in MHz 
// and dish_diam in m
double beam_FWHM(double obs_freq, double dish_diam) {
    double lambda = SOL/(obs_freq*1e6);
    return 1.2 * lambda / dish_diam * RADTODEG;
}


// Any GB-specific derived parameters go here
void set_obs_params(char *buf, struct psrfits *p) {

    // TODO could double-check telescope name first
    
    // Set the beamwidth
    if (strcmp("GBT", p->hdr.telescope)==0)
        p->hdr.beam_FWHM = beam_FWHM(p->hdr.fctr, 100.0);
    else if (strcmp("GB43m", p->hdr.telescope)==0)
        p->hdr.beam_FWHM = beam_FWHM(p->hdr.fctr, 43.0);
    else if (strcmp("NRT", p->hdr.telescope)==0) 
        p->hdr.beam_FWHM = beam_FWHM(p->hdr.fctr, 94.0);  // A rectangular beam actually
    else
        p->hdr.beam_FWHM = 0.0;

    // Receiver orientations, poln parameters, etc for GBT !!!!
    // Defaults:
    // TODO NRT !!!!!	
    p->hdr.fd_hand = -1;
    p->hdr.fd_sang = 45.0;
    p->hdr.fd_xyph = 0.0;
    // Special cases:
    //   - Linear-feed gregorian rcvrs (L, S, C bands) are rotated
    //     90 degrees from PSRFITS convention.
    if (strcmp("Rcvr1_2", p->hdr.frontend)==0) {
        p->hdr.fd_sang=-45.0;
    } else if (strcmp("Rcvr2_3", p->hdr.frontend)==0) {
        p->hdr.fd_sang=-45.0;
    } else if (strcmp("Rcvr4_6", p->hdr.frontend)==0) {
        p->hdr.fd_sang=-45.0;
    } else if (strcmp("LBand", p->hdr.frontend)==0) {
        p->hdr.fd_sang=-45.0;
    }

    // Backend cross-term phase
    if (strcmp("GUPPI", p->hdr.backend)==0)
        p->hdr.be_phase = -1;
    else 
        p->hdr.be_phase = -1;
    
}

// Read networking parameters
int read_net_params(char *buf, udp_params *u) {
    get_str(buf,"DATAHOST", u->sender, 80);
    get_str(buf,"DST_IP", u->dst_ip, 80);
    get_str(buf,"RCV_IP", u->rcv_ip, 80);
    get_int(buf,"DATAPORT", &u->port, 1491);
    get_str(buf,"PKTFMT", u->packet_format, 32);
    get_int(buf,"PKTSIZE", &u->packet_size, 9000);  //Add LOUIS
    if (strncmp(u->packet_format, "PASP", 4)==0) {
        //u->packet_size = 8208;
        }
    else if (strncmp(u->packet_format, "SONATA", 6)==0) // TODO SonATA
        u->packet_size = 4160;
    else if (strncmp(u->packet_format, "PARKES", 6)==0)
	u->packet_size = 2056;
    else if (strncmp(u->packet_format, "1SFA", 4)==0)
        u->packet_size = 8224;
    else if (strncmp(u->packet_format, "SHORT", 5)==0)
        u->packet_size = 544;
    else if (strncmp(u->packet_format, "FAST4K", 6)==0)
        u->packet_size = 4128;
    //else
    //    u->packet_size = 8208;
    return(OK);
}

// Read networking parameters
int read_fwd_params(char *buf, udp_params *u) {
    get_str(buf,"RAWHOST", u->sender, 80);
    get_int(buf,"RAWPORT", &u->port, 6000);
    // if we don't want to forward
    if (strcmp(u->sender, "None")==0 || strlen(u->sender)==0) return 0;
    else return(1);
}

/* Some code just needs a simple way to get the obs mode string */
void read_obs_mode(char *buf, char *mode) {
    get_str(buf,"OBS_MODE", mode, 8);
}

// Read a status buffer all of the key observation paramters
void read_subint_params(char *buf, obs_params *g, struct psrfits *p) {
    // Parse packet size, # of packets, etc.
    get_lon(buf,"PKTIDX", &g->packetindex, -1L);
    get_int(buf,"PKTSIZE", &g->packetsize, 0);
    get_int(buf,"NPKT", &g->n_packets, 0);
    get_int(buf,"NDROP", &g->n_dropped, 0);
    get_dbl(buf,"DROPAVG", &g->drop_frac_avg, 0.0);
    get_dbl(buf,"DROPTOT", &g->drop_frac_tot, 0.0);
    get_lon(buf,"BLOCSIZE", &g->packets_per_block, 0);
    printf("",g->packets_per_block);
    if (g->packetsize>0)
        g->packets_per_block /= g->packetsize;
    if (g->n_packets>0)
        g->drop_frac = (double) g->n_dropped / (double) g->n_packets;
    else
        g->drop_frac = 0.0;

    // Valid obs start time
    get_int(buf,"STTVALID", &g->stt_valid, 0);

    // Observation params
    get_dbl(buf,"AZ", &p->sub.tel_az, 0.0);
    if (p->sub.tel_az < 0.0) p->sub.tel_az += 360.0;
    get_dbl(buf,"ZA", &p->sub.tel_zen, 0.0);
    get_dbl(buf,"RA", &p->sub.ra, 0.0);
    get_dbl(buf,"DEC", &p->sub.dec, 0.0);

    // Backend HW parameters
    get_int(buf,"ACC_LEN", &g->decimation_factor, 0);
    get_int(buf,"NBITSADC", &g->n_bits_adc, 8);
    get_int(buf,"PFB_OVER", &g->pfb_overlap, 4);

    // Check fold mode 
    int fold=0;
    if (strcmp("PSR", p->hdr.obs_mode)==0) { fold=1; }
    if (strcmp("CAL", p->hdr.obs_mode)==0) { fold=1; }
    if (strcmp("FON", p->hdr.obs_mode)==0) { fold=1; }
    if (strcmp("FOF", p->hdr.obs_mode)==0) { fold=1; }

    // Fold-specifc stuff
    if (fold) {
        get_dbl(buf,"TSUBINT", &p->sub.tsubint, 0.0); 
	//char subint[36];
	//get_str(buf,"TSUBINT", subint, 36); 

        get_dbl(buf,"OFFS_SUB", &p->sub.offs, 0.0); 
        get_int(buf,"NPOLYCO", &p->fold.n_polyco_sets, 0);
    } else {
        int bytes_per_dt = p->hdr.nchan * p->hdr.npol * g->n_bits_adc / 8;
        p->sub.offs = p->hdr.dt * (double)(g->packetindex * g->packetsize / bytes_per_dt) + 0.5 * p->sub.tsubint;
        p->fold.n_polyco_sets = 0;
    }
    { // MJD and LST calcs
        int imjd, smjd, lst_secs;
        double offs, mjd;
        get_current_mjd(&imjd, &smjd, &offs);
        mjd = (double) imjd + ((double) smjd + offs) / 86400.0;
        get_current_lst(mjd, &lst_secs);
        p->sub.lst = (double) lst_secs;
    }

    // Until we need them...
    p->sub.feed_ang = 0.0;
    p->sub.pos_ang = 0.0;
    p->sub.par_ang = 0.0;
    
    // Galactic coords
    slaEqgal(p->sub.ra*DEGTORAD, p->sub.dec*DEGTORAD, &p->sub.glon, &p->sub.glat);
    p->sub.glon *= RADTODEG;
    p->sub.glat *= RADTODEG;
}

// Read a status buffer all of the key observation paramters
int read_status_params(char *buf, obs_params *g, struct psrfits *p) {
    char base[200], dir[200]="/data2";

    //get_int(buf, "N_GPU", &g->n_gpu);

    // Software data-stream modification params
    get_int(buf, "DS_TIME", &p->hdr.ds_time_fact, 1); // Time down-sampling
    get_int(buf, "DS_FREQ", &p->hdr.ds_freq_fact, 1); // Freq down-sampling
    get_int(buf, "ONLY_I", &p->hdr.onlyI, 0);         // Only output Stokes I

    int zero_int = 0;
    double zero_dbl = 0.0;

    // Freq, BW, etc.
    get_dbl(buf, "OBSFREQ", &p->hdr.fctr, 0.0);
    get_dbl(buf, "OBSBW", &p->hdr.BW, 0.0);
    exit_on_missing_dbl("OBSBW", &p->hdr.BW, &zero_dbl);
    get_int(buf, "OBSNCHAN", &p->hdr.nchan, 0);
    exit_on_missing_int("OBSNCHAN", &p->hdr.nchan, &zero_int); //GD
    get_int(buf, "NPOL", &p->hdr.npol, 0);
    exit_on_missing_int("NPOL", &p->hdr.npol, &zero_int);
    get_int(buf, "NBITS", &p->hdr.nbits, 8);
    exit_on_missing_int("NBITS", &p->hdr.nbits, &zero_int);
    get_dbl(buf, "TBIN", &p->hdr.dt, 0.0);
    exit_on_missing_dbl("TBIN", &p->hdr.dt, &zero_dbl);
    get_dbl(buf, "CHAN_BW", &p->hdr.df, 0.0);
    if (p->hdr.df==0.0) p->hdr.df = p->hdr.BW/p->hdr.nchan;
    get_dbl(buf,"SCANLEN", &p->hdr.scanlen, 0.0);
    get_int(buf,"NRCVR", &p->hdr.rcvr_polns, 2);
    get_int(buf,"NBITSADC", &g->n_bits_adc, 8);
    p->hdr.orig_df = p->hdr.df;
    p->hdr.orig_df =p->hdr.BW/p->hdr.orig_nchan;
    p->hdr.orig_nchan = p->hdr.nchan;

    // Observation information
    get_str(buf,"TELESCOP", p->hdr.telescope, 24);
    get_str(buf,"OBSERVER", p->hdr.observer, 24);
    get_str(buf,"SRC_NAME", p->hdr.source, 24);
    get_str(buf,"STT_JDAY", p->hdr.jday, 24);
    get_str(buf,"BEAM_NUM", p->hdr.beam, 24);
    get_str(buf,"FRONTEND", p->hdr.frontend, 24);
    get_str(buf,"BACKEND", p->hdr.backend, 24);
    get_str(buf,"PROJID", p->hdr.project_id, 24);
    get_str(buf,"FD_POLN", p->hdr.poln_type, 8);
    get_str(buf,"POL_TYPE", p->hdr.poln_order, 16);
    get_int(buf,"SCANNUM", &p->hdr.scan_number, 1);
    get_int(buf,"MJDSTART", &p->hdr.smjd_start, 1);
    get_int(buf, "N_GPU", &p->hdr.gpu_id, 1);
    get_str(buf,"DATADIR", dir, 200);
    if (strcmp(p->hdr.poln_order, "AA+BB")==0 ||
        strcmp(p->hdr.poln_order, "INTEN")==0)
        p->hdr.summed_polns = 1;
    else
        p->hdr.summed_polns = 0;
    get_str(buf,"TRK_MODE", p->hdr.track_mode, 16);
    get_str(buf,"RA_STR", p->hdr.ra_str, 16);
    get_str(buf,"DEC_STR", p->hdr.dec_str, 16);
    // Should set other cal values if CAL_MODE is on
    get_str(buf,"CAL_MODE", p->hdr.cal_mode, 8);
    get_str(buf,"OBS_MODE", p->hdr.obs_mode, 8);

    if (!(strcmp(p->hdr.cal_mode, "CAL")==0)) {  // Cals not off
        get_dbl(buf,"CAL_FREQ", &p->hdr.cal_freq, 0.0);
        get_dbl(buf,"CAL_DCYC", &p->hdr.cal_dcyc, 0.0);
        get_dbl(buf,"CAL_PHS", &p->hdr.cal_phs, 0.0);
    }

    // Fold mode specific stuff
    int fold=0;
    get_int(buf, "NBIN", &p->fold.nbin, 2048);
    get_dbl(buf, "TFOLD", &p->fold.tfold, 30.0);
    //p->fold.tfold = 1;
    // TODO
    get_str(buf, "PARFILE", p->fold.parfile, 256);
    if (strcmp("FOLD", p->hdr.obs_mode)==0) { fold=1; }
    if (strcmp("PSR", p->hdr.obs_mode)==0) { fold=1; }
    if (strcmp("CAL", p->hdr.obs_mode)==0) { fold=1; }
    if (strcmp("FON", p->hdr.obs_mode)==0) { fold=1; }
    if (strcmp("FOF", p->hdr.obs_mode)==0) { fold=1; }
    if (fold) 
        p->hdr.nbin = p->fold.nbin;
    else 
        p->hdr.nbin = 1;

    // Coherent dedispersion params
    get_int(buf,"FFTLEN", &p->dedisp.fft_len, 0);
    get_int(buf,"OVERLAP", &p->dedisp.overlap, 0);
    get_dbl(buf,"CHAN_DM", &p->hdr.chan_dm, 0.0);
    get_dbl(buf,"RM", &p->hdr.rm, 0.0);

    //printf("overlap=%d\n",p->dedisp.overlap);fflush(stdout);
    
    { // Start time, MJD
        int mjd_d, mjd_s;
        double mjd_fs;
        get_int(buf,"STT_IMJD", &mjd_d, 0);
        get_int(buf,"STT_SMJD", &mjd_s, 0);
        get_dbl(buf,"STT_OFFS", &mjd_fs, 0.0);
        p->hdr.MJD_epoch = (long double) mjd_d;
        p->hdr.MJD_epoch += ((long double) mjd_s + mjd_fs) / 86400.0;
	printf("mjd_d=%d\n",mjd_d);
        p->hdr.start_day = mjd_d;
        p->hdr.start_sec = mjd_s + mjd_fs;
    }
    
    // Set the base filename
    int i;
    char backend[24];
    strncpy(backend, p->hdr.backend, 24);
    for (i=0; i<24; i++) { 
        if (backend[i]=='\0') break;
        backend[i] = tolower(backend[i]); 
    }
    if ( (strcmp("CAL", p->hdr.obs_mode)==0) || (strcmp("FON", p->hdr.obs_mode)==0)  || (strcmp("FOF", p->hdr.obs_mode)==0) ) { 
        sprintf(base, "%s_%5d_%06d_%04d_BEAM%s_cal", p->hdr.source, p->hdr.start_day, p->hdr.scan_number, (int)p->hdr.fctr, p->hdr.beam);
        //sprintf(base, "%s_%5d_%s_%06d_%04d_cal", backend, p->hdr.start_day, p->hdr.source, p->hdr.scan_number, (int)p->hdr.fctr);
    } else {
        sprintf(base, "%s_%s_%5d_%06d_%04d_BEAM%s", p->hdr.source, p->hdr.jday, p->hdr.start_day, p->hdr.scan_number, (int)p->hdr.fctr, p->hdr.beam);
        //sprintf(base, "%s_%5d_%s_%06d_%04d", backend, p->hdr.start_day, p->hdr.source, p->hdr.scan_number, (int)p->hdr.fctr);
    }
    //dir[200]="/data2";
    sprintf(p->basefilename, "%s/%s", dir, base);
    printf("filename=%s\n",p->basefilename);

    { // Date and time of start
        int YYYY, MM, DD, h, m;
        double s;
        datetime_from_mjd(p->hdr.MJD_epoch, &YYYY, &MM, &DD, &h, &m, &s);
        sprintf(p->hdr.date_obs, "%04d-%02d-%02dT%02d:%02d:%06.3f", 
                YYYY, MM, DD, h, m, s);
    }

    // TODO: call telescope-specific settings here
    // Eventually make this depend on telescope name
    //printf("param: pf.hdr.onlyI=%i\n",p.hdr.onlyI);
    set_obs_params(buf, p);
    printf("param: pf.hdr.onlyI=%i\n",p->hdr.onlyI);
    
    // Now bookkeeping information
    
    {
        int ii, jj, kk;
        int64_t bytes_per_dt = p->hdr.nchan * p->hdr.npol * g->n_bits_adc / 8;
        char key[10];
        double offset, scale, dtmp;
        long long max_bytes_per_file;
	


	char subint[36];
	get_str(buf,"TSUBINT", subint, 36); 
	printf("nsblk   char   tsubint = %s\n",subint);
	printf("nsblk   double tsubint = %f\n",strtod(subint, NULL));
	printf("nsblk   p->sub.tsubint = %f\n",p->sub.tsubint);




        //get_lon(buf,"BLOCSIZE", &p->sub.bytes_per_subint, 0);
	hgeti5(buf, "BLOCSIZE", &p->sub.bytes_per_subint);
	printf("nsblk   p->sub.bytes_per_subint = %lld\n",(long long)p->sub.bytes_per_subint);
	printf("nsblk   bytes_per_dt = %lld\n",(long long)bytes_per_dt);
	printf("nsblk   p->sub.bytes_per_subint/bytes_per_dt = %lld\n",(long long)p->sub.bytes_per_subint/bytes_per_dt);
	//printf("nsblk   BLOCSIZE = %d\n",p->sub.bytes_per_subint);
        p->hdr.nsblk = (uint64_t)((long long)p->sub.bytes_per_subint / (long long)bytes_per_dt);
	printf("nsblk   p->hdr.nsblk = %lld\n",p->hdr.nsblk);
	printf("nsblk   p->hdr.dt = %f\n",p->hdr.dt);
        p->sub.FITS_typecode = TBYTE;
        p->sub.tsubint = (float)(p->hdr.nsblk * p->hdr.dt );
	printf("nsblk   p->sub.tsubint = %f\n",p->sub.tsubint);
	printf("pf. GD > bytes_per_subint=%lld bytes_per_dt=%d nchan=%d npol=%d nbits=%d dt=%e nsblk = %lld\n",p->sub.bytes_per_subint, bytes_per_dt, p->hdr.nchan, p->hdr.npol, p->hdr.nbits, p->hdr.dt, p->hdr.nsblk);
	//exit(0);
        if (fold) { 
            //p->hdr.nsblk = 1;
            p->sub.FITS_typecode = TFLOAT;
            get_dbl(buf,"TSUBINT", &p->sub.tsubint, 0.0); 
            p->sub.bytes_per_subint = (long long)(sizeof(float) * (long long)p->hdr.nbin * (long long)p->hdr.nchan * (long long)p->hdr.npol);

	    printf("pf. GD > bytes_per_subint=%lld bytes_per_dt=%d nchan=%d npol=%d nbits=%d dt=%e nsblk = %lld\n",p->sub.bytes_per_subint, bytes_per_dt, p->hdr.nchan, p->hdr.npol, p->hdr.nbits, p->hdr.dt, p->hdr.nsblk);

            max_bytes_per_file = PSRFITS_MAXFILELEN_FOLD * 1073741824L;
        } else {
            max_bytes_per_file = PSRFITS_MAXFILELEN_SEARCH * 1073741824L;
        }
        // Will probably want to tweak this so that it is a nice round number
        if (p->sub.bytes_per_subint!=0)
            p->rows_per_file = p->hdr.ds_freq_fact * p->hdr.ds_time_fact * 8.0/p->hdr.nbits *
                (p->hdr.onlyI ? 4 : 1) * max_bytes_per_file / 
                p->sub.bytes_per_subint;

        // Free the old ones in case we've changed the params
	free_psrfits(p);

        // Allocate the subband arrays
        p->sub.dat_freqs = (float *)malloc(sizeof(float) * p->hdr.nchan);
        p->sub.dat_weights = (float *)malloc(sizeof(float) * p->hdr.nchan);
        // The following correctly accounts for the middle-of-bin FFT offset
        // XXX This might not be correct for coherent dedisp mode.  Need 
        //     to determine the right way of denoting which nodes are getting
        //     which channels
        dtmp = p->hdr.fctr - 0.5 * p->hdr.BW;
        for (ii = 0 ; ii < p->hdr.nchan ; ii++) {
            //p->sub.dat_freqs[ii] = dtmp + ii * p->hdr.df; // Orig version
            p->sub.dat_freqs[ii] = dtmp + (ii+0.5) * p->hdr.df;
            p->sub.dat_weights[ii] = 1.0;
        }
        // Explicitly weight the DC and Nyquist channels zero
        // because of how power is split between them
        // XXX this needs to be changed for coherent dedisp..
        //p->sub.dat_weights[0] = 0.0;
        
        p->sub.dat_offsets = (float *)malloc(sizeof(float) *  
                                             p->hdr.nchan * p->hdr.npol);
        p->sub.dat_scales = (float *)malloc(sizeof(float) *  
                                            p->hdr.nchan * p->hdr.npol);
        for (ii = 0 ; ii < p->hdr.npol ; ii++) {
            sprintf(key, "OFFSET%d", ii);
            get_dbl(buf,key, &offset, 0.0);
            sprintf(key, "SCALE%d", ii);
            get_dbl(buf,key, &scale, 0.0);
            for (jj = 0, kk = ii*p->hdr.nchan ; jj < p->hdr.nchan ; jj++, kk++) {
                p->sub.dat_offsets[kk] = offset;
                p->sub.dat_scales[kk] = scale;
            }
        }
    }
    
    // Read information that is appropriate for the subints
    read_subint_params(buf, g, p);
    p->hdr.azimuth = p->sub.tel_az;
    p->hdr.zenith_ang = p->sub.tel_zen;
    p->hdr.ra2000 = p->sub.ra;
    p->hdr.dec2000 = p->sub.dec;
    p->hdr.start_lst = p->sub.lst;
    p->hdr.feed_angle = p->sub.feed_ang;
    return(OK);
}

void free_psrfits(struct psrfits *p) {
    // Free any memory allocated in to the psrfits struct
    if (p->sub.dat_freqs) free(p->sub.dat_freqs);
    if (p->sub.dat_weights) free(p->sub.dat_weights);
    if (p->sub.dat_offsets) free(p->sub.dat_offsets);
    if (p->sub.dat_scales) free(p->sub.dat_scales);
}
