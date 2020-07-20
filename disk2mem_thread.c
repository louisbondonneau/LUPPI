/* rawdisk_thread.c
 *
 * Write databuf blocks out to disk.
 */

#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>

#include "fitshead.h"
#include "psrfits.h"
#include "logging.h"
#include "status.h"
#include "databuf.h"
#include "params.h"
#include "cpu_utils.h"

#define STATUS "DISKSTAT"
#include "threads.h"

#define MAX_HEADER_SIZE 128*1024

typedef struct {
    int block_count;
    int hdlen;
    int blocksize;
    double t_write;

} timing_stats;

void print_stats(timing_stats *ts) {
    char strlog[256];
    sprintf(strlog, "Read %d blocks", ts->block_count);
    log_info("disk2mem_thread", strlog);
    float datasize = ts->block_count * (ts->hdlen + ts->blocksize)/ (1024.*1024.);
    sprintf(strlog, "Read %f MB in %f s : %f MB/s", datasize, ts->t_write, datasize/ts->t_write);
    log_info("disk2mem_thread", strlog);
}

void disk2mem_thread(void *_args) {

    char strlog[256];

    // -- Get arguments --
    thread_args *args = (thread_args *)_args;

    // -- cpu affinity --
    setcpuaffinity(args->output_buffer - 1);

    // -- process priority --
    //setabspriority(args->priority);

    // -- Attach to status shared mem area --
    status st;
    timing_stats ts;
    int rv;
    rv = status_attach(&st, args->gpu_id);
    if (rv!=OK) {
        log_error("rawdisk_thread", "Error attaching to status shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)status_detach, &st);
    pthread_cleanup_push((void *)set_exit_status, &st);

    // -- Init status --
    status_lock_safe(&st);
    hputs(st.buf, STATUS, "init");
    status_unlock_safe(&st);

    // -- Read in general parameters --
    obs_params gp;
    struct psrfits pf;
    pf.sub.dat_freqs = NULL;
    pf.sub.dat_weights = NULL;
    pf.sub.dat_offsets = NULL;
    pf.sub.dat_scales = NULL;
    pthread_cleanup_push((void *)free_psrfits, &pf);

    // -- Attach to databuf shared mem --
    databuf *db;
    db = databuf_attach(args->output_buffer);
    if (db==NULL) {
        log_error("rawdisk_thread", "Error attaching to databuf shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)databuf_detach, db);

    // -- Timing stats --
    pthread_cleanup_push((void *)print_stats, &ts);

    /* Init output file */
    char basefilename[64];
    strcpy(basefilename, args->string);
    FILE *fraw = NULL;
    if((fraw=fopen(basefilename,"r")) == NULL) {
        printf("Could not open %s\n", basefilename);
	exit(-1);
    } else printf("%s opened\n", basefilename);


    /* Loop */
    int packetidx=0, npacket=0, ndrop=0, packetsize=0, blocksize=0;
    int curblock=0;
    int block_count=0, blocks_per_file=5120, filenum=0;
    int got_packet_0=0, first=1;
    char *ptr, *hend;

    double t_write = 0.0;
    ts.block_count = 0;
    struct timespec t_write_start, t_write_stop;

    /* header */
    char *hdr_buffer;
    hdr_buffer = (char *)malloc(MAX_HEADER_SIZE * sizeof(char));
    ptr = hdr_buffer;

    // Set databuf to be free
    sleep(5);

    signal(SIGINT,cc);
    while (run) {

        /* Note weaiting status */
        status_lock_safe(&st);
        hputs(st.buf, STATUS, "reading");
        status_unlock_safe(&st);

	/* Read header */
	//printf("Current position in file %d\n", ftell(fraw));
	memset(hdr_buffer, 0, MAX_HEADER_SIZE);

	if (first) {
	//fseek(fraw, 0, SEEK_SET);
	rv = fread(hdr_buffer, 1, MAX_HEADER_SIZE, fraw);
	//fseek(fraw, -MAX_HEADER_SIZE, SEEK_CUR);
	//hend = ksearch(hdr_buffer, "END");
	ts.hdlen = gethlength (hdr_buffer);
	if (ts.hdlen < 0) exit(0);

	fseek(fraw, -MAX_HEADER_SIZE+ts.hdlen, SEEK_CUR);
	}
	else {
	    rv = fread(hdr_buffer, 1, ts.hdlen, fraw);
	}
	//printf("fread=%d bytes   Header len = %d\n", rv, ts.hdlen);

	read_status_params(hdr_buffer, &gp, &pf);

	printf("curblock = %d/%d\n", curblock, db->n_block);

	/* Update status memory */
        status_lock_safe(&st);
	hputi4(st.buf, "DS_TIME", pf.hdr.ds_time_fact);
	hputi4(st.buf, "DS_FREQ", pf.hdr.ds_freq_fact);
	hputi4(st.buf, "ONLY_I", pf.hdr.onlyI);

	hputr8(st.buf, "OBSFREQ", pf.hdr.fctr);
	hputr8(st.buf, "OBSBW", pf.hdr.BW);
	hputi4(st.buf, "OBSNCHAN", pf.hdr.nchan);
	hputi4(st.buf, "NPOL", pf.hdr.npol);
	hputi4(st.buf, "NBITS", pf.hdr.nbits);
	hputr8(st.buf, "TBIN", pf.hdr.dt);
	hputr8(st.buf, "CHAN_BW", pf.hdr.df);

	hputs(st.buf, "TELESCOP", "Nancay");
	hputs(st.buf, "OBSERVER", pf.hdr.observer);
	hputs(st.buf, "FRONTEND", pf.hdr.frontend);
	hputs(st.buf, "BACKEND", pf.hdr.backend);
	hputs(st.buf, "PROJID", pf.hdr.project_id);
	hputs(st.buf, "SRC_NAME", pf.hdr.source);
	hputs(st.buf, "RA_STR", pf.hdr.ra_str);
	hputs(st.buf, "DEC_STR", pf.hdr.dec_str);

	hputs(st.buf, "CAL_MODE", pf.hdr.cal_mode);
	hputs(st.buf, "OBS_MODE", "PSR");

	hputi4(st.buf, "FFTLEN", pf.dedisp.fft_len);
	hputi4(st.buf, "OVERLAP", pf.dedisp.overlap);
	hputi4(st.buf, "CHAN_DM", pf.hdr.chan_dm);

	hputi4(st.buf, "STT_IMJD", (int)pf.hdr.MJD_epoch);
	hputi4(st.buf, "STT_SMJD", (int)((pf.hdr.MJD_epoch-(int)pf.hdr.MJD_epoch)*86400.));
	hputr8(st.buf, "STT_OFFS", 0.0);

        hgeti4(hdr_buffer, "BLOCSIZE", &blocksize);
        status_unlock_safe(&st);

        rv = databuf_wait_free(db, curblock);

	/* Read param struct for this block */
	ptr = databuf_header(db, curblock);
	//rv = fread(ptr, 1, (size_t)ts.hdlen, fraw);
	memcpy(ptr, hdr_buffer, (size_t)ts.hdlen);
	hputi4(ptr, "DS_TIME", 128);
	hputi4(ptr, "ONLY_I", 0);
	//hputr8(ptr, "CHAN_DM", 5.75066);
    //hputr8(ptr, "DM", 0.0);
    hputr8(ptr, "RM", 3.1);
	//hputr8(ptr, "CHAN_DM", 0.0);
	//hputr8(ptr, "DM", 26.7641);
	//hputs(ptr, "SRC_NAME", "B0950+08");
	//hputs(ptr, "OBS_MODE", "PSR");
	hputs(ptr, "OBS_MODE", "SEARCH");
	//hputi4(ptr, "TFOLD", 30);
	hputs(ptr, "DATADIR", "/data2/");
	//hputi4(ptr, "OVERLAP", 64*1024);
	hputi4(ptr, "NBITS", 8);
	//hputi4(ptr, "NBITS", 4);
	//hputs(ptr, "PARFILE", "/ephem/B0809+74.par");
	//hputs(ptr, "PARFILE", "/ephem/B0329+54.par");

	ptr = databuf_data(db, curblock);
	rv = fread(ptr, 1, (size_t)blocksize, fraw);
	printf("read done rv=%d blocksize=%d src=%s\n", rv, blocksize, pf.hdr.source);

        //int i;
	//for (i=0; i<2097152; i++) fprintf(stderr,"%d\n",ptr[i]);
	//exit(0);

	if (rv != blocksize) {
	    sleep(4);
	    hputi4(st.buf, "RUN", 0);
	    break;
	}

        /* Wait for buf to have data */
	if (first) {
	    hputi4(st.buf, "STTVALID", 1);
	    hputi4(st.buf, "PKTIDX", 0);
	    first = 0;
	}    
        rv = databuf_set_filled(db, curblock);

	block_count++;
	printf("databuf_set_filled %d/%d\n\n",block_count,blocks_per_file);
//#if 0
        /* See if we need to open next file */
        if (block_count >= blocks_per_file) {
	    printf("block_count >= blocks_per_file\n\n");
            fclose(fraw);
            filenum++;
            char fname[256];
            sprintf(fname, "%s.%4.4d.raw", pf.basefilename, filenum);
	    sprintf(strlog, "Opening raw file '%s'", fname);
	    log_info("rawdisk_thread", strlog);
            fraw = fopen(fname, "r");
	    //memset(hdr_buffer, 0, MAX_HEADER_SIZE);
	    //rv = fread(hdr_buffer, 1, MAX_HEADER_SIZE, fraw);
	    //printf("hdr_buffer=%s\n",hdr_buffer);
            if (fraw==NULL) {
                log_info("rawdisk_thread", "File do not existe  run=0.");
                sleep(4);
                run = 0;
                hputi4(st.buf, "RUN", 0);
                pthread_exit(NULL);
        
            }
        else
            fseeko(fraw, 0, SEEK_SET);
            block_count=0;
        }

        /* See how full databuf is */
        //total_status = databuf_total_status(db);

//#endif	    
        if (feof(fraw)) {
	    sleep(4);
	    run = 0;
	    hputi4(st.buf, "RUN", 0);
	}    

        /* Mark as free */
        //databuf_set_free(db, curblock);

        /* Go to next block */
        curblock = (curblock + 1) % db->n_block;

        /* Check for cancel */
        pthread_testcancel();

    }

    pthread_exit(NULL);

    pthread_cleanup_pop(0); /* Closes print_stat */
    pthread_cleanup_pop(0); /* Closes databuf_detach */
    pthread_cleanup_pop(0); /* Closes free_psrfits */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes status_detach */

}
