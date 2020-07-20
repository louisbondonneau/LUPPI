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

//log
//19/03/20  extract "BLOCKSIZE" with hgeti5 (int64_t) in place of hgeti4 (int)
//19/03/20  int64_t blocksize=0 in place of int
//19/03/20  improv log 'Error writing data' with blocksize value

typedef struct {
    int block_count;
    int hdlen;
    int64_t blocksize;
    double t_write;

} timing_stats;

void print_timing_stats(timing_stats *ts) {
    char strlog[256];
    sprintf(strlog, "Wrote %d blocks", ts->block_count);
    log_info("rawdisk_thread", strlog);
    float datasize = ts->block_count * (ts->hdlen + ts->blocksize)/ (1024.*1024.);
    sprintf(strlog, "Wrote %f MB in %f s : %f MB/s", datasize, ts->t_write, datasize/ts->t_write);
    log_info("rawdisk_thread", strlog);
}

int safe_fclose(FILE *f) {
    if (f==NULL) return 0;
    sync();
    return fclose(f);
}

void rawdisk_thread(void *_args) {

    char strlog[256];

    // -- Get arguments --
    thread_args *args = (thread_args *)_args;

    // -- cpu affinity --
    printf("rawdisk_thread: setcpuaffinity(%d);\n",args->output_buffer - 1);
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
    db = databuf_attach(args->input_buffer);
    if (db==NULL) {
        log_error("rawdisk_thread", "Error attaching to databuf shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)databuf_detach, db);

    // -- Timing stats --
    pthread_cleanup_push((void *)print_timing_stats, &ts);

    /* Init output file */
    FILE *fraw = NULL;
    pthread_cleanup_push((void *)safe_fclose, fraw);

    /* Loop */
    int64_t blocksize=0;
    int packetidx=0, npacket=0, ndrop=0, packetsize=0;
    int curblock=0;
    int block_count=0, blocks_per_file=5120, filenum=0;
    int got_packet_0=0, first=1;
    size_t written;
    char *ptr, *hend;

    double t_write = 0.0;
    ts.block_count = 0;
    struct timespec t_write_start, t_write_stop;
    signal(SIGINT,cc);
    while (run) {

        /* Note waiting status */
        status_lock_safe(&st);
        hputs(st.buf, STATUS, "waiting");
        status_unlock_safe(&st);

        /* Wait for buf to have data */
        rv = databuf_wait_filled(db, curblock);
	printf("                         1 RAWDISK_THREAD rv = %d\n",rv);
        if (rv!=0) continue;
	printf("                         2 RAWDISK_THREAD\n");fflush(stdout);

        /* Read param struct for this block */
        ptr = databuf_header(db, curblock);
        if (first) {
	    printf("rawdisk_thread: read_status_params");
            read_status_params(ptr, &gp, &pf);
            first = 0;
        }    
        else
            read_subint_params(ptr, &gp, &pf);

        /* Parse packet size, npacket from header */
        hgeti4(ptr, "PKTIDX", &packetidx);
        hgeti4(ptr, "PKTSIZE", &packetsize);
        hgeti4(ptr, "NPKT", &npacket);
        hgeti4(ptr, "NDROP", &ndrop);
        hgeti5(ptr, "BLOCSIZE", &blocksize);
	ts.blocksize = blocksize;
	printf("PKTIDX= %d\n",packetidx);
        /* Wait for packet 0 before starting write */
        if (got_packet_0==0 && packetidx==0 && gp.stt_valid==1) {
            got_packet_0 = 1;
	    printf("rawdisk_thread: read_status_params");
            read_status_params(ptr, &gp, &pf);
            char fname[256];
            sprintf(fname, "%s.%4.4d.raw", pf.basefilename, filenum);
            sprintf(strlog, "Opening raw file '%s'", fname);
            log_info("rawdisk_thread", strlog);

	    // -- Check if file already exists, don't want to write on it --
	    if(fopen(fname, "r")!=NULL) {
	        log_error("rawdisk_thread", "Same file already exists !! Exiting");
                pthread_exit(NULL);
	    }		
            fraw = fopen(fname, "w");
            if (fraw==NULL) {
                log_error("rawdisk_thread", "Error opening file.");
                pthread_exit(NULL);
            }
        }

        /* See if we need to open next file */
        if (block_count >= blocks_per_file) {
            fclose(fraw);
            filenum++;
            char fname[256];
            sprintf(fname, "%s.%4.4d.raw", pf.basefilename, filenum);
	    sprintf(strlog, "Opening raw file '%s'", fname);
	    log_info("rawdisk_thread", strlog);
            fraw = fopen(fname, "w");
            if (fraw==NULL) {
                log_error("rawdisk_thread", "Error opening file.");
                pthread_exit(NULL);
            }
            block_count=0;
        }

        /* See how full databuf is */
        //total_status = databuf_total_status(db);

        /* If we got packet 0, write data to disk */
        if (got_packet_0) { 

            /* Note waiting status */
            status_lock_safe(&st);
            hputs(st.buf, STATUS, "writing");
            status_unlock_safe(&st);

            /* Write header to file */
            hend = ksearch(ptr, "END");
	    ts.hdlen = gethlength (ptr);
	    clock_gettime(CLOCK_REALTIME, &t_write_start);
            for (ptr=ptr; ptr<=hend; ptr+=80) {
                written = fwrite(ptr, 80, 1, fraw);
            }
	    clock_gettime(CLOCK_REALTIME, &t_write_stop);
	    t_write += (double)timediff(t_write_start, t_write_stop);

            /* Write data */
            ptr = databuf_data(db, curblock);

	    clock_gettime(CLOCK_REALTIME, &t_write_start);
            rv = fwrite(ptr, 1, (size_t)blocksize, fraw);
	    clock_gettime(CLOCK_REALTIME, &t_write_stop);
	    t_write += (double)timediff(t_write_start, t_write_stop);

            if (rv != blocksize) { 
                sprintf(strlog, "Error writing data %lld != %lld  .", rv, blocksize);
                log_error("rawdisk_thread", strlog);
            }

            /* Increment counter */
            block_count++;
	    ts.block_count ++;
	    ts.t_write = t_write;

            /* flush output */
            fflush(fraw);
        }

        /* Mark as free */
        databuf_set_free(db, curblock);

        /* Go to next block */
        curblock = (curblock + 1) % db->n_block;

        /* Check for cancel */
        pthread_testcancel();

    }

    pthread_exit(NULL);

    pthread_cleanup_pop(0); /* Closes fclose */
    pthread_cleanup_pop(0); /* Closes print_stat */
    pthread_cleanup_pop(0); /* Closes databuf_detach */
    pthread_cleanup_pop(0); /* Closes free_psrfits */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes status_detach */

}
