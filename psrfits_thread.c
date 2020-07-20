/* guppi_psrfits_thread.c
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
#include "polyco.h"
#include "psrfits.h"
#include "fold.h"
#include "logging.h"
#include "status.h"
#include "databuf.h"
#include "params.h"
#include "cpu_utils.h"

#define STATUS "DISKSTAT"
#include "threads.h"

// Read a status buffer all of the key observation paramters
//extern void read_status_params(char *buf, obs_params *g, struct psrfits *p);

/* Parse info from buffer into param struct */
//extern void read_subint_params(char *buf, obs_params *g, struct psrfits *p);


/* Downsampling functions */
extern void get_stokes_I(struct psrfits *pf);
extern void downsample_freq(struct psrfits *pf);
extern void downsample_time(struct psrfits *pf);
extern void update_ds_params(struct psrfits *pf);

void zero_end_chans(struct psrfits *pf)
{
    int ii, jj;
    struct hdrinfo *hdr = &(pf->hdr);
    char *data = (char *)pf->sub.data;
    const int nchan = hdr->nchan;
    const int nspec = hdr->nsblk * hdr->npol;
    
    for (ii = 0, jj = 0 ; ii < nspec ; ii++, jj += nchan)
        data[jj] = data[jj+nchan-1] = 0;
}


void psrfits_thread(void *_args) {
    
    // -- Get arguments --
    thread_args *args = (thread_args *)_args;
    pthread_cleanup_push((void *)thread_set_finished, args);
    
    // -- cpu affinity --
    //char strlog[128];
    printf("psrfits_thread: setcpuaffinity(%d);\n",args->output_buffer - 1);
    setcpuaffinity(args->output_buffer - 1);
    
    // -- process priority --
    //setabspriority(args->priority);
    
    // -- Logging
    char strlog[256];
    
    /* Attach to status shared mem area */
    status st;
    int rv;
    rv = status_attach(&st, args->gpu_id);
    if (rv!=OK) {
        log_error("psrfits_thread", "Error attaching to status shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)status_detach, &st);
    pthread_cleanup_push((void *)set_exit_status, &st);
    
    /* Init status */
    status_lock_safe(&st);
    hputs(st.buf, STATUS, "init");
    status_unlock_safe(&st);
    
    /* Initialize some key parameters */
    obs_params gp;
    struct psrfits pf;
    pf.sub.data = NULL;
    pf.sub.dat_freqs = pf.sub.dat_weights = NULL;
    pf.sub.dat_offsets = pf.sub.dat_scales = NULL;
    pf.hdr.chan_dm = 0.0;
    pf.filenum = 0; // This is crucial
    pthread_cleanup_push((void *)free_psrfits, &pf);
    pthread_cleanup_push((void *)psrfits_close, &pf);
    pf.multifile = 0;  // Use a single file for fold mode
    //pf.multifile = 1;  // Use a multiple files for fold mode
    pf.quiet = 0;      // Print a message per each subint written
    
    /* Attach to databuf shared mem */
    databuf *db;
    char strlog[128];
    sprintf(strlog,"databuf_attach(%d)", args->input_buffer);
    log_info("psrfits_thread", strlog);
    db = databuf_attach(args->input_buffer);
    if (db==NULL) {
        log_error("psrfits_thread", "Error attaching to databuf shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)databuf_detach, db);
    
    /* Loop */
    int curblock=0, total_status=0, firsttime=1, run=1, got_packet_0=0;
    int mode=SEARCH_MODE;
    char *ptr;
    char tmpstr[256];
    foldbuf fb;
    polyco pc[64];  
    memset(pc, 0, sizeof(pc));
    int n_polyco_written=0;
    float *fold_output_array = NULL;
    int scan_finished=0;
    signal(SIGINT, cc);
    do {
        /* Note waiting status */
        status_lock_safe(&st);
        if (got_packet_0)
            sprintf(tmpstr, "waiting(%d)", curblock);
        else
            sprintf(tmpstr, "ready");
        hputs(st.buf, STATUS, tmpstr);
        status_unlock_safe(&st);
        
        /* Wait for buf to have data */
        rv = databuf_wait_filled(db, curblock);
        if (rv!=0) {
            // This is a big ol' kludge to avoid this process hanging
            // due to thread synchronization problems.
            sleep(1);
            continue; 
        }
        /* Note current block */
        status_lock_safe(&st);
        hputi4(st.buf, "CURBLOCK", curblock);
        status_unlock_safe(&st);

        /* See how full databuf is */
        total_status = databuf_total_status(db);
        
        /* Read param structs for this block */
        ptr = databuf_header(db, curblock);
        if (firsttime) {
	    printf("psrfits_thread: read_status_params\n");
	    printf("ptr=%s\n",ptr);
            read_status_params(ptr, &gp, &pf);
	    printf("pf.hdr.nsblk=%d\n",pf.hdr.nsblk);
	    printf("pf.hdr.onlyI=%i\n",pf.hdr.onlyI);
            firsttime = 0;
        } else {
            read_subint_params(ptr, &gp, &pf);
        }

        /* Find out what mode this data is in */
        mode = psrfits_obs_mode(pf.hdr.obs_mode);
        /* Check if we got both packet 0 and a valid observation
         * start time.  If so, flag writing to start.
         */
	//printf("pf.:  got_packet_0==%i=0 && gp.packetindex==%i=0 && gp.stt_valid==%i=1\n",got_packet_0,gp.packetindex,gp.stt_valid);
        if (got_packet_0==0 && gp.packetindex==0 && gp.stt_valid==1) {
            log_info("psrfits_thread", "Received start order\n");
            got_packet_0 = 1;
	    printf("psrfits_thread: read_status_params");
	    printf("pf.hdr.nsblk=%d\n",pf.hdr.nsblk);
            read_status_params(ptr, &gp, &pf);
            update_ds_params(&pf);
            memset(pc, 0, sizeof(pc));
            n_polyco_written=0;
        }

	printf("pf.hdr.obs_mode=%s\n",pf.hdr.obs_mode);
	printf("pf.hdr.obs_mode=%i\n",mode);
	printf("pf.FOLD_MODE=%i\n",FOLD_MODE);
	printf("pf.hdr.onlyI=%i\n",pf.hdr.onlyI);
        /* If actual observation has started, write the data */
        if (got_packet_0) { 

            /* Note waiting status */
            status_lock_safe(&st);
            hputs(st.buf, STATUS, "writing");
            hputr8(st.buf, "OBS_LEN", pf.T);
            status_unlock_safe(&st);
            
            /* Get the pointer to the current data */
            //log_info("psrfits_thread: observation mode = ", (char)mode);
            if (mode==FOLD_MODE) {
                fb.nchan = pf.hdr.nchan;
                fb.npol = pf.hdr.npol;
                fb.nbin = pf.hdr.nbin;
                fb.order = pol_bin_chan; // XXX fix this!
                fb.data = (float *)databuf_data(db, curblock);
                fb.count = (unsigned *)(databuf_data(db, curblock)
                        + foldbuf_data_size(&fb));
                fold_output_array = (float *)realloc(fold_output_array,
                        sizeof(float) * pf.hdr.nbin * pf.hdr.nchan * 
                        pf.hdr.npol);
                pf.sub.data = (unsigned char *)fold_output_array;
                pf.fold.pc = (polyco *)(databuf_data(db,curblock)
                        + foldbuf_data_size(&fb) + foldbuf_count_size(&fb));
            } 
	    else 
                pf.sub.data = (unsigned char *)databuf_data(db, curblock);
            
            /* Set the DC and Nyquist channels explicitly to zero */
            /* because of the "FFT Problem" that splits DC power  */
            /* into those two bins.                               */
            // XXX why are we doing this? we should just set weight to 0
            //zero_end_chans(&pf);

            /* Output only Stokes I (in place) */
	    //printf("pf. pf.hdr.onlyI=%i && pf.hdr.npol==4=%i\n",pf.hdr.onlyI,pf.hdr.npol);
	    
            if (pf.hdr.onlyI && pf.hdr.npol==4) {
		get_stokes_I(&pf);
		}
            /* Downsample in frequency (in place) */
            if (pf.hdr.ds_freq_fact > 1)
                downsample_freq(&pf);

            /* Downsample in time (in place) */
            if (pf.hdr.ds_time_fact > 1)
                downsample_time(&pf);

            /* Folded data needs a transpose */
            if (mode==FOLD_MODE)
                normalize_transpose_folds(fold_output_array, &fb);

            /* Write the data */
            int last_filenum = pf.filenum;
	    printf("pf.filename=%s\n",pf.filename);
            psrfits_write_subint(&pf);

            /* Any actions that need to be taken when a new file
             * is created.
             */
            if (pf.filenum!=last_filenum) {
                /* No polycos yet written to the new file */
                n_polyco_written=0;
            }

            /* Write the polycos if needed */
            int write_pc=0, i, j;
            for (i=0; i<pf.fold.n_polyco_sets; i++) {
                if (pf.fold.pc[i].used==0) continue; 
                int new_pc=1;
                for (j=0; j<n_polyco_written; j++) {
                    if (polycos_differ(&pf.fold.pc[i], &pc[j])==0) {
                        new_pc=0;
                        break;
                    }
                }
                if (new_pc || n_polyco_written==0) {
                    pc[n_polyco_written] = pf.fold.pc[i];
                    n_polyco_written++;
                    write_pc=1;
                } else {
                    pf.fold.pc[i].used = 0; // Already have this one
                }
            }
            if (write_pc) 
                psrfits_write_polycos(&pf, pf.fold.pc, pf.fold.n_polyco_sets);

            /* Is the scan complete? */
            if ((pf.hdr.scanlen > 0.0) && 
                (pf.T > pf.hdr.scanlen)) scan_finished = 1;
            
            /* For debugging... */
            if (gp.drop_frac > 0.0) {
               sprintf(strlog, "Block %d dropped %.3g%% of the packets", pf.tot_rows, gp.drop_frac*100.0);
	       log_info("psrfits_thread", strlog);
            }

        }

        /* Clear the buffer */
        //memset(databuf_data(db, curblock), 0, db->block_size);

        /* Mark as free */
        databuf_set_free(db, curblock);
        
        /* Go to next block */
        curblock = (curblock + 1) % db->n_block;
        
        /* Check for cancel */
        pthread_testcancel();
        
    } while (run && !scan_finished);
    
    /* Cleanup */
    
    if (fold_output_array!=NULL) free(fold_output_array);


    printf("Closing psrfits_thread\n");
    log_info("psrfits_thread", "Closing psrfits_thread");
    pthread_exit(NULL);
    
    pthread_cleanup_pop(0); /* Closes psrfits_close */
    pthread_cleanup_pop(0); /* Closes free_psrfits */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes set_finished */
    pthread_cleanup_pop(0); /* Closes status_detach */
    pthread_cleanup_pop(0); /* Closes databuf_detach */
}
