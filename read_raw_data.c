#include <stdio.h>
#include <stdlib.h>

#include "fitshead.h"
#include "logging.h"
#include "status.h"
#include "databuf.h"
#include "cpu_utils.h"

#define STATUS "RRAWSTAT"
#include "threads.h"


// -- Signal handlers for run control --
void handle_sigint(int sig) { run=0; }
void handle_sigusr1(int sig) { run=1; }


void readraw_thread(void *_args) {

    // -- Get arguments --
    thread_args *args = (thread_args *)_args;

    char strlog[128];

    // -- Install signal handlers --
    signal(SIGINT, cc);

    // -- cpu affinity --
    printf("readraw_thread: setcpuaffinity(%d);\n",args->output_buffer - 1);
    setcpuaffinity(args->output_buffer - 1);
    
    // -- process priority --
    setabspriority(args->priority);

    // -- Attach to status shm --
    int rv;
    status st;
    rv = status_attach(&st, args->gpu_id);
    if(rv!=OK) {
        log_error("net_thread","Error attaching to status shm");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)status_detach, &st);
    pthread_cleanup_push((void *)set_exit_status, &st);
    log_info("net_thread","Attached to status shm OK");

    //  -- Read params from status shm and put it into temp status_buf --
    obs_params gp;
    struct psrfits pf;
    pf.sub.dat_freqs = NULL;
    pf.sub.dat_weights = NULL;
    pf.sub.dat_offsets = NULL;
    pf.sub.dat_scales = NULL;

    /* Need this ? */
    char status_buf[STATUS_SIZE];
    status_lock_safe(&st);
    memcpy(status_buf,st.buf,STATUS_SIZE);
    status_unlock_safe(&st);

    /*  Don't need the following --
    rv = read_status_params(status_buf, &gp, &pf);
    if(rv!=OK) {
        log_error("net_thread","Error in read_status_params");
        exit(-1);
    }
    log_info("net_thread","read_status_params completed");
    */


    // -- Init databuf struct --
    databuf *db=NULL;

    // -- Attach to databuf shared memory --
    db = databuf_attach(args->output_buffer);
    if(db == NULL) {
        sprintf(strlog,"Error attaching to data shm");
        log_error("net_thread",strlog );
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)databuf_detach, db);


    // -- List of datablock_stats currently in use --
    const int nblock = 2;
    datablock_stats blocks[nblock];
    for (i=0; i<nblock; i++)
        init_block(&blocks[i], db, packet_data_size, packets_per_block, overlap_packets);

    // -- Convenience names for first/last blocks in set --
    datablock_stats *fblock, *lblock;
    fblock = &blocks[0];
    lblock = &blocks[nblock-1];

  

    char *hdr_buffer;
    FILE *pfi;


    while(run) {

        if (first) {

	    pfi = fopen(fname, "r");
	    fread(hdr_buffer, HDR_SIZE, 1, pfi);
	    gethlength(hdr_buffer);

              // -- Reset buffers -- 
              for (i=0; i<nblock-1; i++) {
                  //if (blocks[i].block_idx>=0) finalize_block(&blocks[i]);
                  reset_block(&blocks[i]);
              }


        }



          /* Wait for new block to be free, then clear it
           * if necessary and fill its header with new values.
           */
          while ((rv=databuf_wait_free(db, lblock->block_idx)) != OK) {
              if (rv==TIMEOUT) {
                  waiting=1;
                  status_lock_safe(&st);
                  hputs(st.buf, STATUS, "blocked");
                  status_unlock_safe(&st);
                  continue;
              } else {
                  log_error("net_thread", "error waiting for free databuf");
                  run=0;
                  pthread_exit(NULL);
                  break;
              }
          }





	// -- If end of file, try to open the next file. If failed, exit --
	if (feof(pfi)) {
	    fclose(pfi);
	    sprintf(fname, "");
	    pfi = fopen(fname, "r");
	    if (pfi==NULL) {
		printf("Failed to open next file : %s. Exiting\n", fname);
		exit(0);
	    }
	}

	// -- Send header to the shm memory --
	

	// -- Send databuf to the databuf shm --



        // -- Will exit if thread has been cancelled -- 
        pthread_testcancel();


    }

    printf("-- read_RAW_data  RUN=0 --\n");fflush(stdout);
    pthread_exit(NULL);

    // -- Have to close all push's -- 
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes status_detach */
    pthread_cleanup_pop(0); /* Closes databuf_detach */


}

