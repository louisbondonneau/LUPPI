/* null_thread.c
 *
 * Marks databufs empty as soon as they're full
 */

#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
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

#define STATUS "NULLSTAT"
#include "threads.h"

void null_thread(void *_args) {

    int rv;
    /* Get args */
    thread_args *args = (thread_args *)_args;

    /* Attach to status shared mem area */
    status st;
    rv = status_attach(&st, args->gpu_id);
    if (rv!=OK) {
        log_error("null_thread", "Error attaching to status shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)set_exit_status, &st);

    /* Init status */
    status_lock_safe(&st);
    hputs(st.buf, STATUS, "init");
    status_unlock_safe(&st);

    /* Attach to databuf shared mem */
    databuf *db;
    db = databuf_attach(args->input_buffer);
    if (db==NULL) {
        log_error("null_thread",
                "Error attaching to databuf shared memory.");
        pthread_exit(NULL);
    }

    /* Loop */
    char *ptr;
    obs_params gp;
    struct psrfits pf;
    pf.sub.dat_freqs = NULL;
    pf.sub.dat_weights = NULL;
    pf.sub.dat_offsets = NULL;
    pf.sub.dat_scales = NULL;
    int curblock=0;
    signal(SIGINT,cc);
    while (run) {

        /* Note waiting status */
        status_lock_safe(&st);
        hputs(st.buf, STATUS, "waiting");
        status_unlock_safe(&st);

        /* Wait for buf to have data */
        rv = databuf_wait_filled(db, curblock);
        if (rv!=0) {
            //sleep(1);
            continue;
        }

        /* Note waiting status, current block */
        status_lock_safe(&st);
        hputs(st.buf, STATUS, "blanking");
        hputi4(st.buf, "CURBLOCK", curblock);
        status_unlock_safe(&st);

        /* Get params */
        ptr = databuf_header(db, curblock);
        read_status_params(ptr, &gp, &pf);

        /* Output if data was lost */
        if (gp.n_dropped!=0 && (gp.packetindex==0 || strcmp(pf.hdr.obs_mode,"SEARCH"))) {
            printf("Block beginning with pktidx=%lld dropped %d packets\n", gp.packetindex, gp.n_dropped);
            fflush(stdout);
        }

        /* Mark as free */
        databuf_set_free(db, curblock);

        /* Go to next block */
        curblock = (curblock + 1) % db->n_block;

        /* Check for cancel */
        pthread_testcancel();

    }

    pthread_exit(NULL);

    pthread_cleanup_pop(0); /* Closes set_exit_status */

}
