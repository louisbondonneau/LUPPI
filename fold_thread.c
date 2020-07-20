/* fold_thread.c
 *
 * Fold data, etc.
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
#include "polyco.h"
#include "fold.h"
#include "cpu_utils.h"
#include "params.h"

#define STATUS "FOLDSTAT"
#include "threads.h"

// Read a status buffer all of the key observation paramters
//extern void guppi_read_obs_params(char *buf, params *g, struct psrfits *p);

/* Parse info from buffer into param struct */
//extern void guppi_read_subint_params(char *buf, params *g, struct psrfits *p);


static const int nthread = 6;
static void join_all_threads(pthread_t *ids) {
    int i;
    for (i=0; i<nthread; i++)
        if (ids[i]!=0)
            pthread_join(ids[i],NULL);
}


void fold_thread(void *_args) {

    // -- Get arguments -- 
    thread_args *args = (thread_args *)_args;

    // -- cpu affinity --
    printf("fold_thread: setcpuaffinity(%d);\n",args->output_buffer - 1);
    setcpuaffinity(args->output_buffer - 1);
    
    // -- process priority --
    setabspriority(args->priority);

    /* Attach to status shared mem area */
    int rv;
    status st;
    rv = status_attach(&st, args->gpu_id);
    if (rv!=OK) {
        log_error("fold_thread", "Error attaching to status shared memory.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)status_detach, &st);
    pthread_cleanup_push((void *)set_exit_status, &st);
    pthread_cleanup_push((void *)thread_set_finished, args);

    /* Init status */
    status_lock_safe(&st);
    hputs(st.buf, STATUS, "init");
    status_unlock_safe(&st);

    /* Read in general parameters */
    obs_params gp;
    struct psrfits pf;
    pf.sub.dat_freqs = NULL;
    pf.sub.dat_weights = NULL;
    pf.sub.dat_offsets = NULL;
    pf.sub.dat_scales = NULL;
    pthread_cleanup_push((void *)free_psrfits, &pf);

    /* Attach to databuf shared mem */
    databuf *db_in, *db_out;
    db_in = databuf_attach(args->input_buffer);
    char errmsg[256];
    if (db_in==NULL) {
        sprintf(errmsg, "Error attaching to input databuf(%d) shared memory.", args->input_buffer);
        log_error("fold_thread", errmsg);
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)databuf_detach, db_in);
    db_out = databuf_attach(args->output_buffer);
    if (db_out==NULL) {
        sprintf(errmsg, "Error attaching to output databuf(%d) shared memory.", args->output_buffer);
        log_error("fold_thread", errmsg);
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)databuf_detach, db_out);

    /* Load polycos */
    int imjd;
    double fmjd, fmjd0, fmjd_next=0.0;
    int npc=0, ipc;
    polyco *pc=NULL;
    FILE *polyco_file=NULL;

    /* Total foldbuf */
    foldbuf fb;
    fb.nbin = 0;
    fb.nchan = 0;
    fb.npol = 0;
    fb.order = chan_pol_bin;
    fb.data = NULL;
    fb.count = NULL;

    /* Sub-thread management */
    pthread_t thread_id[nthread];
    int input_block_list[nthread];
    fold_args fargs[nthread];
    int i;
    for (i=0; i<nthread; i++) {
        thread_id[i] = 0;
        input_block_list[i] = -1;
        fargs[i].data = NULL;
        fargs[i].fb = (foldbuf *)malloc(sizeof(foldbuf));
        fargs[i].fb->nbin = fb.nbin;
        fargs[i].fb->nchan = 0;
        fargs[i].fb->npol = 0;
        fargs[i].fb->order = fb.order;
        fargs[i].fb->data = NULL;
        fargs[i].fb->count = NULL;
        fargs[i].nsamp = 0;
        fargs[i].tsamp = 0.0;
        fargs[i].raw_signed = 1;
    }
    pthread_cleanup_push((void *)join_all_threads, thread_id);

    /* Loop */
    int curblock_in=0, curblock_out=0;
    int refresh_polycos=1, next_integration=0, first=1, reset_foldbufs=1;
    int nblock_int=0, npacket=0, ndrop=0;
    double tsubint=0.0, offset=0.0, suboffs=0.0;
    int cur_thread=0;
    char *hdr_in=NULL, *hdr_out=NULL;
    signal(SIGINT,cc);
    while (run) {

        /* Note waiting status */
        status_lock_safe(&st);
        hputs(st.buf, STATUS, "waiting");
        status_unlock_safe(&st);

        /* Wait for buf to have data */
        rv = databuf_wait_filled(db_in, curblock_in);
        if (rv!=0) continue;

        /* Note current block(s) */
        status_lock_safe(&st);
        hputi4(st.buf, "CURBLOCK", curblock_in);
        hputi4(st.buf, "CURFOLD", curblock_out);
        status_unlock_safe(&st);

        /* Note waiting status */
        status_lock_safe(&st);
        hputs(st.buf, STATUS, "folding");
        status_unlock_safe(&st);

        /* Read param struct for this block */
        hdr_in = databuf_header(db_in, curblock_in);
        if (first) { 
            read_status_params(hdr_in, &gp, &pf);
	}    
        else
            read_subint_params(hdr_in, &gp, &pf);

        /* Refresh params, dump any previous subint on a 0 packet */
        if (gp.packetindex==0)  {
	    // TODO : boucler sur le nombre de gpu et 
            // TODO read_status_params(hdr_in, &gp, &pf[i]);
            read_status_params(hdr_in, &gp, &pf);
            if (!first) next_integration=1; 
            refresh_polycos=1;
            reset_foldbufs=1;
        }

        /* Figure out what time it is */
        offset = pf.hdr.dt * gp.packetindex * gp.packetsize 
            / pf.hdr.nchan / pf.hdr.npol; // Only true for 8-bit data
        imjd = pf.hdr.start_day;
        fmjd = (pf.hdr.start_sec + offset) / 86400.0;

        /* Do any first time stuff */
        if (first) {

            /* Set mjds */
            fmjd0 = fmjd;
            fmjd_next = fmjd0 + pf.fold.tfold/86400.0;

            /* Set nbin, nchan, npol */
            fb.nbin = pf.fold.nbin;
            fb.nchan = pf.hdr.nchan;
            fb.npol = pf.hdr.npol;

            /* Set up first output header */
            hdr_out = databuf_header(db_out, curblock_out);
            memcpy(hdr_out, databuf_header(db_in, curblock_in), STATUS_SIZE);
            hputi4(hdr_out, "NBIN", fb.nbin);
            if (strncmp(pf.hdr.obs_mode,"CAL",3))
                hputs(hdr_out, "OBS_MODE", "PSR");

            /* Set up output data ptrs */
            fb.data = (float *)databuf_data(db_out, curblock_out);
            fb.count = (unsigned *)((char *)fb.data + foldbuf_data_size(&fb));

            /* Check that output databuf has enough space to hold
             * fold data, fold counts, and 2 polyco structs.
             */
            size_t total_output_size = foldbuf_data_size(&fb) + foldbuf_count_size(&fb) + 2*sizeof(polyco);
            if (total_output_size > db_out->block_size) {
                log_error("fold_thread", "Insufficient memory per block to hold fold results.");
                pthread_exit(NULL);
            }

            /* Clear output data buffer */
            clear_foldbuf(&fb);

            fprintf(stderr, "nbin=%d nchan=%d npol=%d tfold=%f\n", fb.nbin, fb.nchan, fb.npol, pf.fold.tfold);

            first=0;
        }

        /* Check if we need to move to next subint */
        if (fmjd>fmjd_next) { next_integration=1; }

        /* Combine thread results if needed */
        if (cur_thread==nthread || next_integration) {

            /* Loop over active threads */
            for (i=0; i<cur_thread; i++) {

                /* Wait for thread */
                int *thread_rv;
                rv = pthread_join(thread_id[i], (void**)&thread_rv);
                if (rv) {
                    log_error("fold_thread", "Error joining subthread");
                    continue;
                }
                if (*thread_rv!=0) {
                    fprintf(stderr, "fold_thread returned %d\n", *thread_rv);
                }

                /* Mark input block as free */
                if (input_block_list[i]>=0) 
                    databuf_set_free(db_in, input_block_list[i]);
                
                /* Combine result into total int */
                rv = accumulate_folds(&fb, fargs[i].fb);
                if (rv!=0) 
                    fprintf(stderr, "accumulate_folds returned %d\n",rv);

                /* Reset thread info */
                clear_foldbuf(fargs[i].fb);
                thread_id[i] = 0;
                input_block_list[i] = -1;
            }

            /* Reset thread count */
            cur_thread = 0;
        }

        /* Reset / reallocate per-thread fold buffer memory */
        if (reset_foldbufs) {

            /* Set output fold params */
            fb.nbin = pf.fold.nbin;
            fb.nchan = pf.hdr.nchan;
            fb.npol = pf.hdr.npol;

            /* Loop over thread foldbufs */
            for (i=0; i<nthread; i++) {
                fargs[i].fb->nbin = fb.nbin;
                fargs[i].fb->nchan = pf.hdr.nchan;
                fargs[i].fb->npol = pf.hdr.npol;
                if (fargs[i].fb->data!=NULL) free_foldbuf(fargs[i].fb);
                malloc_foldbuf(fargs[i].fb);
                clear_foldbuf(fargs[i].fb);
            }

            reset_foldbufs=0;
        }

        /* Finalize this output block if needed, move to next */
        if (next_integration) {

            /* Add polyco info to current output block */
            int n_polyco_used = 0;
            polyco *pc_ptr = (polyco *)(databuf_data(db_out, curblock_out)
                        + foldbuf_data_size(&fb) + foldbuf_count_size(&fb));
            for (i=0; i<npc; i++) { 
                if (pc[i].used) { 
                    n_polyco_used += 1;
                    *pc_ptr = pc[i];
                    pc_ptr++;
                }
            }
            hputi4(hdr_out, "NPOLYCO", n_polyco_used);

            /* Close out current integration */
            databuf_set_filled(db_out, curblock_out);

            /* Set up params for next int */
            fmjd0 = fmjd;
            fmjd_next = fmjd0 + pf.fold.tfold/86400.0;
            fb.nchan = pf.hdr.nchan;
            fb.npol = pf.hdr.npol;

            /* Wait for next output buf */
            curblock_out = (curblock_out + 1) % db_out->n_block;
            databuf_wait_free(db_out, curblock_out);
            hdr_out = databuf_header(db_out, curblock_out);
            memcpy(hdr_out, databuf_header(db_in, curblock_in), STATUS_SIZE);
            if (strncmp(pf.hdr.obs_mode,"CAL",3))
                hputs(hdr_out, "OBS_MODE", "PSR");
            hputi4(hdr_out, "NBIN", fb.nbin);
            hputi4(hdr_out, "PKTIDX", gp.packetindex);

            fb.data = (float *)databuf_data(db_out, curblock_out);
            fb.count = (unsigned *)((char *)fb.data + foldbuf_data_size(&fb));
            clear_foldbuf(&fb);

            nblock_int=0;
            npacket=0;
            ndrop=0;
            tsubint=0.0;
            suboffs=0.0;
            next_integration=0;
        }

        /* Check src, get correct polycos */
        if (refresh_polycos) { 
            // Auto polyco making:
            // 1. if mode==cal, generate const-freq polyco
            // 2. if mode==psr and PARFILE is set, generate polycos
            // 3. if mode==psr and no PARFILE, try reading polyco.dat
            if (strncmp(pf.hdr.obs_mode,"CAL",3)==0) {
                // Cal mode
                npc = make_const_polyco(pf.hdr.cal_freq, &pf.hdr, &pc);
            } else {
                // Psr mode
                if (pf.fold.parfile[0]=='\0') {
                    // Try reading polyco.dat
                    fprintf(stderr, "Reading polyco.dat\n"); // DEBUG
                    polyco_file = fopen("polyco.dat", "r");
                    if (polyco_file==NULL) { 
                        log_error("fold_thread", "Couldn't open polyco.dat");
                        pthread_exit(NULL);
                    }
                    npc = read_all_pc(polyco_file, &pc, pf.hdr.source);
                    if (npc==0) { 
                        log_error("fold_thread", "Error parsing polyco file.");
                        pthread_exit(NULL);
                    }
                    fclose(polyco_file);
                } else {
                    // Try calling tempo
                    fprintf(stderr, "Calling tempo on %s\n", pf.fold.parfile); // DEBUG
                    npc = make_polycos(pf.fold.parfile, &pf.hdr, NULL, &pc);
                    if (npc<=0) {
                        log_error("fold_thread", "Error generating polycos.");
                        pthread_exit(NULL);
                    }
                }
                fprintf(stderr, "Read %d polycos (%.3f->%.3f)\n", 
                        npc, (double)pc[0].mjd + pc[0].fmjd, 
                        (double)pc[npc-1].mjd + pc[npc-1].fmjd);
            }

            refresh_polycos=0;
        }

        /* Select polyco set */
        if (strncmp(pf.hdr.obs_mode,"CAL",3)) {
            // PSR mode, select appropriate block
            ipc = select_pc(pc, npc, NULL, imjd, fmjd);
            if (ipc<0) { 
                sprintf(errmsg, 
                        "No matching polycos "
                        "(npc=%d, src=%s, imjd=%d, fmjd=%f)",
                        npc, pf.hdr.source, imjd, fmjd);
                log_error("fold_thread", errmsg);
                pthread_exit(NULL);
            }
        } else {
            // CAL mode, use the (only) const-polyco block
            ipc = 0;
        }
        pc[ipc].used = 1;

        /* Launch fold thread */
        input_block_list[cur_thread] = curblock_in;
        fargs[cur_thread].data = databuf_data(db_in, curblock_in);
        fargs[cur_thread].pc = &pc[ipc];
        fargs[cur_thread].imjd = imjd;
        fargs[cur_thread].fmjd = fmjd;
        fargs[cur_thread].fb->nbin = fb.nbin;
        fargs[cur_thread].fb->nchan = pf.hdr.nchan;
        fargs[cur_thread].fb->npol = pf.hdr.npol;
        fargs[cur_thread].nsamp = gp.n_packets*gp.packetsize 
            / pf.hdr.nchan / pf.hdr.npol;
        fargs[cur_thread].tsamp = pf.hdr.dt;
        fargs[cur_thread].raw_signed = 1;
        rv = pthread_create(&thread_id[cur_thread], NULL, 
                fold_8bit_power_thread, &fargs[cur_thread]);
        if (rv!=0) 
            log_error("fold_thread", "error launching fold subthread");
        else  
            cur_thread++;

        nblock_int++;
        npacket += gp.n_packets;
        ndrop += gp.n_dropped;
        tsubint = pf.hdr.dt * (npacket - ndrop) * gp.packetsize 
            / pf.hdr.nchan / pf.hdr.npol; // Only true for 8-bit data
        suboffs += offset;
        hputi4(hdr_out, "NBLOCK", nblock_int);
        hputi4(hdr_out, "NPKT", npacket);
        hputi4(hdr_out, "NDROP", ndrop);
        hputr8(hdr_out, "TSUBINT", tsubint);
        hputr8(hdr_out, "OFFS_SUB", suboffs / (double)nblock_int);

        /* Mark in as free.. not yet! */
        //databuf_set_free(db_in, curblock_in);

        /* Go to next input block */
        curblock_in = (curblock_in + 1) % db_in->n_block;

        /* Check for cancel */
        pthread_testcancel();

    }

    pthread_exit(NULL);

    pthread_cleanup_pop(0); /* Closes join_all_threads */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
    pthread_cleanup_pop(0); /* Closes set_finished */
    pthread_cleanup_pop(0); /* Closes free_psrfits */
    pthread_cleanup_pop(0); /* Closes status_detach */
    pthread_cleanup_pop(0); /* Closes databuf_detach */
    pthread_cleanup_pop(0); /* Closes databuf_detach */

}
