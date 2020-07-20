/* dedisp_thread.c
 *
 * Dedisperse incoming baseband data
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

#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <omp.h>

#include "fitshead.h"
#include "psrfits.h"
#include "polyco.h"
#include "logging.h"
#include "status.h"
#include "databuf.h"
#include "params.h"
#include "cpu_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "dedisperse_gpu.h"
#include "dedisperse_utils.h"
#include "fold.h"
#include "fold_gpu.h"

#define STATUS "DISPSTAT"
#include "threads.h"

/* Parse info from buffer into param struct */
//extern void read_subint_params(char *buf, params *g, struct psrfits *p);
//extern void guppi_read_obs_params(char *buf, params *g, struct psrfits *p);



void dedisp_thread(void *_args) {

    // -- Get arguments --
    thread_args *args = (thread_args *)_args;

    // -- Select GPU --
    cudaSetDevice(args->gpu_id%2);

    int rv;

    // -- cpu affinity --
    printf("dedisp_thread: setcpuaffinity(%d);\n",args->output_buffer - 1);
    setcpuaffinity(args->output_buffer - 1);

    // -- process priority --
    //setabspriority(args->priority);

    /* Attach to status shared mem area */
    status st;
    rv = status_attach(&st, args->gpu_id);
    /*
    if (rv!=OK) {
        log_error("dedisp_thread", "Error attaching to status shared memory.");
        pthread_exit(NULL);
    }
    */
    pthread_cleanup_push((void *)set_exit_status, &st);

    /* Init status */
    status_lock_safe(&st);
    hputs(st.buf, STATUS, "init");
    status_unlock_safe(&st);

    /* Attach to databuf shared mem */
    databuf *db_in, *db_out;
    db_in = databuf_attach(args->input_buffer);
    if (db_in==NULL) {
        char msg[256];
        sprintf(msg, "Error attaching to databuf(0x%08x) shared memory.", args->input_buffer);
        log_error("dedisp_thread", msg);
        pthread_exit(NULL);
    }
    db_out = databuf_attach(args->output_buffer);
    if (db_out==NULL) {
        char msg[256];
        sprintf(msg, "Error attaching to databuf(0x%08x) shared memory.", args->output_buffer);
        log_error("dedisp_thread", msg);
        pthread_exit(NULL);
    }

    foldbuf fb, fb_tot;
    fb.nbin = fb_tot.nbin = 0;
    fb.nchan = fb_tot.nchan = 0;
    fb.npol = fb_tot.npol = 0;
    fb.order = fb_tot.order = pol_bin_chan;
    fb.data = fb_tot.data = NULL;
    fb.count = fb_tot.count = NULL;

    /* Loop */
    char *hdr_in=NULL, *hdr_out=NULL;
    char strlog[256];
    char msg[256];
    obs_params gp;
    struct psrfits pf;
    pf.sub.dat_freqs = pf.sub.dat_weights = pf.sub.dat_offsets = pf.sub.dat_scales = NULL;
    dedispersion_setup ds;
    pthread_cleanup_push((void *)free_dedispersion, &ds);
    pthread_cleanup_push((void *)print_timing_report, &ds);
    int curblock_in=0, curblock_out=0, got_packet_0=0;
    unsigned char *rawdata=NULL;
    float *outdata=NULL;
    int imjd;
    double fmjd0, fmjd_next=0.0, fmjd, offset;
    polyco *pc=NULL;
    int npc=0, ipc;
    int first=1, next_integration=0, refresh_polycos=1;
    int nblock_int=0;
    int64_t npacket=0, ndrop=0, overlap_packets;
    double tsubint=0.0, suboffs=0.0;
    signal(SIGINT,cc);
    while (run) {

        /* Note waiting status */
        status_lock_safe(&st);
        hputs(st.buf, STATUS, "waiting");
        status_unlock_safe(&st);

        /* Wait for buf to have data */
        
        
        //if ((rv=databuf_wait_filled(db_in, curblock_in)) != OK) {
        //      if (rv==TIMEOUT) {
        //          printf("Wait on block %u to be filled\n",curblock_in);
        //          log_info("dedisp_thread", "Wait for new block to be filled");
        //      }
        //}
        
        rv = databuf_wait_filled(db_in, curblock_in);


        if (rv!=0) continue;

        /* Note waiting status, current block */
        status_lock_safe(&st);
        hputs(st.buf, STATUS, "processing");
        hputi4(st.buf, "CURBLOCK", curblock_in);
        status_unlock_safe(&st);

        /* Get params */
        hdr_in = databuf_header(db_in, curblock_in);
        if (first) {
            read_status_params(hdr_in, &gp, &pf);
	}    
        else 
            read_subint_params(hdr_in, &gp, &pf);
	printf("pf.  input dedisp_thread:  pf.hdr.onlyI=%i\n",pf.hdr.onlyI);

        /* Check to see if a new obs started */
        if (gp.packetindex==0) {
            got_packet_0=1;
            read_status_params(hdr_in, &gp, &pf);
            if (!first) next_integration=1;
            refresh_polycos=1;
        }

        /* Get current time */
        //offset = pf.hdr.dt * gp.packetindex * gp.packetsize 
        //    / pf.hdr.nchan / pf.hdr.npol; // Only true for 8-bit data
        const size_t bytes_per_samp = 4;
        offset = pf.hdr.dt * gp.packetindex * gp.packetsize / bytes_per_samp / pf.hdr.nchan;
        imjd = pf.hdr.start_day;
        fmjd = (pf.hdr.start_sec + offset) / 86400.0;
	printf("imjd = %d   fmjd = %lf  offset = %lf \n", imjd, fmjd, offset);

        /* Refresh polycos if needed */
        if (refresh_polycos) {
            if (strncmp(pf.hdr.obs_mode,"CAL",3)==0 || strncmp(pf.hdr.obs_mode,"FON",3)==0 || strncmp(pf.hdr.obs_mode,"FOF",3)==0) {
                npc = make_const_polyco(pf.hdr.cal_freq, &pf.hdr, &pc);
            } else {
                if (pf.fold.parfile[0]=='\0') {
                    /* If no parfile, read polyco.dat */
		    char polycofile[128];
		    sprintf(polycofile, "polyco.dat-%d", (int) pf.hdr.fctr);
                    FILE *pcf = fopen(polycofile, "r");
                    if (pcf==NULL) { 
                        printf("Couldn't open %s\n", polycofile);
			sprintf(strlog, "Couldn't open %s", polycofile);
                        log_error("dedisp_thread", strlog);
                        pthread_exit(NULL);
                    }
                    npc = read_all_pc(pcf, &pc, pf.hdr.source);
                    if (npc==0) {
                        printf("Error parsing %s\n", polycofile);
			sprintf(strlog, "Error parsing %s", polycofile);
                        log_error("dedisp_thread", strlog);
                        pthread_exit(NULL);
                    }
                    fclose(pcf);
                } else {
                    /* Generate from parfile */
                    //fprintf(stderr, "Calling tempo on %s\n", pf.fold.parfile);

                    printf("befor make_polycos:\n");
                    npc = make_polycos(pf.fold.parfile, &pf.hdr, NULL, &pc);
		    printf("dedisp_thread: make_polyco %s",pf.fold.parfile);
                    if (npc<=0) {
                        log_error("fold_thread", "Error generating polycos.");
                        pthread_exit(NULL);
                    }
                }
                //fprintf(stderr, "Read %d polycos (%.3f->%.3f)\n",
                //        npc, (double)pc[0].mjd + pc[0].fmjd,
                //        (double)pc[npc-1].mjd + pc[npc-1].fmjd);
            }
            refresh_polycos=0;
        }

        /* Any first-time init stuff */
        if (first) {
            fmjd0 = fmjd;
            fmjd_next = fmjd0 + pf.fold.tfold/86400.0;

            /* Fold params */
            fb.nbin = fb_tot.nbin = pf.fold.nbin;
            fb.nchan = fb_tot.nchan = pf.hdr.nchan;
            fb.npol = fb_tot.npol = pf.hdr.npol;

            /* Output data buf */
            hdr_out = databuf_header(db_out, curblock_out);
            memcpy(hdr_out, databuf_header(db_in, curblock_in), STATUS_SIZE);
            hputi4(hdr_out, "NBIN", fb.nbin);
            if (strncmp(pf.hdr.obs_mode,"CAL",3) && strncmp(pf.hdr.obs_mode,"FON",3) && strncmp(pf.hdr.obs_mode,"FOF",3) ) 
                hputs(hdr_out, "OBS_MODE", "PSR");

            /* Output fold bufs */
            fb_tot.data = (float *)databuf_data(db_out, curblock_out);
            fb_tot.count = (unsigned *)((char *)fb_tot.data  + foldbuf_data_size(&fb_tot));
            clear_foldbuf(&fb_tot);

            /* Fill in some dedispersion params */
            ds.rf = pf.hdr.fctr;
            ds.bw = pf.hdr.df;
            ds.fft_len = pf.dedisp.fft_len;
            ds.overlap = pf.dedisp.overlap;
            printf("debug  pf.hdr.nsblk = %lld\n", (int64_t)pf.hdr.nsblk);
            ds.npts_per_block = (int64_t)pf.hdr.nsblk;
            ds.nbins_fold = pf.fold.nbin;
            ds.gp = &gp;

            /* Set up freqs */
            int i;
            ds.nchan = pf.hdr.nchan;
            for (i=0; i<ds.nchan; i++) {
                ds.freq[i] = ds.rf - pf.hdr.BW/2.0 + ((double)i+0.5)*pf.hdr.df;
		printf("Freq #%02d : %f\n", i, ds.freq[i]);
	    }		

            /* Buffers to transfer fold results */
            const int npol = 4;
            const size_t foldbuf_size = sizeof(float) * npol * ds.nbins_fold;
            const size_t foldcount_size = sizeof(unsigned) * ds.nbins_fold;

            cudaMallocHost((void**)&fb.data, foldbuf_size * fb.nchan);

            cudaMallocHost((void**)&fb.count, foldcount_size * fb.nchan);

            /* Init dedispersion on GPU */
            /* TODO what to do about dm or earthz4 changing... */
            ds.dm = pc[0].dm;
            ds.earth_z4 = pc[0].earthz4;
            ds.rm = pf.hdr.rm;
            ds.rmfac = ds.rm*89875.51787368176/(1.0+ds.earth_z4/1.0e4);
            printf("RM = %f\n" , ds.rm);
            printf("rmfac =%f\n" , ds.rmfac);fflush(stdout);

            log_info("dedisp_thread", "Init dedispersion");
            init_dedispersion(&ds);

            /* Init folding */
            log_info("dedisp_thread", "Init folding");
            init_fold(&ds);

            /* Clear first time flag */
            first=0;
        }

        /* Check to see if integration is done */
        if (fmjd>fmjd_next) { next_integration=1; }

        /* Finalize this fold if it's time */
        if (next_integration) {
	    //log_info("dedisp_thread","Next integration");

            /* Add polyco info to current output block */
            int n_polyco_used = 0;
            polyco *pc_ptr = (polyco *)(databuf_data(db_out, curblock_out) + foldbuf_data_size(&fb_tot) + foldbuf_count_size(&fb_tot));
            int i;
            for (i=0; i<npc; i++) { 
		//printf("pc[%d].used=%d\n",i,pc[i].used);
                if (pc[i].used) { 
                    n_polyco_used += 1;
                    *pc_ptr = pc[i];
                    pc_ptr++;
                }
            }
            hputi4(hdr_out, "NPOLYCO", n_polyco_used);
	    sprintf(strlog,"n_polyco_used= %d",n_polyco_used);
	    log_info("dedisp_thread", strlog);

            /* Close out current integration */
            databuf_set_filled(db_out, curblock_out);

            /* Set up params for next int */
            fmjd0 = fmjd;
            fmjd_next = fmjd0 + pf.fold.tfold/86400.0;
            fb.nchan = fb_tot.nchan = pf.hdr.nchan;
            fb.npol = fb_tot.npol = pf.hdr.npol;

            /* Wait for next output buf */
            curblock_out = (curblock_out + 1) % db_out->n_block;

            
            if ((rv=databuf_wait_free(db_out, curblock_out)) != OK) {
                  if (rv==TIMEOUT) {
                      printf("Wait on output %u to be free\n",curblock_in);
                      log_info("dedisp_thread", "Wait for new output block to be free");
                  }
            }

            databuf_wait_free(db_out, curblock_out);
            hdr_out = databuf_header(db_out, curblock_out);
            memcpy(hdr_out, databuf_header(db_in, curblock_in), STATUS_SIZE);
            if (strncmp(pf.hdr.obs_mode,"CAL",3) && strncmp(pf.hdr.obs_mode,"FON",3) && strncmp(pf.hdr.obs_mode,"FOF",3) ) 
                hputs(hdr_out, "OBS_MODE", "PSR");
            hputi4(hdr_out, "NBIN", fb.nbin);
            hputi4(hdr_out, "PKTIDX", gp.packetindex);

            fb_tot.data = (float *)databuf_data(db_out, curblock_out);
            fb_tot.count = (unsigned *)((char *)fb_tot.data + foldbuf_data_size(&fb_tot));
            clear_foldbuf(&fb_tot);

            nblock_int=0;
            npacket=0;
            ndrop=0;
            tsubint=0.0;
            suboffs=0.0;
            next_integration=0;
        }

        /* Set current time, evaluate polycos */
        ds.imjd = imjd;
        ds.fmjd = fmjd;
        ipc = select_pc(pc, npc, NULL, imjd, fmjd);
        if (ipc<0) {
            
            sprintf(msg, "No matching polycos (npc=%d, src=%s, imjd=%d, fmjd=%f)", npc, pf.hdr.source, imjd, fmjd);
            log_error("dedisp_thread", msg);
            pthread_exit(NULL);
        }
        compute_fold_params(&ds, &pc[ipc]);
        pc[ipc].used = 1;

        /* Loop over channels in the block */
        unsigned ichan;
        for (ichan=0; ichan<ds.nchan; ichan++) {

            /* Pointer to raw data
             * 4 bytes per sample for 8-bit/2-pol/complex data
             */
            rawdata = (unsigned char *)databuf_data(db_in, curblock_in) + (size_t)4 * pf.hdr.nsblk * ichan;

	    /*
	    int j;
            printf("Chan = %d\n", ichan);
            for (j=0; j<2048; j++)
	        printf("%d %d %d %d\n", rawdata[4*j], rawdata[4*j+1], rawdata[4*j+2], rawdata[4*j+3]);
	    */
            /* Call dedisp fn */
            dedisperse(&ds, ichan, rawdata, outdata);

            /* call fold function */
            fold(&ds, ichan, &fb);

#if 0 
            float nn = fb.count[ichan*fb.nbin];
            printf("dedisp_thread: %d %e %e %e %e %d\n", ichan, 
                    fb.data[ichan*fb.nbin*fb.npol + 0]/nn,
                    fb.data[ichan*fb.nbin*fb.npol + 1]/nn,
                    fb.data[ichan*fb.nbin*fb.npol + 2]/nn,
                    fb.data[ichan*fb.nbin*fb.npol + 3]/nn,
                    fb.count[ichan*fb.nbin]);
	    int ii;
	    for (ii=0; ii<fb.nbin; ii++) {
	        printf("dedisp_thread: %4d %e %d\n", ii, fb.data[4*ii], fb.count[ii]);
	    }
	    printf("\n");
#endif

        }
	//log_info("dedisp_thread", "GD : Finished");

	//sprintf(strlog,"L410 %f > %f",fmjd,fmjd_next);
        //log_info("dedisp_thread", strlog);
        /* Add into total foldbuf */
        scale_counts(&fb, (float)(gp.n_packets-gp.n_dropped)/(float)gp.packets_per_block);
        accumulate_folds(&fb_tot, &fb);

        /* Update counters, etc */
        nblock_int++;
        //npacket += gp.n_packets;
        overlap_packets =(int64_t)(((float)gp.packets_per_block*(float)ds.overlap/(float)ds.npts_per_block)+(float)1) ; //ADD LOUIS
        npacket += (int64_t)gp.packets_per_block - overlap_packets;
        ndrop += (int64_t)((gp.packets_per_block - gp.n_packets) + gp.n_dropped);
        tsubint = pf.hdr.dt * (float)(npacket - ndrop ) * gp.packetsize / pf.hdr.nchan / pf.hdr.npol; // Only true for 8-bit data  
	printf("packet_size=%d nchan=%d npol=%d dt_per_block=%e  ==> tsubint=%e\n", gp.packetsize, pf.hdr.nchan, pf.hdr.npol, pf.hdr.dt*gp.packets_per_block* gp.packetsize / pf.hdr.nchan / pf.hdr.npol, (double)tsubint);
	printf("pf.hdr.dt=%f gp.packets_per_block=%d gp.packetsize=%d pf.hdr.nchan=%d pf.hdr.npol=%d\n", pf.hdr.dt, gp.packets_per_block, gp.packetsize, pf.hdr.nchan, pf.hdr.npol);
	printf("npacket=%lld ndrop=%lld nblock_int=%d\n", npacket, ndrop, nblock_int);

            //ds.fft_len = pf.dedisp.fft_len;
            //ds.overlap = pf.dedisp.overlap;
        suboffs += offset;
        hputi4(hdr_out, "NBLOCK", nblock_int);
        hputi4(hdr_out, "NPKT", (int)npacket);
        hputi4(hdr_out, "NDROP", (int)ndrop);
        hputr8(hdr_out, "TSUBINT", tsubint);
        hputr8(hdr_out, "OFFS_SUB", suboffs / (double)nblock_int);
        hputs (hdr_out, "DATATYPE", "FOLDED");

        status_lock_safe(&st);
        hputr8(st.buf, "TSUBINT", tsubint);
        status_unlock_safe(&st);

        /* Mark as free */
        databuf_set_free(db_in, curblock_in);

        /* Go to next block */
        curblock_in = (curblock_in + 1) % db_in->n_block;

        /* Check for cancel */
        pthread_testcancel();

    }
    run = 0;

    printf("Closing dedisp_thread\n");
    log_info("dedisp_thread", "Closing dedisp_thread");

    pthread_exit(NULL);

    pthread_cleanup_pop(0); /* Closes print_timing_report */
    pthread_cleanup_pop(0); /* Closes free_dedispersion */
    pthread_cleanup_pop(0); /* Closes set_exit_status */
}
