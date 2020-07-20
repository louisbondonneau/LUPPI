/*  nuppi_read_raw:
 *  Program to read raw data on disk and to
 *  write search mode or dedispersed profiles
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <signal.h>
#include <poll.h>
#include <getopt.h>
#include <errno.h>
#include <stdbool.h>

#include "fitshead.h"
#include "logging.h"
#include "status.h"
#include "databuf.h"
#include "params.h"

#define STATUS "NUPPSTAT"
#include "threads.h"

#include "thread_main.h"

void usage() {
    fprintf(stderr,
            "Usage: nuppi_read_raw [options]\n"
            "Options:\n"
            "  -h, --help        This message\n"
            "  -d, --dedisds     Dedisperse and downsample\n"
            "  -D, --dedis       Dedisperse and fold\n"
            "  -s, --ds          Downsample\n"
            "  -f, --file1       Set the basefilename for GPU0\n"
            "  -F, --file2       Set the basefilename for GPU1\n"
	       "  -g, --gpu         GPUid (default 0)\n"
           );
}

/* Thread declarations */
void *disk2mem_thread(void *args);
void *dedisp_thread(void *args);
void *dedisp_ds_thread(void *args);
void *ds_thread(void *args);
void *psrfits_thread(void *args);

int main(int argc, char *argv[]) {

    static struct option long_opts[] = {
        {"help",    0, NULL, 'h'},
        {"file1",   0, NULL, 'f'},
        {"file2",   0, NULL, 'F'},
        {"dedisds", 0, NULL, 'd'},
        {"dedis",   0, NULL, 'D'},
        {"ds",      0, NULL, 's'},
	   {"gpu",     0, NULL, 'g'},
        {0,0,0,0}
    };
    int opt, opti;
    int dedis = false, dedisds= false, ds = false;
    char basename1[64], basename2[64];
    bool gpu0 = false, gpu1 = false;
    int gpu = 0;
    int upload = 1;
    
    int i=1;
    while(i<argc) {
      if(strncmp(argv[i],"-g",2) == 0) { i++; sscanf(argv[i],"%d",&gpu); }
      if(strncmp(argv[i],"--gpu",2) == 0) { i++; sscanf(argv[i],"%d",&gpu); }
      i++;
    }
    
    
    while ((opt=getopt_long(argc,argv,"hdDgf:F:",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'd':
                dedisds=true;
                break;
            case 'D':
                dedis=true;
                break;
            case 's':
                ds=true;
                break;
	       case 't':
	           upload = 0;
		      break;
	       case 'g':
		      break;
	    case 'f':
	        sscanf(optarg, "%s", basename1);
		gpu0 = true;
		break;
	    case 'F':
	        sscanf(optarg, "%s", basename2);
		gpu1 = true;
		break;
            default:
            case 'h':
                usage();
                exit(0);
                break;
        }
    }

    printf("BEAM = %d\n",gpu);
    printf("gpu = %d\n",gpu%2 );
    // -- First part of the band --
    thread_args disk2mem_args, dedisp_args, disk_args;
    thread_args disk2mem_args2, dedisp_args2, disk_args2;
    pthread_t disk2mem_thread_id, dedisp_thread_id, disk_thread_id;
    pthread_t disk2mem_thread_id2, dedisp_thread_id2, disk_thread_id2;
    status stat;

    if (gpu0) {

	// -- thread args, start at 1 --
	thread_args_init(&disk2mem_args);
	thread_args_init(&dedisp_args);
	thread_args_init(&disk_args);
	
	
	disk2mem_args.gpu_id = dedisp_args.gpu_id = disk_args.gpu_id = gpu;
	disk2mem_args.output_buffer = 1 + 4*(disk2mem_args.gpu_id) ;
	
	dedisp_args.input_buffer = disk2mem_args.output_buffer;
	dedisp_args.output_buffer = 2 + 4*(dedisp_args.gpu_id);
	
	disk_args.input_buffer = dedisp_args.output_buffer;
	disk_args.output_buffer = 3 + 4*(disk_args.gpu_id);
	
	disk2mem_args.priority = 15;
	dedisp_args.priority = 20;
	disk_args.priority = 20;
	
	
	strcpy(disk2mem_args.string, basename1);
	// -- Init status shared mem --
	databuf *dbuf=NULL, *dbuf_fold=NULL;
	int rv = status_attach(&stat, disk2mem_args.gpu_id);
        printf("ICIIIIIIIIIIII8\n");
	if (rv!=OK) {
	    log_error("nuppi_read_raw", "Error connecting to status shm");
	    exit(1);
	}
   
	// -- Read status shm, init RUN and read filename --
	status_lock_safe(&stat);
	//hgets(stat.buf, "BASENAME", 256, basename);
	hputi4(stat.buf, "RUN", 1);
	status_unlock_safe(&stat);


	// -- Create or just attach to net shared mem --
	dbuf = databuf_attach(disk2mem_args.output_buffer);
        printf("ICIIIIIIIIIIII7\n");
	printf("disk2mem_args.output_buffer = 0x%06x \n",disk2mem_args.output_buffer);
	if (dbuf==NULL) dbuf = databuf_create(DATABUF_NB, (int64_t)DATABUF_SIZE*1024*1024, disk2mem_args.output_buffer);
	if (dbuf==NULL) {
	    log_error("nuppi_read_raw", "Error connecting to databuf net shm");
	    exit(1);
	}
	databuf_clear(dbuf);
        log_error("nuppi_read_raw","Create or just attach to fold shared mem");
	// -- Create or just attach to fold shared mem --
	dbuf_fold = databuf_attach(dedisp_args.output_buffer);
	if (dbuf_fold==NULL) dbuf_fold = databuf_create(DATABUF_NB, (int64_t)DATABUF_SIZE*1024*1024, dedisp_args.output_buffer);
	if (dbuf_fold==NULL) {
	    log_error("nuppi_read_raw", "Error connecting to databuf fold shm");
	    exit(1);
	}
	databuf_clear(dbuf_fold);
        log_info("nuppi_read_raw","Launch disk2mem thread");
	/* Launch disk2mem thread */
	rv = pthread_create(&disk2mem_thread_id, NULL, disk2mem_thread, (void *)&disk2mem_args);
	if (rv) { 
	    log_error("nuppi_read_raw", "Error creating net thread");
	    perror("pthread_create");
	    exit(1);
	}
	
        log_info("nuppi_read_raw","Launch dedisp thread");
	/* Launch dedisp thread */
	if (ds) rv = pthread_create(&dedisp_thread_id, NULL, ds_thread, (void *)&dedisp_args);
	else if (dedisds) rv = pthread_create(&dedisp_thread_id, NULL, dedisp_ds_thread, (void *)&dedisp_args);
	else if (dedis) rv = pthread_create(&dedisp_thread_id, NULL, dedisp_thread, (void *)&dedisp_args);
	if (rv) { 
	    log_error("nuppi_read_raw", "Error creating dedisp/ds thread");
	    exit(1);
	}

	/* Launch psrfits thread */
	printf("nuppi_read_raw: databuf_id=%d\n",disk_args.output_buffer);
	rv = pthread_create(&disk_thread_id, NULL, psrfits_thread, (void *)&disk_args);
	if (rv) {
	    log_error("nuppi_read_raw", "Error creating psrfits thread");
	    perror("pthread_create");
	    exit(1);
	}
    }



    status stat2;
    if(gpu1) {
	// -- thread args, start at 1 --
	thread_args_init(&disk2mem_args2);
	thread_args_init(&dedisp_args2);
	thread_args_init(&disk_args2);
	disk2mem_args2.output_buffer = 5; disk2mem_args2.gpu_id = 2; disk2mem_args2.priority = 15;
	dedisp_args2.input_buffer = disk2mem_args2.output_buffer; dedisp_args2.output_buffer = 6; dedisp_args2.gpu_id = 2; dedisp_args2.priority = 20;
	//Add louis
	disk_args2.input_buffer = dedisp_args2.output_buffer; disk_args2.output_buffer = 7; disk_args2.gpu_id = 2; disk_args2.priority = 20;
	//Add louis

	strcpy(disk2mem_args2.string, basename2);

	// -- Init status shared mem --
	databuf *dbuf=NULL, *dbuf_fold=NULL;
	int rv = status_attach(&stat2, disk2mem_args2.gpu_id);
	if (rv!=OK) {
		log_error("nuppi_read_raw", "Error connecting to status shm");
		exit(1);
	}
   
	// -- Read status shm, init RUN and read filename --
	status_lock_safe(&stat2);
	//hgets(stat2.buf, "BASENAME", 256, basename);
	hputi4(stat2.buf, "RUN", 1);
	status_unlock_safe(&stat2);

	// -- Create or just attach to net shared mem --
	dbuf = databuf_attach(disk2mem_args2.output_buffer);
	printf("disk2mem_args.output_buffer = 0X%06x \n",disk2mem_args2.output_buffer);
	if (dbuf==NULL) dbuf = databuf_create(DATABUF_NB, (int64_t)DATABUF_SIZE*1024*1024, disk2mem_args2.output_buffer);
	if (dbuf==NULL) {
	    log_error("nuppi_read_raw", "Error connecting to databuf net shm");
	    exit(1);
	}
	databuf_clear(dbuf);

	// -- Create or just attach to fold shared mem --
	dbuf_fold = databuf_attach(dedisp_args2.output_buffer);
	if (dbuf_fold==NULL) dbuf_fold = databuf_create(DATABUF_NB, (int64_t)DATABUF_SIZE*1024*1024, dedisp_args2.output_buffer);
	if (dbuf_fold==NULL) {
	    log_error("nuppi_read_raw", "Error connecting to databuf fold shm");
	    exit(1);
	}
	databuf_clear(dbuf_fold);

	/* Launch disk2mem thread */
	rv = pthread_create(&disk2mem_thread_id2, NULL, disk2mem_thread, (void *)&disk2mem_args2);
	if (rv) { 
	    log_error("nuppi_read_raw", "Error creating net thread");
	    perror("pthread_create");
	    exit(1);
	}

	/* Launch dedisp thread */
	if (ds) rv = pthread_create(&dedisp_thread_id2, NULL, ds_thread, (void *)&dedisp_args2);
	else if (dedisds) rv = pthread_create(&dedisp_thread_id2, NULL, dedisp_ds_thread, (void *)&dedisp_args2);
	else if (dedis) rv = pthread_create(&dedisp_thread_id2, NULL, dedisp_thread, (void *)&dedisp_args2);
	if (rv) { 
	    log_error("nuppi_read_raw", "Error creating dedisp/ds thread");
	    exit(1);
	}

	/* Launch psrfits thread */
	char strlog[256];
	rv = pthread_create(&disk_thread_id2, NULL, psrfits_thread, (void *)&disk_args2);
	if (rv) {
	    log_error("nuppi_read_raw", "Error creating psrfits thread");
	    perror("pthread_create");
	    exit(1);
	}
    }
    log_info("nuppi_read_raw", "-- Run Signal --");
    printf("-- Run Signal --\n");
    // -- Run Signal --
    run=1;
    signal(SIGINT, cc);

    /* Wait for end */
    while (run) { 
		// Read the RUN keyword in the first status shm, to look for a stop order
	    if(gpu0) {
		status_lock_safe(&stat);
		status_unlock_safe(&stat);
		sleep(5);
		if (run == 0) log_info("nuppi_read_raw", "GPU #0 : Caught RUN = 0 signal for end of observation");
	    }	

	    if(gpu1) {
		status_lock_safe(&stat2);
		hgeti4(stat2.buf, "RUN", &run);
		status_unlock_safe(&stat2);
		if (run == 0) log_info("nuppi_read_raw", "GPU #1 : Caught RUN = 0 signal for end of observation");
	    }	

		//if (null_args.finished || null_args2.finished) run=0;
    }

    printf("-- nuppi_read RAW  RUN=0 --\n");fflush(stdout);
    if(gpu0) {
	// -- First cancel threads -- 
	pthread_cancel(disk2mem_thread_id);
	pthread_cancel(dedisp_thread_id);
	pthread_cancel(disk_thread_id);

	// -- Then kill threads -- 
	pthread_kill(disk2mem_thread_id,SIGINT);
	pthread_kill(dedisp_thread_id,SIGINT);
	pthread_kill(disk_thread_id,SIGINT);

        // -- Join threads --
	pthread_join(disk2mem_thread_id,NULL);
	log_info("nuppi_read_raw", "Joined disk2mem thread");
	pthread_join(dedisp_thread_id,NULL);
	log_info("nuppi_read_raw", "Joined dedisp/ds thread");
	pthread_join(disk_thread_id,NULL);
	log_info("nuppi_read_raw", "Joined disk thread");

	// -- Destroy args --
	thread_args_destroy(&disk2mem_args);
	thread_args_destroy(&dedisp_args);
	thread_args_destroy(&disk_args);
    }	

    if(gpu1) {
	// -- First cancel threads -- 
	pthread_cancel(disk2mem_thread_id2);
	pthread_cancel(dedisp_thread_id2);
	pthread_cancel(disk_thread_id2);

	// -- Then kill threads -- 
	pthread_kill(disk2mem_thread_id2,SIGINT);
	pthread_kill(dedisp_thread_id2,SIGINT);
	pthread_kill(disk_thread_id2,SIGINT);

        // -- Join threads --
	pthread_join(disk2mem_thread_id2,NULL);
	log_info("nuppi_read_raw", "Joined disk2mem thread #2");
	pthread_join(dedisp_thread_id2,NULL);
	log_info("nuppi_read_raw", "Joined dedisp/ds thread #2");
	pthread_join(disk_thread_id2,NULL);
	log_info("nuppi_read_raw", "Joined disk thread #2");

	// -- Destroy args --
	thread_args_destroy(&disk2mem_args2);
	thread_args_destroy(&dedisp_args2);
	thread_args_destroy(&disk_args2);
    }	

    // -- Log file --
    char cmd[128], hostname[128];
    gethostname(hostname, 127);
    //sprintf(cmd, "mv %s /home/pulsar/data/%s-%s.log", LOG_FILENAME, basename, hostname);
    //log_info("nuppi_daq_dedisp", cmd);
    //system(cmd);

}
