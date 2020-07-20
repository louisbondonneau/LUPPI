/* 
 *  nuppi_daq_dedisp
 * 
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

#include "fitshead.h"
#include "logging.h"
#include "status.h"
#include "databuf.h"
#include "params.h"
#include "cuda_utils.h"

#define STATUS "NUPPSTAT"
#include "threads.h"

#include "thread_main.h"

void usage() {
    fprintf(stderr,
            "Usage: nuppi_daq_dedisp [options]\n"
            "Options:\n"
            "  -h, --help        This message\n"
            "  -n, --null        No disk output\n"
	    "  -D, --ds          Downsample instead of fold\n"
           );
}

/* Thread declarations */
void *net_thread(void *args);
void *dedisp_thread(void *args);
void *dedisp_ds_thread(void *args);
void *psrfits_thread(void *args);
void *null_thread(void *args);

int main(int argc, char *argv[]) {

    static struct option long_opts[] = {
        {"help",   0, NULL, 'h'},
        {"null",   0, NULL, 'n'},
	{"ds",     0, NULL, 'D'},
        {0,0,0,0}
    };
    int use_null_thread = 0;
    int ds = 0;
    int opt, opti;
    while ((opt=getopt_long(argc,argv,"hnD",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'n':
                use_null_thread = 1;
                break;
	    case 'D':
	        ds = 1;
		break;
            default:
            case 'h':
                usage();
                exit(0);
                break;
        }
    }

    // -- Ids --
    printf("nuppi_dac_dedisp\n\n");
    log_info("nuppi_daq_dedisp", "nuppi_dac_dedisp start");
    thread_args net_args, dedisp_args, disk_args;
    thread_args net_args_2, dedisp_args_2, disk_args_2;
    thread_args_init(&net_args);
    thread_args_init(&dedisp_args);
    thread_args_init(&disk_args);
    net_args.output_buffer = 1;
    dedisp_args.input_buffer = net_args.output_buffer;
    dedisp_args.output_buffer = 2;
    disk_args.input_buffer = dedisp_args.output_buffer;
    disk_args.output_buffer = 3;
    net_args.gpu_id =  dedisp_args.gpu_id = disk_args.gpu_id = 1;


    net_args.priority = 15;
    dedisp_args.priority = 20;
    disk_args.priority = 20;

    // -- Attach to status shared mem --
    status stat;
    databuf *dbuf_net=NULL, *dbuf_fold=NULL;
    int rv = status_attach(&stat, 1);
    if (rv!=OK) {
        log_error("nuppi_daq_dedisp9", "Error connecting to status shm");
        exit(1);
    }

    // -- Init mutex for PSRFITS template --
    rv = pthread_mutex_init(&lock_psrfits_tpl, NULL);
    if (rv != 0) {
        log_error("nuppi_daq_dedisp10", "Mutex initialization failed");
	exit(1);
    }

    // -- Read status shm, init RUN and read filename --
    char basename[256];
    status_lock_safe(&stat);
    hgets(stat.buf, "BASENAME", 256, basename);
    hputi4(stat.buf, "RUN", 1);
    status_unlock_safe(&stat);

    run=1;
    signal(SIGINT, cc);

    // -- Create or just attach to net shared mem --
    dbuf_net = databuf_attach(net_args.output_buffer);
    if (dbuf_net==NULL)
        dbuf_net = databuf_create(DATABUF_NB, (int64_t)DATABUF_SIZE*1024*1024, net_args.output_buffer);
    if (dbuf_net==NULL) {
        log_error("nuppi_daq_dedisp11", "Error connecting to databuf net shm");
        exit(1);
    }
    databuf_clear(dbuf_net);

    // -- Create or just attach to fold shared mem --
    dbuf_fold = databuf_attach(dedisp_args.output_buffer);
    if (dbuf_fold==NULL)
        dbuf_fold = databuf_create(DATABUF_NB, (int64_t)DATABUF_SIZE*1024*1024, dedisp_args.output_buffer);
    if (dbuf_fold==NULL) {
        log_error("nuppi_daq_dedisp12", "Error connecting to databuf fold shm");
        exit(1);
    }
    databuf_clear(dbuf_fold);


    /* Launch net thread */
    pthread_t net_thread_id;
    rv = pthread_create(&net_thread_id, NULL, net_thread, (void *)&net_args);
    if (rv) { 
        log_error("nuppi_daq_dedisp13", "Error creating net thread");
        perror("pthread_create");
        exit(1);
    }

    /* Launch dedisp thread */
    pthread_t dedisp_thread_id;
    if (ds)
        rv = pthread_create(&dedisp_thread_id, NULL, dedisp_ds_thread, (void *)&dedisp_args);
    else
        rv = pthread_create(&dedisp_thread_id, NULL, dedisp_thread, (void *)&dedisp_args);

    if (rv) { 
        log_error("nuppi_daq_dedisp14", "Error creating dedisp thread");
        perror("pthread_create");
        exit(1);
    }

    /* Launch psrfits/null thread */
    pthread_t disk_thread_id=0;
    if (use_null_thread)
        rv = pthread_create(&disk_thread_id, NULL, null_thread, (void *)&disk_args);
    else
        rv = pthread_create(&disk_thread_id, NULL, psrfits_thread, (void *)&disk_args);
    if (rv) { 
        log_error("nuppi_daq_dedisp15", "Error creating psrfits thread");
        perror("pthread_create");
        exit(1);
    }


    // -- Second GPU --
    pthread_t net_thread_id_2;
    pthread_t dedisp_thread_id_2;
    pthread_t disk_thread_id_2=0;

    //if(getCUDADeviceCount() > 1) {
    if(0) {
        sleep(2);

        // -- Ids --
        thread_args_init(&net_args_2);
        thread_args_init(&dedisp_args_2);
        thread_args_init(&disk_args_2);
        net_args_2.output_buffer = 5;
        dedisp_args_2.input_buffer = net_args_2.output_buffer;
        dedisp_args_2.output_buffer = 6;
        disk_args_2.input_buffer = dedisp_args_2.output_buffer;
        disk_args_2.output_buffer = 7;

	net_args_2.gpu_id =  dedisp_args_2.gpu_id = disk_args_2.gpu_id = 2;

        net_args_2.priority = 20;
        dedisp_args_2.priority = 20;
        disk_args_2.priority = 20;

        databuf *dbuf_net_2=NULL, *dbuf_fold_2=NULL;

        // -- Create or just attach to net shared mem --
	dbuf_net_2 = databuf_attach(net_args_2.output_buffer);
	if (dbuf_net_2==NULL)
	    dbuf_net_2 = databuf_create(DATABUF_NB, (int64_t)DATABUF_SIZE*1024*1024, net_args_2.output_buffer);
	if (dbuf_net_2==NULL) {
	    log_error("nuppi_daq_dedisp16", "Error connecting to databuf net shm for GPU #2");
	    exit(1);
	}
	log_error("nuppi_daq_dedisp", "Connected to databuf net shm for GPU #2");
	databuf_clear(dbuf_net_2);

	// -- Create or just attach to fold shared mem --
	dbuf_fold_2 = databuf_attach(dedisp_args_2.output_buffer);
	if (dbuf_fold_2==NULL)
	    dbuf_fold_2 = databuf_create(DATABUF_NB, (int64_t)DATABUF_SIZE*(int64_t)1024*1024, dedisp_args_2.output_buffer);
	if (dbuf_fold_2==NULL) {
	    log_error("nuppi_daq_dedisp17", "Error connecting to databuf fold shm for GPU #2");
	    exit(1);
	}
	log_error("nuppi_daq_dedisp", "Connected to databuf net shm for GPU #2");
	databuf_clear(dbuf_fold_2);



	/* Launch net thread */
	rv = pthread_create(&net_thread_id_2, NULL, net_thread, (void *)&net_args_2);
	if (rv) { 
	    log_error("nuppi_daq_dedisp18", "Error creating net thread 2");
	    perror("pthread_create");
	    exit(1);
	}

	/* Launch dedisp thread */
	if (ds)
	    rv = pthread_create(&dedisp_thread_id_2, NULL, dedisp_ds_thread, (void *)&dedisp_args_2);
	else
	    rv = pthread_create(&dedisp_thread_id_2, NULL, dedisp_thread, (void *)&dedisp_args_2);
	if (rv) { 
	    log_error("nuppi_daq_dedisp19", "Error creating dedisp thread 2");
	    perror("pthread_create");
	    exit(1);
	}

	/* Launch psrfits/null thread */
	if (use_null_thread)
	    rv = pthread_create(&disk_thread_id_2, NULL, null_thread, (void *)&disk_args_2);
	else
	    rv = pthread_create(&disk_thread_id_2, NULL, psrfits_thread, (void *)&disk_args_2);
	if (rv) { 
	    log_error("nuppi_daq_dedisp20", "Error creating psrfits thread 2");
	    perror("pthread_create");
	    exit(1);
	}
    }	




    /* Alt loop, wait for run=0 */
    while (run) {
        sleep(1); 

	// Read the RUN keyword in the first status shm, to look for a stop order
	status_lock_safe(&stat);
	hgeti4(stat.buf, "RUN", &run);
	status_unlock_safe(&stat);

	if (run == 0) log_info("nuppi_daq_dedisp1", "Caught RUN = 0 signal for end of observation");

        if (disk_args.finished) run=0;
    }

    /* Clean up */
    pthread_cancel(dedisp_thread_id);
    pthread_cancel(net_thread_id);
    pthread_cancel(disk_thread_id);
    pthread_kill(dedisp_thread_id,SIGINT);
    pthread_kill(net_thread_id,SIGINT);
    pthread_kill(disk_thread_id,SIGINT);
    if(getCUDADeviceCount() > 1) {
        pthread_cancel(dedisp_thread_id_2);
	pthread_cancel(net_thread_id_2);
	pthread_cancel(disk_thread_id_2);
	pthread_kill(dedisp_thread_id_2,SIGINT);
	pthread_kill(net_thread_id_2,SIGINT);
	pthread_kill(disk_thread_id_2,SIGINT);
    }

    // -- Join threads --
    pthread_join(net_thread_id,NULL);
    log_info("nuppi_daq_dedisp2", "Joined net thread");
    if(getCUDADeviceCount() > 1) {
        pthread_join(net_thread_id_2,NULL);
	log_info("nuppi_daq_dedisp3", "Joined net thread 2");
    }
    pthread_join(dedisp_thread_id,NULL);
    log_info("nuppi_daq_dedisp4", "Joined dedisp thread"); 
    if(getCUDADeviceCount() > 1) {
        pthread_join(dedisp_thread_id_2,NULL);
	log_info("nuppi_daq_dedisp5", "Joined dedisp thread 2");
    }
    pthread_join(disk_thread_id,NULL);
    log_info("nuppi_daq_dedisp6", "Joined disk thread"); 
    if(getCUDADeviceCount() > 1) {
        pthread_join(disk_thread_id_2,NULL);
	log_info("nuppi_daq_dedisp7", "Joined disk thread 2");
    }

    // -- Destroy args --
    thread_args_destroy(&net_args);
    thread_args_destroy(&dedisp_args);
    thread_args_destroy(&disk_args);

    thread_args_destroy(&net_args_2);
    thread_args_destroy(&dedisp_args_2);
    thread_args_destroy(&disk_args_2);

    pthread_mutex_destroy(&lock_psrfits_tpl);

    char filename[128], hostname[128], strlog[128];
    gethostname(hostname, 127);
    sprintf(filename, "/data/data2/%s-%s.log", basename, hostname);
    if (rename(LOG_FILENAME, filename)!=0) {
        sprintf(strlog, "Error moving log file to %s: %s", filename, strerror(errno));
	log_error("nuppi_daq_dedisp", strlog);
    } else {	
	sprintf(strlog, "Moving log file to %s", filename);
	log_info("nuppi_daq_dedisp8", strlog);
    }	

}
