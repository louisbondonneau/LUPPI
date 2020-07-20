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
#include <sys/types.h>
#include <sys/stat.h>

#include <signal.h>
#include <poll.h>
#include <getopt.h>
#include <errno.h>
#include <utime.h>
#include <dirent.h>

#include "fitshead.h"
#include "logging.h"
#include "status.h"
#include "databuf.h"
#include "params.h"
#include "cuda_utils.h"
#include "cpu_utils.h"

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
	    "  -t, --test        Test mode without ssh upload\n"
	    "  -g, --gpu         GPUid (default 0)\n"
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
	{"gpu",     0, NULL, 'g'},
        {0,0,0,0}
    };
    int use_null_thread = 0;
    int ds = 0;
    int gpu = 0;
    int upload = 1;
    int opt, opti;


    int i=1;
    while(i<argc) {
      if(strncmp(argv[i],"-g",2) == 0) { i++; sscanf(argv[i],"%d",&gpu); }
      if(strncmp(argv[i],"--gpu",2) == 0) { i++; sscanf(argv[i],"%d",&gpu); }
      i++;
    }


    while ((opt=getopt_long(argc,argv,"htnDg",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'n':
                use_null_thread = 1;
                break;
            case 'D':
                ds = 1;
                break;
            case 't':
                upload = 0;
                break;
            case 'g':
                break;
            default:
            case 'h':
                usage();
                exit(0);
                break;
        }
    }

    printf("upload = %d\n",upload);
    printf("BEAM = %d\n",gpu);
    printf("gpu = %d\n",gpu%2 );
    // -- Ids --
    printf("nuppi_dac_dedisp\n\n");
    log_info("nuppi_daq_dedisp", "nuppi_dac_dedisp start");
    thread_args net_args, dedisp_args, disk_args;
    thread_args net_args_2, dedisp_args_2, disk_args_2;
    thread_args_init(&net_args);
    thread_args_init(&dedisp_args);
    thread_args_init(&disk_args);

    net_args.gpu_id =  dedisp_args.gpu_id = disk_args.gpu_id = gpu;
    net_args.output_buffer = 1 + 4*(net_args.gpu_id);
    dedisp_args.input_buffer = net_args.output_buffer;
    dedisp_args.output_buffer = 2 + 4*(net_args.gpu_id);
    disk_args.input_buffer = dedisp_args.output_buffer;
    disk_args.output_buffer = 3 + 4*(net_args.gpu_id);
    //setcpuaffinity(4 + 4*(net_args.gpu_id) -1);
   /*
    net_args.output_buffer = 1;
    dedisp_args.input_buffer = net_args.output_buffer;
    dedisp_args.output_buffer = 2;
    disk_args.input_buffer = dedisp_args.output_buffer;
    disk_args.output_buffer = 3;
    net_args.gpu_id =  dedisp_args.gpu_id = disk_args.gpu_id = 1;
   */
    net_args.priority = 20;
    dedisp_args.priority = 20;
    disk_args.priority = 20;

    // -- Attach to status shared mem --
    status stat;
    databuf *dbuf_net=NULL, *dbuf_fold=NULL;
    int rv = status_attach(&stat, (int)gpu);
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
    char psrname[256];
    char projid[256];
    char julianday[256];
    char year[32], month[32], day[32];
    status_lock_safe(&stat);
    hgets(stat.buf, "BASENAME", 256, basename);
    hgets(stat.buf, "BASENAME", 256, psrname);
    hgets(stat.buf, "PROJID", 256, projid);
    hgets(stat.buf, "STT_JDAY", 256, julianday);
    strncpy(year, julianday+1, 4);
    strncpy(month, julianday+5, 2);
    strncpy(day, julianday+7, 9);
    printf("%.4s/%.2s/%.2s  %.4s\n", year, month, day, projid);

    int port;
    get_int(stat.buf,"DATAPORT", &port, 1491);
    printf("\ndataport = %d\n", port );

    char ds_time[256];
    hgets(stat.buf, "DS_TIME", 256 , ds_time);
    if(atoi(ds_time) > 1) ds = 1;   // add LOUIS

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
    // -- Join threads --
    pthread_join(net_thread_id,NULL);
    log_info("nuppi_daq_dedisp2", "Joined net thread");
    pthread_join(dedisp_thread_id,NULL);
    log_info("nuppi_daq_dedisp4", "Joined dedisp thread"); 
    pthread_join(disk_thread_id,NULL);
    log_info("nuppi_daq_dedisp6", "Joined disk thread"); 

    // -- Destroy args --
    thread_args_destroy(&net_args);
    thread_args_destroy(&dedisp_args);
    thread_args_destroy(&disk_args);

    pthread_mutex_destroy(&lock_psrfits_tpl);

    char filename[128], hostname[128], strlog[128], logname[128], logdir[128];

    char source[256], mode[256], datadir[256], path[256], commande[2048];
    hgets(stat.buf, "SRC_NAME", 256, source);
    hgets(stat.buf, "OBS_MODE", 256, mode);
    hgets(stat.buf, "DATADIR", 256, datadir);

    gethostname(hostname, 127);
    sprintf(logdir, "%s/BEAM%d_%d", datadir, gpu, (int)getpid());
    //sprintf(filename, "/data/data2/%s-%s.log", basename, hostname);
    sprintf(filename, "%s/%s-%s.log", logdir, basename, hostname);
    sprintf(logname,"%s_%d.log", LOG_FILENAME, (int)getpid());


    printf("%.4s/%.2s/%.2s  %.4s\n", year, month, day, projid);


    //############SRC/MODE DIR###############

    sprintf(path, "%s/DATA/%s", datadir,source);
    opendir(path);
    switch (errno) {
        case ENOENT:
	    sprintf(commande, "mkdir %s 2>&1", path);
            system(commande);
            sprintf(commande, "ssh nfrplsobs@databf2dt 'mkdir /databf2/nenufar-pulsar/DATA/%s' 2>&1", source);
            system(commande);
	    sprintf(strlog, "mkdir directory %s", path);
	    log_info("nuppi_daq_dedisp", strlog);
    }
    
    
    sprintf(path, "%s/DATA/%s/%s", datadir, source, mode);
    opendir(path);
    switch (errno) {
        case ENOENT:
	    sprintf(commande, "mkdir %s 2>&1", path);
            system(commande);
            sprintf(commande, "ssh nfrplsobs@databf2dt 'mkdir /databf2/nenufar-pulsar/DATA/%s/%s' 2>&1", source, mode);
            system(commande);
	    sprintf(strlog, "mkdir directory %s", path);
	    log_info("nuppi_daq_dedisp", strlog);
    }



    //############projid/year/month DIR###############

    sprintf(path, "%s/%.4s", datadir, projid);
    opendir(path);
    switch (errno) {
        case ENOENT:
        sprintf(commande, "mkdir %s 2>&1", path);
            system(commande);
        sprintf(strlog, "mkdir directory %s", path);
        log_info("nuppi_daq_dedisp", strlog);
    }
    

    sprintf(path, "%s/%.4s/%.4s", datadir, projid, year);
    opendir(path);
    switch (errno) {
        case ENOENT:
        sprintf(commande, "mkdir %s 2>&1", path);
        system(commande);
        sprintf(commande, "ssh nfrplsobs@databf2dt 'mkdir /databf2/nenufar-pulsar/%.4s/%.4s' 2>&1", projid, year);
        system(commande);
        sprintf(strlog, "mkdir directory %s", path);
        log_info("nuppi_daq_dedisp", strlog);
    }

    sprintf(path, "%s/%.4s/%.4s/%.2s", datadir, projid, year, month);
    opendir(path);
    switch (errno) {
        case ENOENT:
        sprintf(commande, "mkdir %s 2>&1", path);
        system(commande);
        sprintf(commande, "ssh nfrplsobs@databf2dt 'mkdir /databf2/nenufar-pulsar/%.4s/%.4s/%.2s' 2>&1", projid, year, month);
        system(commande);
        sprintf(strlog, "mkdir directory %s", path);
        log_info("nuppi_daq_dedisp", strlog);
    }
    

    
    //############log DIR###############

    sprintf(commande, "sudo mkdir %s 2>&1", logdir);
    system(commande);
    sprintf(strlog, "Moving log file to %s", filename);
    log_info("nuppi_daq_dedisp8", strlog);
    if (rename(logname, filename)!=0) {
        sprintf(strlog, "Error moving log file to %s: %s", filename, strerror(errno));
	log_error("nuppi_daq_dedisp", strlog);
    }
    

    printf("%s \n", psrname);fflush(stdout);
	char *obs_time = strtok(psrname, "_");
	obs_time = strtok(NULL, "_D");
	obs_time = strtok(obs_time, "T");
    printf("%s \n", obs_time);fflush(stdout);
    printf("%s \n", basename);fflush(stdout);
	

    sprintf(commande, "sudo mv %s/%s*.fits %s/ 2>&1", datadir, basename, path);
    printf("%s \n", commande);fflush(stdout);
    system(commande);
	
    sprintf(commande, "sudo cp %s/PSETUP-at-%s*.log %s/ 2>&1", datadir, obs_time, logdir);
    printf("%s \n", commande);fflush(stdout);
    system(commande);

    sprintf(commande, "sudo cp %s/PARSET-at-%s*.parset %s/ 2>&1", datadir, obs_time, logdir);
    printf("%s \n", commande);fflush(stdout);
    system(commande);

    sprintf(commande, "sudo mv %s/SHELL--at-%s*BEAM%d.log %s/ 2>&1", datadir, obs_time, gpu, logdir);
    printf("%s \n", commande);fflush(stdout);
    system(commande);

    sprintf(commande, "sudo tar -zcvf %s-%s.tar -C %s BEAM%d_%d/ 2>&1", basename, hostname, datadir, gpu, (int)getpid());
    //sprintf(commande, "sudo tar -zcvf %s-%s.tar %s/ 2>&1", basename, hostname, logdir);
    printf("%s \n", commande);fflush(stdout);
    system(commande);

    sprintf(commande, "sudo mv %s-%s.tar %s/ 2>&1", basename, hostname, path);
    printf("%s \n", commande);fflush(stdout);
    system(commande);


    if (upload) {
    //sprintf(commande, "rsync -av --remove-source-files -e \"ssh \"  %s/%s*  nfrplsobs@databf2dt:/databf2/nenufar-pulsar/DATA/%s/%s/ 2>&1", path, basename, source, mode);
    sprintf(commande, "rsync -avt --remove-source-files --bwlimit=50000 -e \"ssh \"  %s/%s*  nfrplsobs@databf2dt:/databf2/nenufar-pulsar/%.4s/%.4s/%.2s/ 2>&1", path, basename, projid, year, month);
    printf("%s \n", commande);fflush(stdout);
    system(commande);

    sprintf(commande, "ssh nfrplsobs@databf2dt 'ln /databf2/nenufar-pulsar/%.4s/%.4s/%.2s/%s* /databf2/nenufar-pulsar/DATA/%s/%s/' 2>&1", projid, year, month, basename, source, mode);
    printf("%s \n", commande);fflush(stdout);
    system(commande);
    
    }

    sprintf(commande, "sudo rm -r %s 2>&1", logdir);
    printf("%s \n", commande);fflush(stdout);
    system(commande);
	
    sprintf(commande, "sudo mv `ls %s/PARSET-at-*.parset | grep -v %s` %s/OLD_parset/ 2>&1", datadir, obs_time, datadir);
    printf("%s \n", commande);fflush(stdout);
    system(commande);

    sprintf(commande, "sudo mv `ls %s/PSETUP-at-*.log | grep -v %s` %s/OLD_LOG/ 2>&1", datadir, obs_time, datadir);
    printf("%s \n", commande);fflush(stdout);
    system(commande);



}
