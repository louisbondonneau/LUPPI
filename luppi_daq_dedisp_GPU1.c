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
            "  -D, --ds              Downsample instead of fold\n"
            "  -t, --test            Test mode without ssh upload\n"
            "  -f, --fasttransfer    transfer the observation using a mix of scp and rsync for max speed\n"
            "  -g, --gpu             GPUid (default 0)\n"
            "  -j, --databfdirname   optional subdirectory on databf\n"
           );
}

/* Thread declarations */
void *net_thread(void *args);
void *dedisp_thread(void *args);
void *dedisp_ds_thread(void *args);
void *psrfits_thread(void *args);
void *null_thread(void *args);

char datadir[256];
char basename[256];
char script[128], script_fast[128], script_slow[128];
char databfdirname[256] = {0};

int main(int argc, char *argv[]) {

    static struct option long_opts[] = {
        {"help",   0, NULL, 'h'},
        {"null",   0, NULL, 'n'},
        {"ds",     0, NULL, 'D'},
        {"gpu",    0, NULL, 'g'},
        {"databfdirname", 0, NULL, 'j'},
        {0,0,0,0}
    };
    int use_null_thread = 0;
    int ds = 0;
    int gpu = 0;
    int upload = 1;
    int upload_fast = 0;
    int opt, opti;


    int i=1;
    while(i<argc) {
      if(strncmp(argv[i],"-g",2) == 0) { i++; sscanf(argv[i],"%d",&gpu); }
      if(strncmp(argv[i],"--gpu",2) == 0) { i++; sscanf(argv[i],"%d",&gpu); }

      if(strncmp(argv[i],"-j",2) == 0) { i++; sscanf(argv[i],"%d",&databfdirname); }
      if(strncmp(argv[i],"--databfdirname",2) == 0) { i++; sscanf(argv[i],"%d",&databfdirname); }
      i++;
    }


    while ((opt=getopt_long(argc,argv,"htfnDgj",long_opts,&opti))!=-1) {
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
            case 'f':
                upload_fast = 1;
                break;
            case 'g':
                break;
            case 'j':
                break;
            default:
            case 'h':
                usage();
                exit(0);
                break;
        }
    }

    printf("upload = %d\n",upload);
    printf("upload_fast = %d\n",upload_fast);
    printf("BEAM = %d\n",gpu);
    printf("gpu = %d\n",gpu%2 );
    printf("databfdirname = %s\n",databfdirname );
    // -- Ids --
    printf("nuppi_dac_dedisp\n\n");
    log_info("nuppi_daq_dedisp", "nuppi_dac_dedisp start");
    exit(0);
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

    // -- preparing upload scripts -

    char string[1024], filename[128], hostname[128], strlog[128], logname[128], logdir[128];
    char source[256], mode[256] , path[256], commande[2048], date[128];

    hgets(stat.buf, "SRC_NAME", 256, source);
    hgets(stat.buf, "OBS_MODE", 256, mode);
    hgets(stat.buf, "DATADIR", 256, datadir);
    gethostname(hostname, 127);
    sprintf(logdir, "%s/BEAM%d_%d", datadir, gpu, (int)getpid());
    sprintf(filename, "%s/%s-%s.log", logdir, basename, hostname);
    sprintf(logname,"%s_%d.log", LOG_FILENAME, (int)getpid());

    sprintf(script,"%s/SCRIPTS/%s_script.sh", datadir, basename);
    sprintf(script_fast,"%s/SCRIPTS/%s_script_fast.sh", datadir, basename);
    sprintf(script_slow,"%s/SCRIPTS/%s_script_slow.sh", datadir, basename);
    printf("%.4s/%.2s/%.2s  %.4s\n", year, month, day, projid);
    FILE *pfo_slow;
    FILE *pfo_fast;
    if((pfo_slow=fopen(script_slow,"w"))!=NULL) {
    if((pfo_fast=fopen(script_fast,"w"))!=NULL) {
        //######## mkdir DATA/SOURCE
        //sprintf(string,"mkdir %s/DATA/%s 2>&1\n",datadir,source); fputs(string,pfo_slow); fputs(string,pfo_fast);
        //sprintf(string,"if [ \"$1\" !=  \"\" ] && [ \"$1\" !=  \"rsync\" ]; then echo 'error: '$1' is not a valide option (rsync or empty is valide)' ; exit; fi\n"); fputs(string,pfo_fast);
        sprintf(string,"ssh nfrplsobs@databfnfrdt 'mkdir /data/nenufar-pulsar/DATA/%s' 2>&1\n",source); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(strlog, "script_mkdir directory %s/DATA/%s on data", datadir,source);
        log_info("nuppi_daq_dedisp", strlog);
        //######## mkdir DATA/SOURCE/MODE
        //sprintf(string,"mkdir %s/DATA/%s/%s 2>&1\n",datadir,source, mode); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(string,"ssh nfrplsobs@databfnfrdt 'mkdir /data/nenufar-pulsar/DATA/%s/%s' 2>&1\n",source, mode); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(strlog, "script_mkdir directory %s/DATA/%s/%s on data", datadir,source, mode);
        log_info("nuppi_daq_dedisp", strlog);
        //######## mkdir projid
        sprintf(path, "%s/%.4s", datadir, projid);
        sprintf(string,"mkdir %s 2>&1\n",path); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(string,"chmod a+rw %s\n",path); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(strlog, "script_mkdir directory %s on %s", path, hostname);
        log_info("nuppi_daq_dedisp", strlog);
        //######## mkdir projid/year
        sprintf(path, "%s/%.4s/%.4s", datadir, projid, year);
        sprintf(string,"mkdir %s 2>&1\n",path); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(string,"chmod a+rw %s\n",path); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(string,"ssh nfrplsobs@databfnfrdt 'mkdir /data/nenufar-pulsar/%.4s/%.4s' 2>&1\n",projid, year); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(strlog, "script_mkdir directory %s on %s and data", path, hostname);
        log_info("nuppi_daq_dedisp", strlog);
        //######## mkdir projid/year/month
        sprintf(path, "%s/%.4s/%.4s/%.2s", datadir, projid, year, month);
        sprintf(string,"mkdir %s 2>&1\n",path); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(string,"chmod a+rw %s\n",path); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(string,"ssh nfrplsobs@databfnfrdt 'mkdir /data/nenufar-pulsar/%.4s/%.4s/%.2s' 2>&1\n",projid, year, month); fputs(string,pfo_slow); fputs(string,pfo_fast);
        if (databfdirname[0] != '\0') {
            sprintf(string,"ssh nfrplsobs@databfnfrdt 'mkdir /data/nenufar-pulsar/%.4s/%.4s/%.2s/%s' 2>&1\n",projid, year, month, databfdirname); fputs(string,pfo_slow); fputs(string,pfo_fast);
        }
        sprintf(strlog, "script_mkdir directory %s on %s and data", path, hostname);
        log_info("nuppi_daq_dedisp", strlog);
        //######## mkdir log_dir
        sprintf(string,"mkdir %s 2>&1\n",logdir); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(strlog, "script_mkdir directory %s on %s", logdir, hostname);
        log_info("nuppi_daq_dedisp", strlog);
        //######## renamed log_file
        sprintf(string,"mv %s %s 2>&1\n",logname, filename); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(strlog, "script_mv logfile on %s", hostname);
        log_info("nuppi_daq_dedisp", strlog);

        printf("%s \n", psrname);fflush(stdout);
        char *obs_time = strtok(psrname, "_");
        obs_time = strtok(NULL, "_D");
        obs_time = strtok(obs_time, "T");
        printf("%s \n", obs_time);fflush(stdout);
        printf("%s \n", basename);fflush(stdout);


        //######## preparing tar_file
        sprintf(path, "%s/%.4s/%.4s/%.2s", datadir, projid, year, month);
        sprintf(string, "mv %s/%s*.fits %s/ 2>&1\n", datadir, basename, path); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(string, "cp -p %s/PSETUP-at-%s*.log %s/ 2>&1\n", datadir, obs_time, logdir); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(string, "cp -p %s/PARSET-at-%s*.parset %s/ 2>&1\n", datadir, obs_time, logdir); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(string, "mv %s/SHELL--at-%s*BEAM%d.log %s/ 2>&1\n", datadir, obs_time, gpu, logdir); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(string, "tar -zcvf %s-%s.tar -C %s BEAM%d_%d/ 2>&1\n", basename, hostname, datadir, gpu, (int)getpid()); fputs(string,pfo_slow); fputs(string,pfo_fast);
        sprintf(string, "mv %s-%s.tar %s/ 2>&1\n", basename, hostname, path); fputs(string,pfo_slow); fputs(string,pfo_fast);


        //######## upload files on databf2
        sprintf(date, "date \'+%%Y-%%m-%%d %%H:%%M:%%S\'");

        sprintf(string, "echo \"$(%s) $USER %s Start\" >> /data/upload_file.log\n", date, script_slow); fputs(string,pfo_slow);
        sprintf(string, "echo \"$(%s) $USER %s Start\" >> /data/upload_file.log\n", date, script_fast); fputs(string,pfo_fast);
        sprintf(string, "rsync -avt --chmod=o+r,o-wx --remove-source-files --bwlimit=50000 -e \"ssh -T -c aes128-ctr -o Compression=no\"  %s/%s*  nfrplsobs@databfnfrdt:/data/nenufar-pulsar/%.4s/%.4s/%.2s/\n", path, basename, projid, year, month); fputs(string,pfo_slow);
        sprintf(string, "scp -p -o Compression=no -c aes128-ctr %s/%s*  nfrplsobs@databfnfrdt:/data/nenufar-pulsar/%.4s/%.4s/%.2s/\n", path, basename, projid, year, month); fputs(string,pfo_fast);
        sprintf(string, "rsync -avt --chmod=o+r,o-wx --remove-source-files -e \"ssh -T -c aes128-ctr -o Compression=no\"  %s/%s*  nfrplsobs@databfnfrdt:/data/nenufar-pulsar/%.4s/%.4s/%.2s/\n", path, basename, projid, year, month); fputs(string,pfo_fast);
        sprintf(string, "echo \"$(%s) $USER %s Stop\" >> /data/upload_file.log\n", date, script_slow); fputs(string,pfo_slow);
        sprintf(string, "echo \"$(%s) $USER %s Stop\" >> /data/upload_file.log\n", date, script_fast); fputs(string,pfo_fast);
        //######## creating link on databf2 in /databf2/nenufar-pulsar/DATA/
        if (databfdirname[0] != '\0') {
            sprintf(string, "ssh nfrplsobs@databfnfrdt 'ln /data/nenufar-pulsar/%.4s/%.4s/%.2s/%s/%s* /data/nenufar-pulsar/DATA/%s/%s/' 2>&1\n", projid, year, month, databfdirname, basename, source, mode); fputs(string,pfo_slow); fputs(string,pfo_fast);
        } else {
            sprintf(string, "ssh nfrplsobs@databfnfrdt 'ln /data/nenufar-pulsar/%.4s/%.4s/%.2s/%s* /data/nenufar-pulsar/DATA/%s/%s/' 2>&1\n", projid, year, month, basename, source, mode); fputs(string,pfo_slow); fputs(string,pfo_fast);

        }
        //######## cleaning log files and dir
        sprintf(string, "rm -r %s 2>&1\n", logdir); fputs(string,pfo_slow); fputs(string,pfo_fast);
        // sprintf(string, "mv `ls %s/PARSET-at-*.parset | grep -v %s` %s/OLD_parset/ 2>&1\n", datadir, obs_time, datadir); fputs(string,pfo_slow); fputs(string,pfo_fast);
        // sprintf(string, "mv `ls %s/PSETUP-at-*.log | grep -v %s` %s/OLD_LOG/ 2>&1\n", datadir, obs_time, datadir); fputs(string,pfo_slow); fputs(string,pfo_fast);
        // sprintf(string, "mv `ls %s/SHELL--at-*.log | grep -v %s` %s/OLD_LOG/ 2>&1\n", datadir, obs_time, datadir); fputs(string,pfo_slow); fputs(string,pfo_fast);

    }}
    fclose(pfo_slow);
    fclose(pfo_fast);
    FILE *pfo;
    if((pfo=fopen(script,"w"))!=NULL) {
        //######## argu fast and slow -> Default is slow
        sprintf(string,"if [ \"$1\" !=  \"\" ] && [ \"$1\" !=  \"fast\" ] && [ \"$1\" !=  \"slow\" ]; then echo \"error: $1 is not a valide option (fast, slow or empty are valide. Default is slow)\" ; exit; fi; if [ \"$1\" ==  \"fast\" ]; then\n"); fputs(string,pfo);

        //######## Fast script
        pfo_fast=fopen(script_fast,"rt");
        while( ! feof( pfo_fast ) ) {
            fgets(string, 1024, pfo_fast);
            fputs("    ", pfo);
            fputs(string, pfo);
        }
        fclose(pfo_fast);

        sprintf(string,"else\n"); fputs(string,pfo);

        //######## Slow script
        pfo_slow=fopen(script_slow,"rt");
        while( ! feof( pfo_slow ) ) {
            fgets(string, 1024, pfo_slow);
            fputs("    ", pfo);
            fputs(string, pfo);
        }
        fclose(pfo_slow);
        sprintf(string,"fi\n"); fputs(string,pfo);
    }
    fclose(pfo);


    sprintf(commande, "sudo chmod a+rw %s", script);
    printf("%s \n", commande);fflush(stdout);
    system(commande);
    sprintf(commande, "sudo chmod a+rw %s", script_slow);
    printf("%s \n", commande);fflush(stdout);
    system(commande);
    sprintf(commande, "sudo chmod a+rw %s", script_fast);
    printf("%s \n", commande);fflush(stdout);
    system(commande);


    // -- Read status shm, init RUN and read filename -
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

    char datadir[256], commande[2048];
    hgets(stat.buf, "DATADIR", 256, datadir);

    sprintf(commande, "sudo chmod a+rw %s/%s*.fits", datadir, basename);
    printf("%s \n", commande);fflush(stdout);
    system(commande);
    sprintf(commande, "sudo chmod a+rw %s/PSETUP-at-*.log", datadir);
    printf("%s \n", commande);fflush(stdout);
    system(commande);
    sprintf(commande, "sudo chmod a+rw %s/PARSET-at-*.parset", datadir);
    printf("%s \n", commande);fflush(stdout);
    system(commande);
    sprintf(commande, "sudo chmod a+rw %s/SHELL--at-*BEAM%d.log", datadir, gpu);
    printf("%s \n", commande);fflush(stdout);
    system(commande);

    if (upload) {
        if (upload_fast) {
            sprintf(commande, "bash %s/SCRIPTS/%s_script_fast.sh", datadir, basename);
            printf("%s \n", commande);fflush(stdout);
            system(commande);   
        }
        else{
            sprintf(commande, "/data/script_pool/%s_script.sh", basename);
            rename(script, commande);
            printf("%s move in %s\n", script, commande);fflush(stdout);
            remove(script_slow);
            remove(script_fast);
        } 
    }
}
