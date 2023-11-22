/* luppi_write_raw.c
 *
 * Write baseband data
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
#include "cpu_utils.h"

#define STATUS "NUPPSTAT"
#include "threads.h"

#include "thread_main.h"

void usage() {
    fprintf(stderr,
            "Usage: luppi_write_raw [options]\n"
            "Options:\n"
            "  -h, --help        This message\n"
            "  -d, --disk        Write raw data to disk (default no)\n"
            "  -o, --only_net    Run only net_thread\n"
            "  -b, --bands       Select which part of the band to record (0: lowest, 1: highest, 2: both)\n"
            "  -t, --test        Test mode without ssh upload\n"
            "  -f, --fasttransfer    transfer the observation using a mix of scp and rsync for max speed\n"
            "  -g, --gpu         GPUid (default 0)\n"
           );
}

/* Thread declarations */
void *net_thread(void *_up);
void *rawdisk_thread(void *args);
void *null_thread(void *args);

int main(int argc, char *argv[]) {

    static struct option long_opts[] = {
        {"help",    0, NULL, 'h'},
        {"disk",    0, NULL, 'd'},
        {"only_net",0, NULL, 'o'},
        {"bands",   0, NULL, 'b'},
        {"gpu",     0, NULL, 'g'},
        {0,0,0,0}
    };
    int opt, opti;
    int disk=0, only_net=0, cbands=0;
    int gpu = 0;
    int upload = 1;
    int upload_fast = 0;
    char basename[256], datadir[256];
    
    
    int i=1;
    while(i<argc) {
      if(strncmp(argv[i],"-g",2) == 0) { i++; sscanf(argv[i],"%d",&gpu); }
      if(strncmp(argv[i],"--gpu",2) == 0) { i++; sscanf(argv[i],"%d",&gpu); }
      i++;
    }

    
    
    
    while ((opt=getopt_long(argc,argv,"hdtfobg:",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'g':
                break;
            case 't':
                upload=0;
                break;
            case 'f':
                upload_fast = 1;
                break;
            case 'd':
                disk=1;
                break;
            case 'o':
                only_net=1;
                break;
            case 'b':
                cbands = atoi(optarg);
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
    printf("disk = %d\n",disk);
    printf("BEAM = %d\n",gpu);
    printf("gpu = %d\n",gpu%2 );
    // -- First part of the band --
    thread_args net_args;
    thread_args null_args;
    pthread_t net_thread_id;
    pthread_t disk_thread_id=0;
    status stat;
    char psrname[256];
    char projid[256];
    char julianday[256];
    char year[32], month[32], day[32];
    printf("cpuaffinity = %d\n",4 + 4*gpu -1);
    setcpuaffinity(4 + 4*gpu -1);
    if (cbands == 0 || cbands ==2) {

        // -- thread args, start at 1 --
        thread_args_init(&net_args);
        net_args.gpu_id = gpu;
        net_args.output_buffer = 1 + 4*(net_args.gpu_id) ;
        net_args.priority = 15;

        // -- Init status shared mem --
        databuf *dbuf=NULL;
        int rv = status_attach(&stat, net_args.gpu_id);
        if (rv!=OK) {
            log_error("luppi_write_raw", "Error connecting to status shm");
            exit(1);
        }
       
        // -- Read status shm, init RUN and read filename --
        status_lock_safe(&stat);
        hgets(stat.buf, "BASENAME", 256, basename);
        hgets(stat.buf, "BASENAME", 256, psrname);
        hgets(stat.buf, "DATADIR", 256, datadir);
        printf(basename);
        printf(datadir);fflush(stdout);
        hgets(stat.buf, "STT_JDAY", 256, julianday);
        hgets(stat.buf, "PROJID", 256, projid);
        strncpy(year, julianday+1, 4);
        strncpy(month, julianday+5, 2);
        strncpy(day, julianday+7, 9);
        printf("%.4s/%.2s/%.2s  %.4s\n", year, month, day, projid);

        // -- preparing upload scripts --
        char script[128], script_slow[128], script_fast[128], string[1024], filename[128], hostname[128], strlog[128], logname[128], logdir[128];
        char source[256], mode[] = "RAW", path[256], commande[2048], date[128];
        hgets(stat.buf, "SRC_NAME", 256, source);
        hgets(stat.buf, "DATADIR", 256, datadir);
        gethostname(hostname, 127);
        sprintf(logdir, "%s/BEAM%d_%d", datadir, gpu, (int)getpid());
        sprintf(filename, "%s/%s-%s.log", logdir, basename, hostname);
        sprintf(logname,"%s_%d.log", LOG_FILENAME, (int)getpid());
        sprintf(script,"%s/SCRIPTS/%s_script.sh", datadir, basename);
        sprintf(script_slow,"%s/SCRIPTS/%s_script_slow.sh", datadir, basename);
        sprintf(script_fast,"%s/SCRIPTS/%s_script_fast.sh", datadir, basename);
        FILE *pfo_slow;
        FILE *pfo_fast;

        if((pfo_slow=fopen(script_slow,"w"))!=NULL) {
        if((pfo_fast=fopen(script_fast,"w"))!=NULL) {
            //######## mkdir DATA/SOURCE
            //sprintf(string,"mkdir %s/DATA/%s 2>&1\n",datadir,source); fputs(string,pfo_slow); fputs(string,pfo_fast);
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
            sprintf(string, "mv %s/%s*.raw %s/ 2>&1\n", datadir, basename, path); fputs(string,pfo_slow); fputs(string,pfo_fast);
            sprintf(string, "cp -p %s/PSETUP-at-%s*.log %s/ 2>&1\n", datadir, obs_time, logdir); fputs(string,pfo_slow); fputs(string,pfo_fast);
            sprintf(string, "cp -p %s/PARSET-at-%s*.parset %s/ 2>&1\n", datadir, obs_time, logdir); fputs(string,pfo_slow); fputs(string,pfo_fast);
            sprintf(string, "mv %s/SHELL--at-%s*BEAM%d.log %s/ 2>&1\n", datadir, obs_time, gpu, logdir); fputs(string,pfo_slow); fputs(string,pfo_fast);
            sprintf(string, "tar -zcvf %s-%s.tar -C %s BEAM%d_%d/ 2>&1\n", basename, hostname, datadir, gpu, (int)getpid()); fputs(string,pfo_slow); fputs(string,pfo_fast);
            sprintf(string, "mv %s-%s.tar %s/ 2>&1\n", basename, hostname, path); fputs(string,pfo_slow); fputs(string,pfo_fast);
    
    
            //######## upload files on databf2
            sprintf(date, "date \'+%%Y-%%m-%%d %%H:%%M:%%S\'");
    
            sprintf(string, "echo \"$(%s) $USER %s Start\" >> /data/upload_file.log\n", date, script_slow); fputs(string,pfo_slow);
            sprintf(string, "echo \"$(%s) $USER %s Start\" >> /data/upload_file.log\n", date, script_fast); fputs(string,pfo_fast);
            sprintf(string, "rsync -avt --chmod=o+r,o-wx --remove-source-files --progress --bwlimit=50000 -e \"ssh -T -c aes128-ctr -o Compression=no\"  %s/%s*  nfrplsobs@databfnfrdt:/data/nenufar-pulsar/%.4s/%.4s/%.2s/\n", path, basename, projid, year, month); fputs(string,pfo_slow);
            sprintf(string, "scp -p -o Compression=no -c aes128-ctr %s/%s*  nfrplsobs@databfnfrdt:/data/nenufar-pulsar/%.4s/%.4s/%.2s/\n", path, basename, projid, year, month); fputs(string,pfo_fast);
            sprintf(string, "rsync -avt --chmod=o+r,o-wx --remove-source-files --progress -e \"ssh -T -c aes128-ctr -o Compression=no\"  %s/%s*  nfrplsobs@databfnfrdt:/data/nenufar-pulsar/%.4s/%.4s/%.2s/\n", path, basename, projid, year, month); fputs(string,pfo_fast);
            sprintf(string, "echo \"$(%s) $USER %s Stop\" >> /data/upload_file.log\n", date, script_slow); fputs(string,pfo_slow);
            sprintf(string, "echo \"$(%s) $USER %s Stop\" >> /data/upload_file.log\n", date, script_fast); fputs(string,pfo_fast);
            //######## creating link on databf2 in /databf2/nenufar-pulsar/DATA/
            sprintf(string, "ssh nfrplsobs@databfnfrdt 'ln /data/nenufar-pulsar/%.4s/%.4s/%.2s/%s* /data/nenufar-pulsar/DATA/%s/%s/' 2>&1\n", projid, year, month, basename, source, mode); fputs(string,pfo_slow); fputs(string,pfo_fast);
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


        int port;
        get_int(stat.buf,"DATAPORT", &port, 1491);
        printf("\ndataport = %d\n", port );
        hputi4(stat.buf, "RUN", 1);
        status_unlock_safe(&stat);


        dbuf = databuf_attach(net_args.output_buffer);
        /* If attach fails, first try to create the databuf */
        if (dbuf==NULL) dbuf = databuf_create(DATABUF_NB, (int64_t)DATABUF_SIZE*1024*1024, net_args.output_buffer);
        /* If that also fails, exit */
        if (dbuf==NULL) {
            log_error("luppi_write_raw", "Error connecting to databuf net shm");
            exit(1);
        }
        databuf_clear(dbuf);

        /* Launch net thread */
        rv = pthread_create(&net_thread_id, NULL, net_thread, (void *)&net_args);
        if (rv) { 
            log_error("luppi_write_raw", "Error creating net thread");
            perror("pthread_create");
            exit(1);
        }

        /* Launch raw disk (or null) thread */
        thread_args_init(&null_args);
        null_args.input_buffer = net_args.output_buffer;
        null_args.gpu_id = gpu; null_args.priority = 20;
        null_args.output_buffer = 2 + 4*(null_args.gpu_id);

        if (only_net==0) {
            if (disk) rv = pthread_create(&disk_thread_id, NULL, rawdisk_thread, (void *)&null_args);
            else rv = pthread_create(&disk_thread_id, NULL, null_thread, (void *)&null_args);
            if (rv) { 
                log_error("luppi_write_raw", "Error creating rawdisk/null thread");
                exit(1);
            }
        }
    }

    /*
    // -- Use second part of the band --
    thread_args net_args2;
    thread_args null_args2;
    pthread_t net_thread_id2;
    pthread_t disk_thread_id2=0;
    status stat2;
    if(cbands == 1 || cbands ==2) {
        // -- thread args, start at 1 --
        thread_args_init(&net_args2);
        net_args2.output_buffer = 5; net_args2.gpu_id = 2; net_args2.priority = 15;

        // -- Init status shared mem --
        databuf *dbuf=NULL;
        int rv = status_attach(&stat2, net_args2.gpu_id);
        if (rv!=OK) {
            log_error("nuppi_write_raw", "Error connecting to status shm");
            exit(1);
        }
       
        // -- Read status shm, init RUN and read filename --
        status_lock_safe(&stat2);
        hgets(stat2.buf, "BASENAME", 256, basename);
        hputi4(stat2.buf, "RUN", 1);
        status_unlock_safe(&stat2);


        dbuf = databuf_attach(net_args2.output_buffer);
        // If attach fails, first try to create the databuf 
        if (dbuf==NULL) dbuf = databuf_create(DATABUF_NB, (int64_t)DATABUF_SIZE*1024*1024, net_args2.output_buffer);
        // If that also fails, exit 
        if (dbuf==NULL) {
            log_error("nuppi_write_raw", "Error connecting to databuf net shm");
            exit(1);
        }
        databuf_clear(dbuf);

        // Launch net thread 
        rv = pthread_create(&net_thread_id2, NULL, net_thread, (void *)&net_args2);
        if (rv) { 
            log_error("nuppi_write_raw", "Error creating net thread");
            perror("pthread_create");
            exit(1);
        }

        // Launch raw disk (or null) thread 
        thread_args_init(&null_args2);
        null_args2.input_buffer = net_args2.output_buffer;
        null_args2.output_buffer = 6; null_args2.gpu_id = 2; null_args2.priority = 15;

        pthread_t disk_thread_id2=0;
        if (only_net==0) {
            if (disk) rv = pthread_create(&disk_thread_id2, NULL, rawdisk_thread, (void *)&null_args2);
            else rv = pthread_create(&disk_thread_id2, NULL, null_thread, (void *)&null_args2);
            if (rv) { 
                log_error("nuppi_write_raw", "Error creating rawdisk/null thread");
                exit(1);
            }
        }

    }
    */
    // -- Run Signal --
    run=1;
    signal(SIGINT, cc);


    /* Wait for end */
    while (run) { 
        sleep(1);
        // Read the RUN keyword in the first status shm, to look for a stop order
        if(cbands == 0 || cbands ==2) {
        status_lock_safe(&stat);
        hgeti4(stat.buf, "RUN", &run);
        status_unlock_safe(&stat);

        if (run == 0) log_info("luppi_write_raw", "GPU #0 : Caught RUN = 0 signal for end of observation");
        }   
    /*
        if(cbands == 1 || cbands ==2) {
        status_lock_safe(&stat2);
        hgeti4(stat2.buf, "RUN", &run);
        status_unlock_safe(&stat2);
        
        if (run == 0) log_info("nuppi_write_raw", "GPU #1 : Caught RUN = 0 signal for end of observation");
        }   

        //if (null_args.finished || null_args2.finished) run=0;
    */
    }


    if(cbands == 0 || cbands ==2) {
        // -- First cancel threads -- 
        if (disk_thread_id) pthread_cancel(disk_thread_id);
        pthread_cancel(net_thread_id);

        // -- Then kill threads -- 
        if (disk_thread_id) pthread_kill(disk_thread_id,SIGINT);
        pthread_kill(net_thread_id,SIGINT);

        // -- Finally join --
        if (disk_thread_id) {
            pthread_join(disk_thread_id,NULL);
            log_info("luppi_write_raw", "Joined disk thread");
        }
        pthread_join(net_thread_id,NULL);
        log_info("luppi_write_raw", "Joined net thread");

        // -- Destroy args --
        thread_args_destroy(&net_args);
        thread_args_destroy(&null_args);
    } 

    char commande[2048];
    char script[128], script_slow[128], script_fast[128];
    hgets(stat.buf, "DATADIR", 256, datadir);
    sprintf(script,"%s/SCRIPTS/%s_script.sh", datadir, basename);
    sprintf(script_slow,"%s/SCRIPTS/%s_script_slow.sh", datadir, basename);
    sprintf(script_fast,"%s/SCRIPTS/%s_script_fast.sh", datadir, basename);

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
