
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

#include "logging.h"
#include "status.h"
#include "databuf.h"
#include "params.h"
#include "thread_main.h"

/* Thread declarations */
void *net_thread(void *args);
void *psrfits_thread(void *args);

int main(int argc, char *argv[]) {

    // -- thread args --
    thread_args net_args, disk_args;
    thread_args_init(&net_args);
    thread_args_init(&disk_args);
    net_args.output_buffer = 1;
    disk_args.input_buffer = net_args.output_buffer;
    disk_args.output_buffer = 2;
    net_args.gpu_id = disk_args.gpu_id = 1;

    // -- Signal --
    signal(SIGINT, cc);

    // -- Init shared mem --
    status stat;
    databuf *dbuf=NULL;
    int rv = status_attach(&stat, 1);
    if (rv!=OK) {
        fprintf(stderr, "Error connecting to status\n");
        exit(1);
    }
    dbuf = databuf_attach(net_args.output_buffer);
    /* If attach fails, first try to create the databuf */
    if (dbuf==NULL) 
        dbuf = databuf_create(DATABUF_NB, (int64_t)DATABUF_SIZE*1024*1024, net_args.output_buffer);
    /* If that also fails, exit */
    if (dbuf==NULL) {
        fprintf(stderr, "Error connecting to databuf\n");
        exit(1);
    }
    printf("databuf n_block = %d\n",dbuf->n_block);
    databuf_clear(dbuf);

    /* Launch net thread */
    pthread_t net_thread_id;
    rv = pthread_create(&net_thread_id, NULL, net_thread, (void *)&net_args);
    if (rv) { 
        fprintf(stderr, "Error creating net thread.\n");
        perror("pthread_create");
        exit(1);
    }

    /* Launch PSRFITS disk thread */
    pthread_t disk_thread_id;
    rv = pthread_create(&disk_thread_id, NULL, psrfits_thread, (void *)&disk_args);
    if (rv) { 
        fprintf(stderr, "Error creating PSRFITS thread.\n");
        perror("pthread_create");
        exit(1);
    }

    /* Wait for end */
    run=1;
    while (run) { 
        sleep(1); 
        if (disk_args.finished) run=0;
    }
    pthread_cancel(disk_thread_id);
    pthread_cancel(net_thread_id);
    pthread_kill(disk_thread_id,SIGINT);
    pthread_kill(net_thread_id,SIGINT);
    pthread_join(net_thread_id,NULL);
    printf("Joined net thread\n"); fflush(stdout);
    pthread_join(disk_thread_id,NULL);
    printf("Joined disk thread\n"); fflush(stdout);

    thread_args_destroy(&disk_args);

    exit(0);
}
