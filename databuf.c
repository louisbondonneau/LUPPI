/* databuf.c
 *
 * Routines for creating and accessing main data transfer
 * buffer in shared memory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#define _GNU_SOURCE 1
#define __USE_GNU
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <errno.h>
#include <time.h>

#include "fitshead.h"
#include "databuf.h"
#include "logging.h"
#include "status.h"

databuf *databuf_create(int n_block, int64_t block_size, int databuf_id) {

    char strlog[128];
    printf("databuf_create block_size=%lld",block_size);fflush(stdout);

    /* Calc databuf size */
    const size_t header_size = STATUS_SIZE;
    size_t struct_size = sizeof(databuf);
    struct_size = 8192 * (1 + struct_size/8192); /* round up */
    size_t databuf_size = ((size_t)block_size+header_size) * n_block + struct_size;

    /* Get shared memory block, error if it already exists */
    int shmid;
    shmid = shmget(DATABUF_KEY + databuf_id - 1, databuf_size, 0666 | IPC_CREAT | IPC_EXCL);
    if (shmid==-1) {
        log_error("databuf_create", "shmget error");
        return(NULL);
    }
    sprintf(strlog,"databuf_create shmget info for id #%d key=0x%08x", databuf_id, (DATABUF_KEY + databuf_id - 1));
    log_info("databuf_attach", strlog);
    /* Attach */
    databuf *d;
    d = shmat(shmid, NULL, 0);
    if (d==(void *)-1) {
        log_error("databuf_create", "shmat error");
        return(NULL);
    }

    /* Try to lock in memory */
    int rv = shmctl(shmid, SHM_LOCK, NULL);
    if (rv==-1) {
	sprintf(strlog,"Error locking shared memory : %s",strerror(errno));
        log_error("databuf_create", strlog);
        perror("shmctl");
	return(NULL);
    }

    /* Zero out memory */
    memset(d, 0, databuf_size);

    /* Fill params into databuf */
    int i;
    char end_key[81];
    memset(end_key, ' ', 80);
    strncpy(end_key, "END", 3);
    end_key[80]='\0';
    d->shmid = shmid;
    d->semid = 0;
    d->n_block = n_block;
    d->struct_size = struct_size;
    d->block_size = (size_t)block_size;
    d->header_size = header_size;
    sprintf(d->data_type, "unknown");
    for (i=0; i<n_block; i++) { 
        memcpy(databuf_header(d,i), end_key, 80); 
    }
    
    // -- Get semaphores set up --
    d->semid = semget(DATABUF_KEY + databuf_id - 1, n_block, 0666 | IPC_CREAT);
    if (d->semid==-1) { 
        sprintf(strlog,"Error creating semaphores for GPU#%d key=%d",databuf_id, DATABUF_KEY + databuf_id - 1);
        log_error("databuf_create_sem", strlog);
        return(NULL);
    }

    // -- Init semaphores to 0 --
    union semun arg;
    arg.array = (unsigned short *)malloc(sizeof(unsigned short)*n_block);
    memset(arg.array, 0, sizeof(unsigned short)*n_block);
    rv = semctl(d->semid, 0, SETALL, arg);
    free(arg.array);

    return(d);
}

void databuf_clear(databuf *d) {

    /* Zero out semaphores */
    union semun arg;
    arg.array = (unsigned short *)malloc(sizeof(unsigned short)*d->n_block);
    memset(arg.array, 0, sizeof(unsigned short)*d->n_block);
    semctl(d->semid, 0, SETALL, arg);
    free(arg.array);

    /* Clear all headers */
    int i;
    for (i=0; i<d->n_block; i++) {
        fitsbuf_clear(databuf_header(d, i));
    }

}

void fitsbuf_clear(char *buf) {
    char *end, *ptr;
    end = ksearch(buf, "END");
    if (end!=NULL) {
        for (ptr=buf; ptr<=end; ptr+=80) memset(ptr, ' ', 80);
    }
    memset(buf, ' ' , 80);
    strncpy(buf, "END", 3);
}

// -- Return pointer to 'block_id' header -- 
char *databuf_header(databuf *d, int block_id) {
    return((char *)d + d->struct_size + block_id*d->header_size);
}

// -- Return pointer to 'block_id' data block --
char *databuf_data(databuf *d, int block_id) {
    return((char *)d + d->struct_size + d->n_block*d->header_size + block_id*d->block_size);
}


databuf *databuf_attach(int databuf_id) {

    int shmid;
    char strlog[128];
    databuf *d;

    // -- Get shmid --
    shmid = shmget(DATABUF_KEY + databuf_id - 1, 0, 0666);
    printf("databuf_id=%i\n",databuf_id);

    sprintf(strlog,"shmget info for id #%d key=0x%08x", databuf_id, (DATABUF_KEY + databuf_id - 1));
    log_info("databuf_attach", strlog);


    if (shmid==-1) {
        sprintf(strlog,"shmget error (%s) for id #%d key=0x%08x", strerror(errno), databuf_id, (DATABUF_KEY + databuf_id - 1));
        log_error("databuf_attach", strlog);
        return(NULL);
    }
    // -- Attach --
    d = shmat(shmid, NULL, 0);
    if (d == (void *)-1) {
        sprintf(strlog,"shmat error for id #%d key=%x", databuf_id, (unsigned char)(DATABUF_KEY + databuf_id - 1));
        log_error("databuf_attach", strlog);
        return(NULL);
    }
    return(d);
}

int databuf_detach(databuf *d) {
    int rv = shmdt(d);
    if (rv!=0) {
        log_error("databuf_detach", "shmdt error");
	return(ERR_SYS);
    }
    return(OK);
}

int databuf_block_status(databuf *d, int block_id) {
    return(semctl(d->semid, block_id, GETVAL));
}

int databuf_total_status(databuf *d) {

    /* Get all values at once */
    union semun arg;
    arg.array = (unsigned short *)malloc(sizeof(unsigned short)*d->n_block);
    memset(arg.array, 0, sizeof(unsigned short)*d->n_block);
    semctl(d->semid, 0, GETALL, arg);
    int i,tot=0;
    for (i=0; i<d->n_block; i++) tot+=arg.array[i];
    free(arg.array);
    return(tot);

}

int databuf_wait_free(databuf *d, int block_id) {
    int rv;
    struct sembuf op;
    op.sem_num = block_id;
    op.sem_op = 0;
    op.sem_flg = 0;
    struct timespec timeout;
    timeout.tv_sec = 0;
    timeout.tv_nsec = 250000000;
    rv = semtimedop(d->semid, &op, 1, &timeout);
    if (rv==-1) { 
        if (errno==EAGAIN) return(TIMEOUT);
        if (errno==EINTR) return(ERR_SYS);
        log_error("databuf_wait_free", "semop error");
        perror("semop");
        return(ERR_SYS);
    }
    return(0);
}

int databuf_wait_filled(databuf *d, int block_id) {
    /* This needs to wait for the semval of the given block
     * to become > 0, but NOT immediately decrement it to 0.
     * Probably do this by giving an array of semops, since
     * (afaik) the whole array happens atomically:
     * step 1: wait for val=1 then decrement (semop=-1)
     * step 2: increment by 1 (semop=1)
     */
    int rv;
    struct sembuf op[2];
    op[0].sem_num = op[1].sem_num = block_id;
    op[0].sem_flg = op[1].sem_flg = 0;
    op[0].sem_op = -1;
    op[1].sem_op = 1;
    struct timespec timeout;
    timeout.tv_sec = 0;
    timeout.tv_nsec = 250000000;
    rv = semtimedop(d->semid, op, 2, &timeout);
    if (rv==-1) { 
        if (errno==EAGAIN) return(TIMEOUT);
        // Don't complain on a signal interruption
        if (errno==EINTR) return(ERR_SYS);
        log_error("databuf_wait_filled", "semop error");
        perror("semop");
        return(ERR_SYS);
    }
    return(0);
}

int databuf_set_free(databuf *d, int block_id) {
    /* This function should always succeed regardless of the current
     * state of the specified databuf.  So we use semctl (not semop) to set
     * the value to zero.
     */
    int rv;
    union semun arg;
    arg.val = 0;
    rv = semctl(d->semid, block_id, SETVAL, arg);
    if (rv==-1) { 
        log_error("databuf_set_free", "semctl error");
        return(ERR_SYS);
    }
    return(0);
}

int databuf_set_filled(databuf *d, int block_id) {
    /* This function should always succeed regardless of the current
     * state of the specified databuf.  So we use semctl (not semop) to set
     * the value to one.
     */
    int rv;
    union semun arg;
    arg.val = 1;
    rv = semctl(d->semid, block_id, SETVAL, arg);
    if (rv==-1) { 
        log_error("databuf_set_filled", "semctl error");
        return(ERR_SYS);
    }
    return(0);
}
