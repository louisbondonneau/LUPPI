/* 
 *  Adapted from guppi_status.c
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
//#include <semaphore.h>
#include <sys/sem.h>

#include "status.h"
#include "logging.h"

int status_attach(status *s, int shm_id) {
    char strlog[128];
    printf("status_attach\n");
    /* Get shared mem id (creating it if necessary) */
    s->shmid = shmget(STATUS_KEY + shm_id - 1, STATUS_SIZE, 0666 | IPC_CREAT);

    printf("status_attach2\n");
    sprintf(strlog,"Get shared mem id #%d key=0x%08x", shm_id, (STATUS_KEY + shm_id - 1));
    log_error("databuf_attach", strlog);
    if (s->shmid==-1) { 
        log_error("status_attach", "shmget error");
        return(ERR_SYS);
    }

    printf("attach to the segment\n");
    /* Now attach to the segment */
    s->buf = shmat(s->shmid, NULL, 0);
    if (s->buf == (void *)-1) {
        log_error("status_attach", "shmat error");
        return(ERR_SYS);
    }

    key_t key = STATUS_SEM_KEY;

    printf("Initialize semaphore\n");
    // -- Initialize semaphore --
    if ((s->lock = semget(key, 1, 0666 | IPC_CREAT)) == -1) {
        log_error("status_attach", "semget failed");
	return(-1);
    }

    printf("Init buffer with status\n");
    /* Init buffer if needed */
    status_chkinit(s);

    printf("return(OK)\n");
    return(OK);
}

int status_detach(status *s) {
    int rv = shmdt(s->buf);
    if (rv!=0) {
        log_error("status_detach", "shmdt error");
        return(ERR_SYS);
    }
    s->buf = NULL;
    return(OK);
}

/* TODO: put in some (long, ~few sec) timeout */
int status_lock(status *s) {
    //return(sem_wait(s->lock));

    //printf("Lock\n");
    struct sembuf op;
    op.sem_num = 0;
    op.sem_op = -1; /* P() */
    op.sem_flg = SEM_UNDO;
    //op.sem_flg = 0;
    //printf("ICIIIII semid= %d\n",s->lock);
    
    if (semop(s->lock, &op, 1) == -1) {
       log_error("status_lock", "semop failed");
        return(-1);
    }
    return(OK);
}

int status_unlock(status *s) {
    struct sembuf op;
    op.sem_num = 0;
    op.sem_op = 1; /* V() */
    op.sem_flg = SEM_UNDO;
    if (semop(s->lock, &op, 1) == -1) {
        log_error("status_unlock", "semop failed");
        return(-1);
    }
    return(OK);
}

/* Return pointer to END key */
char *status_find_end(char *buf) {
    /* Loop over 80 byte cards */
    int offs;
    char *out=NULL;
    for (offs=0; offs<STATUS_SIZE; offs+=STATUS_CARD) {
        if (strncmp(&buf[offs], "END", 3)==0) { out=&buf[offs]; break; }
    }
    return(out);
}

/* So far, just checks for existence of "END" in the proper spot */
void status_chkinit(status *s) {

    /* Lock */
    status_lock(s);

    /* If no END, clear it out */
    if (status_find_end(s->buf)==NULL) {
        /* Zero bufer */
        memset(s->buf, 0, STATUS_SIZE);
        /* Fill first card w/ spaces */
        memset(s->buf, ' ', STATUS_CARD);
        /* add END */
        strncpy(s->buf, "END", 3);
    }

    /* Unlock */
    status_unlock(s);
}

/* Clear out status buf */
void status_clear(status *s) {

    /* Lock */
    status_lock(s);

    /* Zero bufer */
    memset(s->buf, 0, STATUS_SIZE);
    /* Fill first card w/ spaces */
    memset(s->buf, ' ', STATUS_CARD);
    /* add END */
    strncpy(s->buf, "END", 3);

    /* Unlock */
    status_unlock(s);
}
