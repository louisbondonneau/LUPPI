/* status.h
 *
 * Routines dealing with the guppi status shared memory
 * segment.  Info is passed through this segment using 
 * a FITS-like keyword=value syntax.
 */
#ifndef _STATUS_H
#define _STATUS_H

#include <semaphore.h>

//#include "guppi_params.h"

#define STATUS_KEY 32768
//#define STATUS_SEMID "/psrstatus" 
#define STATUS_SEM_KEY 1234567
#define STATUS_SIZE (2880*64) // FITS-style buffer
#define STATUS_CARD 80 // Size of each FITS "card"

/* Structure describes status memory area */
typedef struct {
    int shmid;   /* Shared memory segment id */
    //sem_t *lock; /* POSIX semaphore descriptor for locking */
    int lock;    /* POSIX semaphore descriptor for locking */
    char *buf;   /* Pointer to data area */
} status;

/* Return a pointer to the status shared mem area, 
 * creating it if it doesn't exist.  Attaches/creates 
 * lock semaphore as well.  Returns nonzero on error.
 */
int status_attach(status *s, int gpu_id);

/* Detach from shared mem segment */
int status_detach(status *s);

/* Lock/unlock the status buffer.  guppi_status_lock() will wait for
 * the buffer to become unlocked.  Return non-zero on errors.
 */
int status_lock(status *s);
int status_unlock(status *s);

/* Check the buffer for appropriate formatting (existence of "END").
 * If not found, zero it out and add END.
 */
void status_chkinit(status *s);

/* Clear out whole buffer */
void status_clear(status *s);

#endif
