/* threads.h
 *
 * Definitions, routines common to 
 * all thread functions.
 */
#ifndef _THREADS_H
#define _THREADS_H

#include "thread_args.h"

/* SIGINT handling capability */
extern int run;
extern void cc(int sig);

/* Safe lock/unlock functions for status shared mem. */
#define status_lock_safe(s) \
    pthread_cleanup_push((void *)status_unlock, s); \
    status_lock(s);
#define status_unlock_safe(s) \
    status_unlock(s); \
    pthread_cleanup_pop(0);

/* Exit handler that updates status buffer */
#ifndef STATUS_KEY
#  define STATUS_KEY "XXXSTAT"
#  define TMP_STATUS_KEY 1
#endif
static void set_exit_status(status *s) {
    status_lock(s);
    hputs(s->buf, STATUS, "exiting");
    status_unlock(s);
}
#if TMP_STATUS_KEY
#  undef STATUS_KEY
#  undef TMP_STATUS_KEY
#endif

#endif
