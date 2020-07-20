#ifndef _THREAD_ARGS_H
#define _THREAD_ARGS_H
/* Generic thread args type with input/output buffer
 * id numbers.  Not all threads have both a input and a
 * output.
 */
#include <pthread.h>
#include <sys/time.h>
#include <math.h>
typedef struct {
    int input_buffer;
    int output_buffer;
    int priority;
    int finished;
    int gpu_id;
    char string[64];
    pthread_cond_t finished_c;
    pthread_mutex_t finished_m;
} thread_args;

// -- Mutex for access to the psrfits template --
pthread_mutex_t lock_psrfits_tpl;

void thread_args_init(thread_args *a);
void thread_args_destroy(thread_args *a);
void thread_set_finished(thread_args *a);
int thread_finished(thread_args *a, float timeout_sec);
#endif
