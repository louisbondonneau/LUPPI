/* thread_main.h
 *
 * Include in any main program that will 
 * call the thread functions.
 */
#ifndef _THREAD_MAIN_H
#define _THREAD_MAIN_H

#include "thread_args.h"

/* Control-C handler */
int run=1;
void cc(int sig) { run=0; }

#endif
