#ifndef _CPU_UTILS_H
#define _CPU_UTILS_H

#define NETABSPRIORITY -1 // absolute priority of the network thread
#define NETNICENESS -10   // standard priority of the network thread
#define NETCPU 0  // on which cpu core the network thread is running

#define RAWCPU 2  // on which cpu core the dedisp thread is running

#define FOLDCPU 4  // on which cpu core the fold thread is running

// -- Timer var --
#define T_START 0
#define T_STOP 1
#define MAX_TIMERS 32

int setcpuaffinity (int affinity); 
int setabspriority (int priority); 
int setniceness(int niceness);

// -- Timer functions --
double timediff(struct timespec t0, struct timespec t1);

#endif
