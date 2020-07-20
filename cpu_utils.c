#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "cpu_utils.h"
#include "logging.h"

int setcpuaffinity (int affinity) {
  cpu_set_t cpuset, cpuset_orig;
  sched_getaffinity(0, sizeof(cpu_set_t), &cpuset_orig);
  CPU_ZERO(&cpuset);
  CPU_SET(affinity, &cpuset);
  char strlog[128];
  sprintf(strlog,"affinity = %d", affinity);
  printf("sched_setaffinity:  %s\n", strlog);
  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset)<0) {
    //char strlog[128];
    //sprintf(strlog,"affinity = %d", affinity);
    printf("error sched_setaffinity:  %s\n", strlog);
    log_error("sched_setaffinity", strlog);
    pthread_exit(NULL);
  }
  return(EXIT_SUCCESS);
}

int setabspriority (int priority) {
  struct sched_param readabspriority;
  readabspriority.sched_priority=priority;
  if (sched_setscheduler (0,  SCHED_FIFO, &readabspriority)<0) {
    char strlog[128];
    sprintf(strlog,"priority = %d", priority);
    log_error("sched_setscheduler", strlog);
    pthread_exit(NULL);
  }
  return(EXIT_SUCCESS);
}
/*
int setpinnedmemory () {
  if ( mlockall(MCL_CURRENT|MCL_FUTURE) < 0) {
    perror("mlockall");
    exit(EXIT_FAILURE);
  }
    return(EXIT_SUCCESS);
}*/

int setniceness(int niceness) {
  if (setpriority(PRIO_PROCESS, 0, niceness)<0) {
    char strlog[128];
    sprintf(strlog,"niceness = %d", niceness);
    log_error("set_priority", strlog);
    pthread_exit(NULL);
  }
  return(EXIT_SUCCESS);
}


/*
 * Taken from asp software
 */

/* Returns t1-t0 in seconds */
double timediff(struct timespec t0, struct timespec t1) {
  time_t s_diff;
  long ns_diff;
  double diff;
  s_diff = t1.tv_sec - t0.tv_sec;
  ns_diff = t1.tv_nsec - t0.tv_nsec;
  diff = ((double)s_diff) + (1e-9)*((double)ns_diff);
  return(diff);
}


