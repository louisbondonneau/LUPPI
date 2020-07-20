/* stats_gpu.h */
#ifndef _STATS_GPU_H
#define _STATS_GPU_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cufft.h>
#include "params.h"

/* Time different operations for benchmarks */
typedef struct {
    double mean1;
    double mean2;
    double variance1;
    double variance2;
    double downsample;
    double total;
    unsigned long long nsamp_tot;
} stats_times;

/* Describes statistics params */
typedef struct {

    // Basic params
    int nthreads;
    int nblocks;
    int npts;		// Number of points on which to perform the statistics
    int *npts_host;		// Number of points on which to perform the statistics
    int *npts_gpu;		// Number of points on which to perform the statistics
    int *first;	        // Only calc mean and var for the first buffer

    // Memory for statistics 
    float *dsbuf_gpu;          // Detected 32 bit data
    float *tbuf_gpu;           // Reduced temporary data
    float *tmean_gpu;           // Temporary mean of the data
    float *tvar_gpu;            // Temporary variance of the data
    char *data_gpu;            // Scaled and packed data to be transfered to the CPU
    int *count_gpu;	       // Counts
    int *tcount_gpu;	       // Reduced temporary counts
    int *tcount_gpu2;	       // Reduced temporary counts

    float *mean_gpu;           // mean of the data
    float *var_gpu;            // variance of the data
    float *mean_host;          // mean of the data
    float *var_host;           // variance of the data

    // Benchmark
    dedispersion_times time;
} stats_setup;

#ifdef __cplusplus
extern "C" {
#endif
void init_stats(dedispersion_setup *ds, stats_setup *s);
void free_stats(stats_setup *s);
void mean_gpu(stats_setup *s);
void var_gpu(stats_setup *s);
void bitconversion(dedispersion_setup *ds, stats_setup *ss, char *dsbuf, int ichan);
#ifdef __cplusplus
}
#endif

#endif
