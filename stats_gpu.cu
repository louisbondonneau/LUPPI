/* downsample_gpu.cu
 * Detection/downsampling in GPU/CUDA
 * Paul Demorest, 2009/10
 */
#include <math.h>
#include "dedisperse_gpu.h"
#include "downsample_gpu.h"
#include "stats_gpu.h"
#include "logging.h"

#define NBLOCKS 128
#define NTHREADS 128

/* Returns the memory size to transfer back to the CPU */
size_t bytes2transfer(dedispersion_setup *s) {
    size_t nbytes = sizeof(char) * s->npts_per_block * s->npol / s->dsfac;
    if (s->nbits > 8) nbytes *= s->nbits/8;
    return nbytes;
}

/*
 * Reduces an array of n elts to blockSize elts
 * npts4mean is used in the final reduce stage
 * to divide the result by the number of elements to get the mean
 *
 */
template <unsigned int blockSize, typename T>
__global__ void final_reduce(T* idata, T* odata, const int n, int *npts4mean) {
    __shared__ T mean[blockSize];
    const int tid = threadIdx.x;
    int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    const int gridSize = blockSize*2*gridDim.x;

    mean[tid] = (T)0.0;


    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n) 
    {
        mean[tid] += idata[i] + idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { mean[tid] += mean[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { mean[tid] += mean[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { mean[tid] += mean[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        if (blockSize >=  64) { mean[tid] += mean[tid + 32]; __syncthreads();}
        if (blockSize >=  32) { mean[tid] += mean[tid + 16]; __syncthreads();}
        if (blockSize >=  16) { mean[tid] += mean[tid +  8]; __syncthreads();}
        if (blockSize >=   8) { mean[tid] += mean[tid +  4]; __syncthreads();}
        if (blockSize >=   4) { mean[tid] += mean[tid +  2]; __syncthreads();}
        if (blockSize >=   2) { mean[tid] += mean[tid +  1]; __syncthreads();}
    }

    // write result for this block to global mem
    if (tid == 0 && blockIdx.x==0) {
	odata[blockIdx.x] = mean[0]/(float)(*npts4mean);
	//printf("final reduce=%f\n", (float) odata[blockIdx.x]);
    }
}

template <unsigned int blockSize, typename T>
__global__ void reduce(T* idata, T* odata, const int n) {
    __shared__ T mean[blockSize];
    const int tid = threadIdx.x;
    int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    const int gridSize = blockSize*2*gridDim.x;

    mean[tid] = (T)0.0;


    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n) 
    {
        mean[tid] += idata[i] + idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { mean[tid] += mean[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { mean[tid] += mean[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { mean[tid] += mean[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        if (blockSize >=  64) { mean[tid] += mean[tid + 32]; __syncthreads();}
        if (blockSize >=  32) { mean[tid] += mean[tid + 16]; __syncthreads();}
        if (blockSize >=  16) { mean[tid] += mean[tid +  8]; __syncthreads();}
        if (blockSize >=   8) { mean[tid] += mean[tid +  4]; __syncthreads();}
        if (blockSize >=   4) { mean[tid] += mean[tid +  2]; __syncthreads();}
        if (blockSize >=   2) { mean[tid] += mean[tid +  1]; __syncthreads();}
    }

    if (tid == 0) {odata[blockIdx.x] = mean[0]; }
}


template <unsigned int blockSize, typename T>
__global__ void variance(T *idata, float *odata, const float *mean, unsigned int n) {
    __shared__ float var[blockSize];
    const int tid = threadIdx.x;
    int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    const int gridSize = blockSize*2*gridDim.x;

    var[tid] = 0.0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
	var[tid] += powf((idata[i]-mean[0]),2.0) + powf((idata[i+blockSize]-mean[0]),2.0);
	i += gridSize;
    }
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { var[tid] += var[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { var[tid] += var[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { var[tid] += var[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        if (blockSize >=  64) { var[tid] += var[tid + 32]; __syncthreads();}
        if (blockSize >=  32) { var[tid] += var[tid + 16]; __syncthreads();}
        if (blockSize >=  16) { var[tid] += var[tid +  8]; __syncthreads();}
        if (blockSize >=   8) { var[tid] += var[tid +  4]; __syncthreads();}
        if (blockSize >=   4) { var[tid] += var[tid +  2]; __syncthreads();}
        if (blockSize >=   2) { var[tid] += var[tid +  1]; __syncthreads();}
    }

    // write result for this block to global mem 
    if (tid == 0) odata[blockIdx.x] = var[0];
}

/* XXX: NOT WORKING YET */
template <unsigned int blockSize, typename T>
__global__ void reduce2(T *idata, T *odata, const float *mean, const float *var, int *count, unsigned int n) {
    __shared__ T sum[blockSize];
    __shared__ int cnt[blockSize];
    const int tid = threadIdx.x;
    int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    const int gridSize = blockSize*2*gridDim.x;

    const float rms = sqrt(var[0]/n);
    const float min = mean[0] - 3 * rms;
    const float max = mean[0] + 3 * rms;
    float data1, data2;

    sum[tid] = (T)0.0;
    cnt[tid] = 0;
    
    //if (threadIdx.x ==0) printf("thread = %d  %f %f %f\n", threadIdx.x, rms, min, max);

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        data1 = idata[i]; data2 = idata[i+blockSize];
    //if (threadIdx.x ==0) printf("thread = %d  %f < %f < %f\n", threadIdx.x, min, data1, max);
        if (min < data1 && data1 < max) {sum[tid] += data1; cnt[tid]++;}
        if (min < data2 && data2 < max) {sum[tid] += data2; cnt[tid]++;}
	i += gridSize;
    }
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sum[tid] += sum[tid + 256]; cnt[tid] += cnt[tid+256];} __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sum[tid] += sum[tid + 128]; cnt[tid] += cnt[tid+128];} __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sum[tid] += sum[tid +  64]; cnt[tid] += cnt[tid+ 64];} __syncthreads(); }

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        if (blockSize >=  64) { sum[tid] += sum[tid + 32]; cnt[tid] += cnt[tid + 32];  __syncthreads();}
        if (blockSize >=  32) { sum[tid] += sum[tid + 16]; cnt[tid] += cnt[tid + 16];  __syncthreads();}
        if (blockSize >=  16) { sum[tid] += sum[tid +  8]; cnt[tid] += cnt[tid +  8];  __syncthreads();}
        if (blockSize >=   8) { sum[tid] += sum[tid +  4]; cnt[tid] += cnt[tid +  4];  __syncthreads();}
        if (blockSize >=   4) { sum[tid] += sum[tid +  2]; cnt[tid] += cnt[tid +  2];  __syncthreads();}
        if (blockSize >=   2) { sum[tid] += sum[tid +  1]; cnt[tid] += cnt[tid +  1];  __syncthreads();}
    }

    // write result for this block to global mem 
    if (tid == 0) {odata[blockIdx.x] = sum[0]; count[blockIdx.x] = cnt[0];}
    //if (tid == 0) {odata[blockIdx.x] = sum[0]; count[blockIdx.x] = cnt[0];printf("%d %f %f %d %f %d\n", blockIdx.x, mean[0], var[0], n, odata[blockIdx.x], count[blockIdx.x]);}

}

/* XXX: NOT WORKING YET */
template <unsigned int blockSize, typename T>
__global__ void variance2(T *idata, float *odata, float *mean, float *variance, unsigned int n) {
    __shared__ float var[blockSize];
    const int tid = threadIdx.x;
    int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    const int gridSize = blockSize*2*gridDim.x;

    const float rms = sqrt(variance[0]/n);
    const float min = mean[0] - 3 * rms;
    const float max = mean[0] + 3 * rms;
    float data1, data2;

    var[tid] = 0.0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        data1 = idata[i]; data2 = idata[i+blockSize];
        if (min < data1 && data1 < max) {var[tid] += powf((data1-mean[0]),2.0);}
        if (min < data2 && data2 < max) {var[tid] += powf((data2-mean[0]),2.0);}
	i += gridSize;
    }
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { var[tid] += var[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { var[tid] += var[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { var[tid] += var[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        if (blockSize >=  64) { var[tid] += var[tid + 32]; __syncthreads();}
        if (blockSize >=  32) { var[tid] += var[tid + 16]; __syncthreads();}
        if (blockSize >=  16) { var[tid] += var[tid +  8]; __syncthreads();}
        if (blockSize >=   8) { var[tid] += var[tid +  4]; __syncthreads();}
        if (blockSize >=   4) { var[tid] += var[tid +  2]; __syncthreads();}
        if (blockSize >=   2) { var[tid] += var[tid +  1]; __syncthreads();}
    }

    // write result for this block to global mem 
    if (tid == 0) odata[blockIdx.x] = var[0];
}

/* Remove baseline, scale and pack the data */
__global__ void remove_baseline_scale(float *idata, char *odata, float *mean, float *var, const int nbits, size_t n) {
    const int nt = blockDim.x * gridDim.x;
    const int tId = blockIdx.x * blockDim.x + threadIdx.x;

    int idat;

    const float rms = sqrt(*var/n); 
    const int max = powf(2, nbits) - 1;
    const float tmp = 3 * rms - *mean;
    float scale = 6 * rms / powf(2, nbits); 
    if (scale==0.0) scale = 1.0;

    for (int i=tId; i<n; i+=nt) {
	idat = __float2int_rn((idata[i]+tmp)/scale);

	if (idat < 0) idat = 0;
	if (idat > max) idat = max;

	odata[i] = (unsigned char) idat; 
    }
}




extern "C"
void init_stats(dedispersion_setup *ds, stats_setup *s) {
    s->nthreads = NTHREADS;
    s->nblocks = NBLOCKS;

    // -- Allocate memory --
    printf("Stats memory = %.1f MB\n", ((2*s->nblocks + 2 * ds->nchan) * sizeof(float) + sizeof(int) + bytes2transfer(ds))  / (1024.*1024.));
    cudaMalloc((void**)&s->data_gpu, bytes2transfer(ds));

    cudaMalloc((void**)&s->tbuf_gpu, s->nblocks * sizeof(float));
    cudaMalloc((void**)&s->tcount_gpu, s->nblocks * sizeof(int));

    cudaMalloc((void**)&s->mean_gpu, ds->nchan * sizeof(float));
    cudaMalloc((void**)&s->var_gpu, ds->nchan * sizeof(float));
    cudaMalloc((void**)&s->count_gpu, ds->nchan * sizeof(int));
    cudaMalloc((void**)&s->tmean_gpu, sizeof(float));
    cudaMalloc((void**)&s->tvar_gpu, sizeof(float));
    cudaMalloc((void**)&s->tcount_gpu2, sizeof(int));
    cudaMalloc((void**)&s->npts_gpu, sizeof(int));

    cudaMallocHost((void**)&s->mean_host, ds->nchan * sizeof(float));
    cudaMallocHost((void**)&s->var_host, ds->nchan * sizeof(float));
    cudaMallocHost((void**)&s->npts_host, sizeof(int));

    s->first = (int *)malloc(ds->nchan * sizeof(int));

    // Check for errors
    cudaThreadSynchronize();
    printf("init_stats cuda_err='%s'\n", cudaGetErrorString(cudaGetLastError()));

    // Initialize
    memset(s->first, 1, ds->nchan * sizeof(int));
    s->npts_host[0] = s->npts;
    cudaMemcpy(s->npts_gpu, s->npts_host, sizeof(int), cudaMemcpyHostToDevice);
}

extern "C"
void free_stats(stats_setup *s) {
    cudaFree(s->data_gpu);
    cudaFree(s->tbuf_gpu);
    cudaFree(s->tcount_gpu);
    cudaFree(s->mean_gpu);
    cudaFree(s->tmean_gpu);
    cudaFree(s->var_gpu);
    cudaFree(s->count_gpu);

    cudaFree(s->mean_host);
    cudaFree(s->var_host);
}

/* Performs a mean calculation in two steps
 *
 */
extern "C"
void mean_gpu(stats_setup *s) {

    reduce<NTHREADS, float><<<s->nblocks, s->nthreads>>> (s->dsbuf_gpu, s->tbuf_gpu, s->npts);
    final_reduce<NBLOCKS/2, float><<<1, s->nblocks/2>>> (s->tbuf_gpu, s->tmean_gpu, s->nblocks, s->npts_gpu); 
}

extern "C"
void var_gpu(stats_setup *s) {

    variance<NTHREADS, float><<<s->nblocks, s->nthreads>>> (s->dsbuf_gpu, s->tbuf_gpu, s->tmean_gpu, s->npts);
    reduce<NBLOCKS/2, float><<<1, s->nblocks/2>>> (s->tbuf_gpu, s->tvar_gpu, s->nblocks);
}

extern "C"
void mean_gpu_pass2(stats_setup *s, int ichan) {

    reduce2<NTHREADS, float><<<s->nblocks, s->nthreads>>> (s->dsbuf_gpu, s->tbuf_gpu, s->tmean_gpu, s->tvar_gpu, s->tcount_gpu, s->npts);
    reduce<NBLOCKS/2, int><<<1, s->nblocks/2>>> (s->tcount_gpu, s->tcount_gpu2, s->nblocks);
    cudaThreadSynchronize();
    cudaMemcpy(&s->count_gpu[ichan], s->tcount_gpu2, sizeof(int), cudaMemcpyDeviceToDevice);
    //reduce<NBLOCKS/2, float><<<1, s->nblocks/2>>> (s->tbuf_gpu, s->tmean_gpu, s->nblocks, 128);
    final_reduce<NBLOCKS/2, float><<<1, s->nblocks/2>>> (s->tbuf_gpu, s->tmean_gpu, s->nblocks, &s->count_gpu[ichan]);
    cudaThreadSynchronize();
}

extern "C"
void var_gpu_pass2(stats_setup *s, int ichan) {

    variance2<NTHREADS, float><<<s->nblocks, s->nthreads>>> (s->dsbuf_gpu, s->tbuf_gpu, s->tmean_gpu, s->tvar_gpu, s->count_gpu[ichan]);
    reduce<NBLOCKS/2, float><<<1, s->nblocks/2>>> (s->tbuf_gpu, s->tvar_gpu, s->nblocks);
}

extern "C"
void bitconversion(dedispersion_setup *ds, stats_setup *ss, char *dsbuf, int ichan) {

    const size_t dsbuf_size = bytes2transfer(ds);
    //printf("LB > %d bits data in bitconversion\n", ds->nbits);
    if (ds->nbits==8 || ds->nbits==32) {
        cudaMemcpy(dsbuf, ds->dsbuf_gpu, dsbuf_size, cudaMemcpyDeviceToHost);
	printf("GD > transfer %d bytes to host\n", dsbuf_size);

    } else if (ds->nbits==4) {
        ss->dsbuf_gpu = (float *)ds->dsbuf_gpu;

        if (ss->first[ichan]) {
	    char strlog[256];
	    mean_gpu(ss);
	    var_gpu(ss);
	    //mean_gpu_pass2(ss, ichan);
	    //var_gpu_pass2(ss, ichan);
	    cudaMemcpy(&ss->mean_gpu[ichan], ss->tmean_gpu, sizeof(float), cudaMemcpyDeviceToDevice);
	    cudaMemcpy(&ss->var_gpu[ichan], ss->tvar_gpu, sizeof(float), cudaMemcpyDeviceToDevice);
            cudaThreadSynchronize();
	    ss->first[ichan] = 0;
	    cudaMemcpy(ss->mean_host, &ss->mean_gpu[ichan], sizeof(float), cudaMemcpyDeviceToHost);
	    cudaMemcpy(ss->var_host, &ss->var_gpu[ichan], sizeof(float), cudaMemcpyDeviceToHost);
            cudaThreadSynchronize();
	    //sprintf(strlog, "Compute stats for chan %d: mean = %f   var = %f", ichan, ss->mean_host[0], ss->var_host[0]);
	    //log_info("stats_gpu", strlog);
	    //printf("ichan %d   mean = %f   var = %f\n", ichan, ss->mean_host[0], ss->var_host[0]);
	}

        remove_baseline_scale<<<ss->nblocks, ss->nthreads>>>(ss->dsbuf_gpu, ss->data_gpu, &ss->mean_gpu[ichan], &ss->var_gpu[ichan], ds->nbits, ss->npts);
        cudaMemcpy(dsbuf, ss->data_gpu, dsbuf_size, cudaMemcpyDeviceToHost);
    } else {
        fprintf(stderr, "Nbits = %d not supported.\nExiting...\n", ds->nbits);
        exit(-1);
    }
}

