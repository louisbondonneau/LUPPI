/* downsample_gpu.cu
 * Detection/downsampling in GPU/CUDA
 * Paul Demorest, 2009/10
 */
#include <math.h>
#include "dedisperse_gpu.h"
#include "downsample_gpu.h"

#define NBLOCKS 512
#define NTHREADS 512

/* Returns number of bytes needed for downsampling block */
size_t get_ds_bytes(const dedispersion_setup *s) {
    //if (s->nbits==8) 
    //    return sizeof(char) * s->npts_per_block * s->npol / s->dsfac;
    //else 
        return (sizeof(float) * s->npts_per_block * s->npol) / s->dsfac;
}

/* Returns number of samples per databuf */
int get_npts_per_block(const dedispersion_setup *s) {
    return s->nfft_per_block * (s->fft_len - s->overlap) / s->dsfac;
}	

/* Initialize the downsampling using values in dedispersion_setup
 * struct.  the s->dsfac field needs to be filled in.
 */
extern "C"
void init_downsample(dedispersion_setup *s) {

    // TODO: check that params satisfy any alignment requirements.

    // Allocate memory for DS results on GPU
    const size_t ds_bytes = get_ds_bytes(s);
    cudaMalloc((void**)&s->dsbuf_gpu, ds_bytes);
    printf("Downsample memory = %.1f MB\n", ds_bytes / (1024.*1024.));

    // Check for errors
    cudaThreadSynchronize();
    printf("init_downsample cuda_err='%s'\n", cudaGetErrorString(cudaGetLastError()));

    // -- Create a plan to remove the 50Hz --
    //cufftPlan1d(&s->plan2, s->fft_len*s->nfft_per_block, CUFFT_C2C, 2); // transform the 2 complex pols
}

/* "naive" version where each thread does one output sample at a time
 * If this isn't fast enough there are lots of optimizations that
 * could be done...
 */
__global__ void detect_downsample_4pol(const float2 *pol0, const float2 *pol1,
        const unsigned dsfac, const unsigned fftlen, const unsigned overlap,
        char4 *out) {

    // Dimensions
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int nvalid = fftlen - overlap;
    const int ifft = blockIdx.x;
    const int iblock = blockIdx.y;
    const int nsamp_per_block = nvalid / gridDim.y;
    const int nout_per_block = nsamp_per_block / dsfac;

    // Data pointers
    const float2 *ptr0 = pol0 + ifft*fftlen + overlap/2 
        + iblock*nsamp_per_block;
    const float2 *ptr1 = pol1 + ifft*fftlen + overlap/2 
        + iblock*nsamp_per_block;
    char4 *optr = out + ifft*nvalid/dsfac + iblock*nout_per_block;

    // Data scaling
    // This should be appropriate for input baseband data with
    // a RMS of ~20 counts.
    const float scale = (float)dsfac * 20.0;

    // Loop over data
    for (int iout=tid; iout<nout_per_block; iout+=nt) {
        float4 otmp= make_float4(0,0,0,0);
        for (int j=0; j<dsfac; j++) {
            float2 p0 = ptr0[iout*dsfac+j];
            float2 p1 = ptr1[iout*dsfac+j];
            otmp.x += p0.x*p0.x + p0.y*p0.y;
            otmp.y += p1.x*p1.x + p1.y*p1.y;
            otmp.z += p0.x*p1.x + p0.y*p1.y;
            otmp.w += p0.x*p1.y - p0.y*p1.x;
        }
        optr[iout].x = __float2int_rn(otmp.x/scale);
        optr[iout].y = __float2int_rn(otmp.y/scale);
        optr[iout].z = __float2int_rn(otmp.z/scale);
        optr[iout].w = __float2int_rn(otmp.w/scale);
    }

}

/* Same as above, except only compute total power */
__global__ void detect_downsample_1pol(const float2 *pol0, const float2 *pol1,
        const unsigned dsfac, const unsigned fftlen, const unsigned overlap,
        char *out) {

    // Dimensions
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int nvalid = fftlen - overlap;
    const int ifft = blockIdx.x;
    const int iblock = blockIdx.y;
    const int nsamp_per_block = nvalid / gridDim.y;
    const int nout_per_block = nsamp_per_block / dsfac;

    // Data pointers
    const float2 *ptr0 = pol0 + ifft*fftlen + overlap/2 
        + iblock*nsamp_per_block;
    const float2 *ptr1 = pol1 + ifft*fftlen + overlap/2 
        + iblock*nsamp_per_block;
    char *optr = out + ifft*nvalid/dsfac + iblock*nout_per_block;

    // Data scaling
    // This should be appropriate for input baseband data with
    // a RMS of ~20 counts in each poln (final 2.0 is for polns).
    const float scale = (float)dsfac * 20.0 * 2.0;

    // Loop over data
    for (int iout=tid; iout<nout_per_block; iout+=nt) {
        float otmp = 0.0;
        for (int j=0; j<dsfac; j++) {
            float2 p0 = ptr0[iout*dsfac+j];
            float2 p1 = ptr1[iout*dsfac+j];
            otmp += p0.x*p0.x + p0.y*p0.y + p1.x*p1.x + p1.y*p1.y;
        }
        optr[iout] = __float2int_rn(otmp/scale);
    }
}

/* Same as above, except only compute total power and convert to 4 bits data*/
__global__ void detect_downsample_1pol_32bits(const float2 *pol0, const float2 *pol1,
        const unsigned dsfac, const unsigned fftlen, const unsigned overlap,
        float *out) {

    // Dimensions
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int nvalid = fftlen - overlap;
    const int ifft = blockIdx.x;
    const int iblock = blockIdx.y;
    const int nsamp_per_block = nvalid / gridDim.y;
    const int nout_per_block = nsamp_per_block / dsfac;

    // Data pointers
    const float2 *ptr0 = pol0 + ifft*fftlen + overlap/2 
        + iblock*nsamp_per_block;
    const float2 *ptr1 = pol1 + ifft*fftlen + overlap/2 
        + iblock*nsamp_per_block;
    float *optr = out + ifft*nvalid/dsfac + iblock*nout_per_block;

    // Loop over data
    for (int iout=tid; iout<nout_per_block; iout+=nt) {
        float otmp = 0.0;
        for (int j=0; j<dsfac; j++) {
            float2 p0 = ptr0[iout*dsfac+j];
            float2 p1 = ptr1[iout*dsfac+j];
            otmp += p0.x*p0.x + p0.y*p0.y + p1.x*p1.x + p1.y*p1.y;
        }
        optr[iout] = otmp/(float)dsfac;
    }
}

__global__ void detect_downsample_4pol_32bits(const float2 *pol0, const float2 *pol1,
        const unsigned dsfac, const unsigned fftlen, const unsigned overlap,
        float4 *out) {

    // Dimensions
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int nvalid = fftlen - overlap;
    const int ifft = blockIdx.x;
    const int iblock = blockIdx.y;
    const int nsamp_per_block = nvalid / gridDim.y;
    const int nout_per_block = nsamp_per_block / dsfac;

    // Data pointers
    const float2 *ptr0 = pol0 + ifft*fftlen + overlap/2 
        + iblock*nsamp_per_block;
    const float2 *ptr1 = pol1 + ifft*fftlen + overlap/2 
        + iblock*nsamp_per_block;
    float4 *optr = out + ifft*nvalid/dsfac + iblock*nout_per_block;

    // Loop over data
    for (int iout=tid; iout<nout_per_block; iout+=nt) {
        float4 otmp= make_float4(0,0,0,0);
        for (int j=0; j<dsfac; j++) {
            float2 p0 = ptr0[iout*dsfac+j];
            float2 p1 = ptr1[iout*dsfac+j];
            otmp.x += p0.x*p0.x + p0.y*p0.y;
            otmp.y += p1.x*p1.x + p1.y*p1.y;
            otmp.z += p0.x*p1.x + p0.y*p1.y;
            otmp.w += p0.x*p1.y - p0.y*p1.x;
        }
	optr[iout].x = otmp.x/(float)dsfac;
	optr[iout].y = otmp.y/(float)dsfac;
	optr[iout].z = otmp.z/(float)dsfac;
	optr[iout].w = otmp.w/(float)dsfac;
    }
}


__global__ void zap_spectrum(float2 *pol0, float2 *pol1, const unsigned npts, const float bw) {

    const int nt = blockDim.x * gridDim.x;
    const int tId = blockIdx.x * blockDim.x + threadIdx.x;
    const float freq=50;
    const int bin1 = (int) (freq * npts * 1/(bw*1e6));
    const int bin2 = (npts-1)-bin1;
    for (int i=tId; i<npts; i+=nt) {
        if (i==bin1 || i==bin2) {
            pol0[i].x = (pol0[i-1].x + pol0[i+1].x)/2.;
            pol0[i].y = (pol0[i-1].y + pol0[i+1].y)/2.;
            pol1[i].x = (pol1[i-1].x + pol1[i+1].x)/2.;
            pol1[i].y = (pol1[i-1].y + pol1[i+1].y)/2.;
	}

    }

}

#if 0
extern "C"
void remove_50Hz(dedispersion_setup *s) {

    /* Forward FFT */
    cufftExecC2C(s->plan2, s->databuf0_gpu, s->databuf0_gpu, CUFFT_FORWARD);

    zap_spectrum<<<128, 256>>>(s->databuf0_gpu, s->databuf1_gpu, s->fft_len*s->nfft_per_block, (float) s->bw);    

    /* Inverse FFT */
    cufftExecC2C(s->plan2, s->databuf0_gpu, s->databuf0_gpu, CUFFT_INVERSE);

}
#endif


/* Detect / downsample data.  Assumes dedispersion results
 * are already in the GPU, as described in the dedispersion_setup
 * struct.
 */
extern "C"
void downsample(dedispersion_setup *s) {

    /* Sizes */
    const size_t ds_bytes = get_ds_bytes(s);

    /* Benchmark */
#define NT 5
    cudaEvent_t t[NT];
    int it;
    for (it=0; it<NT; it++) cudaEventCreate(&t[it]);
    it=0;

    cudaEventRecord(t[it], 0); it++;
    cudaEventRecord(t[it], 0); it++;

    /* Remove the 50Hz signal if no dedispersion was done */
    //if (s->overlap==0) remove_50Hz(s);


    /* Clear out data buf */
    cudaMemset(s->dsbuf_gpu, 0, ds_bytes);

    /* Downsample data to 32 or 8 bit data */
    dim3 gd(s->nfft_per_block, 32, 1);
    if ((s->nbits==4 || s->nbits==32) && s->npol==1)
        detect_downsample_1pol_32bits<<<gd, 64>>>(s->databuf0_gpu, s->databuf1_gpu, s->dsfac, s->fft_len, s->overlap, (float *)s->dsbuf_gpu);
    if ((s->nbits==4 || s->nbits==32) && s->npol==4)
        detect_downsample_4pol_32bits<<<gd, 64>>>(s->databuf0_gpu, s->databuf1_gpu, s->dsfac, s->fft_len, s->overlap, (float4 *)s->dsbuf_gpu);
    if (s->nbits==8 && s->npol==1) 
        detect_downsample_1pol<<<gd, 64>>>(s->databuf0_gpu, s->databuf1_gpu, s->dsfac, s->fft_len, s->overlap, (char *)s->dsbuf_gpu);
    if (s->nbits==8 && s->npol==4)
        detect_downsample_4pol<<<gd, 64>>>(s->databuf0_gpu, s->databuf1_gpu, s->dsfac, s->fft_len, s->overlap, (char4 *)s->dsbuf_gpu);
    cudaEventRecord(t[it], 0); it++;

    /* Final timer */
    cudaEventRecord(t[it], 0);
    cudaEventSynchronize(t[it]);
    cudaThreadSynchronize();

    /* Add up timers */
    float ttmp;
    it=1;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.downsample += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.transfer_to_host += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[0], t[it+1]);
    s->time.total += ttmp;

    /* Cleanup */
    for (it=0; it<NT; it++) cudaEventDestroy(t[it]);

}
