/* dedisperse_gpu.cu 
 * Functions for coherent dedispersion in CUDA/GPUs
 * Paul Demorest, 2009/05
 */
#include "dedisperse_gpu.h"
#include "logging.h"
#include <omp.h>
#include <unistd.h>
#include <pthread.h>

/* CUDA kernel to convert bytes to floats.  Also splits incoming
 * data into two polarizations (assuming polns are interleaved
 * in the raw data).
 */
__global__ void byte_to_float_2pol_complex(
        unsigned short *in, float2 *outx, float2 *outy,
        size_t n) {
    const int nt = blockDim.x * gridDim.x;
    const int tId = blockIdx.x * blockDim.x + threadIdx.x;
    char4 *in_8bit = (char4 *)in;
    for (int i=tId; i<n; i+=nt) {
        outx[i].x = __int2float_rn(in_8bit[i].x);
        outx[i].y = __int2float_rn(in_8bit[i].y);
        outy[i].x = __int2float_rn(in_8bit[i].z);
        outy[i].y = __int2float_rn(in_8bit[i].w);
    }
}

/* CUDA kernel for inplace complex vector (elementwise) multiplication:
 *
 *   Does a[i] *= b[i] for 0<i<n.
 *
 *   a can contain multiple vectors to be multiplied by b, each
 *   should run in its own thread block, ie:
 *     vec_mult_complex<<<n_vector,64>>>(a,b,n_element_per_vector);
 *   where a has n_element_per_vector*n_vector entries, and b
 *   has n_element.
 */
#if 1
__global__ void defaraday(float2 *a,  double freqichan, double rmfac, unsigned nelem) {
    const unsigned nelem_chunk = nelem / gridDim.y;
    const unsigned elem0 = blockIdx.y * nelem_chunk;
    const unsigned elem1 = elem0 + nelem_chunk > nelem ? nelem : elem0 + nelem_chunk;
    float2 *ptr1 = &a[blockIdx.x*nelem];
    float2 *ptr2 = &a[(blockIdx.x+1)*nelem];
    float2 tmp1, tmp2;
    double bw = (200.0/1024.0);
    //float rot = 90.0*M_PI/180.0;
    for (int i=elem0+threadIdx.x; i<elem1; i+=blockDim.x) {
        double dfreq = (blockIdx.x*nelem + (double)i)*bw/((double)nelem-(double)1);
        if (i>nelem/2) dfreq -= bw;
        double rot = rmfac*( ((double)1.0/(freqichan*freqichan))-((double)1.0/((freqichan+dfreq)*(freqichan+dfreq))) );
        //double rot = rmfac*( ((double)1.0/(43.60*43.60))-((double)1.0/((freqichan+dfreq)*(freqichan+dfreq))) );
        //if(freqichan==50.0) printf("i = %d freqichan = %lf dfreq = %lf rot = %lf nelem = %u nelem_chunk = %u blockIdx.y = %u elem0+threadIdx.x = %u blockIdx.x = %u\n", blockIdx.x*nelem + i, freqichan, dfreq, rot, nelem, nelem_chunk, blockIdx.y, elem0+threadIdx.x, blockIdx.x);
        //if(i==nelem/2) printf("i = %d freqichan = %lf dfreq = %lf rot = %lf\n", i, freqichan, dfreq, rot);
        //if(i==1+(nelem/2)) printf("i = %d freqichan = %lf dfreq = %lf rot = %lf\n", i, freqichan, dfreq, rot);
        //if(i==nelem-1) printf("i = %d freqichan = %lf dfreq = %lf rot = %lf\n", i, freqichan, dfreq, rot);
        //double rot = 90.0*M_PI/180.0;
        //rot = rot-((M_PI/2.0)*(double)(int)(rot/(M_PI/2.0)));


        tmp1.x = ptr1[i].x*cos(rot) - ptr2[i].x*sin(rot);
        tmp2.x = ptr1[i].x*sin(rot) + ptr2[i].x*cos(rot);


        tmp1.y = ptr1[i].y*cos(rot) - ptr2[i].y*sin(rot);
        tmp2.y = ptr1[i].y*sin(rot) + ptr2[i].y*cos(rot);

        ptr1[i] = tmp1;
        ptr2[i] = tmp2;
    }
}
#endif
#if 1
__global__ void vector_multiply_rot(float2 *a,  double freqichan, double rmfac, unsigned nelem) {
    const unsigned nelem_chunk = nelem / gridDim.y;
    const unsigned elem0 = blockIdx.y * nelem_chunk;
    const unsigned elem1 = elem0 + nelem_chunk > nelem ? nelem : elem0 + nelem_chunk;
    float2 *ptr = &a[blockIdx.x*nelem];
    float2 tmp;
    double bw = (200.0/1024.0);
    for (int i=elem0+threadIdx.x; i<elem1; i+=blockDim.x) {
        //double dfreq = (blockIdx.x*nelem+(double)i)*bw/(double)nelem;
        double dfreq = (blockIdx.x*nelem + (double)i)*bw/((double)nelem-(double)1);
        if (i>nelem/2) dfreq -= bw;
		//90.0*M_PI/180.0;//
        double rot = rmfac*( ((double)1.0/(freqichan*freqichan))-((double)1.0/((freqichan+dfreq)*(freqichan+dfreq))) );
		rot = 0.0*M_PI/180.0; //rot / 2.0;
		//printf("i = %d freqichan = %lf dfreq = %lf rot = %lf\n", i, freqichan, dfreq, rot);
        tmp.x = ptr[i].x*cos(rot) - ptr[i].y*sin(rot);
        tmp.y = ptr[i].y*cos(rot) + ptr[i].x*sin(rot);
        ptr[i] = tmp;
        //ptr[i].x = ptr[i].x - tmp.y;
        //ptr[i].y = ptr[i].y + tmp.x;
    }
}
__global__ void vector_multiply_complex(float2 *a, const float2 *b, unsigned nelem) {
    const unsigned nelem_chunk = nelem / gridDim.y;
    const unsigned elem0 = blockIdx.y * nelem_chunk;
    const unsigned elem1 = elem0 + nelem_chunk > nelem ? nelem : elem0 + nelem_chunk;
    float2 *ptr = &a[blockIdx.x*nelem];
    float2 tmp;
    for (int i=elem0+threadIdx.x; i<elem1; i+=blockDim.x) {
        tmp.x = ptr[i].x*b[i].x - ptr[i].y*b[i].y;
        tmp.y = ptr[i].y*b[i].x + ptr[i].x*b[i].y;
        ptr[i] = tmp;
    }
}
#endif

/* Full-stokes detection "in-place" 
 * vx and vy arrays are voltage data.  Output total power
 * terms go into vx, and cross terms in vy.  n is total number
 * of data points.
 * TODO: check signs, etc
 * Also, if we're folding on the GPU it probably makes more sense
 * to combine the two operations.
 */
__global__ void detect_4pol(float2 *vx, float2 *vy, size_t n) {
    const int nt = blockDim.x * gridDim.x;
    const int tId = blockIdx.x * blockDim.x + threadIdx.x;
    float2 pp, xp;
    for (int i=tId; i<n; i+=nt) {
        pp.x = vx[i].x*vx[i].x + vx[i].y*vx[i].y;
        pp.y = vy[i].x*vy[i].x + vy[i].y*vy[i].y;
        xp.x = vx[i].x*vy[i].x + vx[i].y*vy[i].y;
        xp.y = vx[i].x*vy[i].y - vx[i].y*vy[i].x;
        vx[i] = pp;
        vy[i] = xp;
    }
}

/* Expand the original input out so that FFTs will overlap */
void expand_overlap(dedispersion_setup *s) {
    const size_t bytes_per_sample = 4; // 8-bit complex, 2 pol
    const size_t bytes_per_fft = bytes_per_sample * s->fft_len;
    const size_t bytes_overlap = bytes_per_sample * s->overlap;
    const size_t bytes_total = bytes_per_sample * s->npts_per_block;
    size_t icount=0, ocount=0;
    for (icount=0, ocount=0;
            icount<=bytes_total-bytes_per_fft; 
            icount+=bytes_per_fft-bytes_overlap,
            ocount+=bytes_per_fft)
        cudaMemcpy(s->overlap_gpu + ocount, s->tbuf_gpu + icount,
                bytes_per_fft, cudaMemcpyDeviceToDevice);
}

/* Transfer the dedispersed data back to the main system memory
 * while simultaneously collapsing the overlap (ie, removing
 * invalid points at FFT edge).
 */
void transfer_collapse_overlap(dedispersion_setup *s) {
    /* At this point, databuf0 holds total-power terms (AA, BB)
     * and databuf1 holds poln cross-terms (Re, Im(AB)).
     */
    const int valid_samp_per_fft = s->fft_len - s->overlap;
    unsigned ifft;
    // TODO Think about the best way to organize this data for later
    // TODO Make sure we're getting the right part of the FFT
    for (ifft=0; ifft<s->nfft_per_block; ifft++) {
        // Each memcpy transfers a single FFT's worth of valid data
        // from 2 (out of 4 total) polns.  
        cudaMemcpy(s->result_host + (4*ifft+0)*valid_samp_per_fft,
                s->databuf0_gpu + ifft*s->fft_len + s->overlap/2,
                valid_samp_per_fft * sizeof(float) * 2,
                cudaMemcpyDeviceToHost); 
        cudaMemcpy(s->result_host + (4*ifft+2)*valid_samp_per_fft,
                s->databuf1_gpu + ifft*s->fft_len + s->overlap/2,
                valid_samp_per_fft * sizeof(float) * 2,
                cudaMemcpyDeviceToHost); 
    }

}

/* Fills in the freq-domain chirp, given the input params.
 * Assumes memory has already been allocated.  If fft_len has not
 * been changed, this func can be called again to change the
 * DM, freq, etc currently being applied.  Formula is taken
 * from ASP's CalcChirp.
 */
void init_chirp(dedispersion_setup *s) {

    // Alloc temporary host memory
    float2 *chirp_host;
    size_t chirp_size = sizeof(float2) * s->fft_len * s->nchan;
    // TODO check that this isn't too big
    printf("sizeof(chirp_gpu) = %d MB\n", (int)(chirp_size >> 20));
    chirp_host = (float2 *)malloc(chirp_size);

    double dmfac = s->dm*2.0*M_PI/(2.41e-10*(1.0+s->earth_z4/1.0e4));
    double band_sign = (s->bw<0.0) ? -1.0 : 1.0;

    int i, ichan;
    for (ichan=0; ichan<s->nchan; ichan++) {
        //printf("Calcul du chirp pout chan = %d\n", ichan);
        double freqichan = s->freq[ichan];
        double bwfor = s->bw;
        int fft_lenfor = s->fft_len;

        for (i=0; i<s->fft_len; i++) {
            double dfreq = (double)i * bwfor / (double)fft_lenfor;
            if (i>fft_lenfor/2) dfreq -= bwfor;

            double freqfac = dfreq / freqichan;
            freqfac = freqfac * freqfac / (freqichan + dfreq);

            double arg = band_sign * dmfac * freqfac;

            // This makes Ingrid happy, but I have no idea where this
            // particular formula comes from.
            // double taper = 1.0/sqrt(1.0 + pow(fabs(dfreq)/(0.47*s->bw),80));
            double taper = 1.0;

            chirp_host[ichan*fft_lenfor + i].x = 
                (float)(cos(arg)*taper/(double)fft_lenfor);
            chirp_host[ichan*fft_lenfor + i].y = 
                -1.0*(float)(sin(arg)*taper/(double)fft_lenfor);
            //if(ichan==s->nchan-1) printf("\n1010101010  %.20f  %.20f  %.20f\n",freqichan + dfreq, ichan*fft_lenfor + i,chirp_host[ichan*fft_lenfor + i].x,chirp_host[ichan*fft_lenfor + i].y);
        #if 0
            chirp_host[ichan*fft_lenfor + i].x = 1.0/(double)fft_lenfor;
            chirp_host[ichan*fft_lenfor + i].y = 0.0;
        #endif
        }

    }

    // Transfer the values to the gpu, free host memory
    cudaMemcpy(s->chirp_gpu[0], chirp_host, chirp_size, cudaMemcpyHostToDevice);
    for (int i=100000; i<101000; i++) {
        printf("CHIRP_CPU   %.36f\n",chirp_host[(s->nchan-1)*s->fft_len+i].x);fflush(stdout);
        }
    free(chirp_host);
    //exit(0);
}





//init_chirp_GPU_kernel
__global__ void init_chirp_kernel(int nchan, unsigned fft_len, double freq0, double bw, float2 *chirp_gpu, double band_sign, double dmfac) {
    const int stride = blockDim.x * gridDim.x;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    int ichan, i;
    //printf("Hello thread 0 index= %.5d  max = %.7d  stride = %.5d\n",index,nchan*fft_len, stride);

    for (int j=index; j<(nchan*fft_len); j+=stride) {
        ichan = (int)j/fft_len;
        i = j - ichan*fft_len;
        double freqichan = freq0+ichan*bw;
	//if (i==1) printf("ichan = %d  freq = %.10f  freq0 = %f  bw = %f\n",ichan, freqichan, freq0, bw);
        double bwfor = bw;
        int fft_lenfor = fft_len;
        double dfreq = (double)i * bwfor / (double)fft_lenfor;
        if (i>fft_lenfor/2) dfreq -= bwfor;
        double freqfac = dfreq / freqichan;
        freqfac = freqfac * freqfac / (freqichan + dfreq);

        double arg = band_sign * dmfac * freqfac;

        double taper = 1.0;
        chirp_gpu[ichan*fft_lenfor + i].x = (float)(cos(arg)*taper/(double)fft_lenfor);
        chirp_gpu[ichan*fft_lenfor + i].y = -1.0*(float)(sin(arg)*taper/(double)fft_lenfor);
    }
}


/* Fills in the freq-domain chirp, given the input params.
 * Assumes memory has already been allocated.  If fft_len has not
 * been changed, this func can be called again to change the
 * DM, freq, etc currently being applied.  Formula is taken
 * from ASP's CalcChirp.
 */
void init_chirp_GPU(dedispersion_setup *s) {

    size_t chirp_size = sizeof(float2) * s->fft_len * s->nchan;
    // TODO check that this isn't too big
    printf("sizeof(chirp_gpu) = %d MB\n", (int)(chirp_size >> 20));

    double dmfac = s->dm*2.0*M_PI/(2.41e-10*(1.0+s->earth_z4/1.0e4));
    printf("s->earth_z4/1.0e4 = %f \n",s->earth_z4/1.0e4);
    //exit(0);
    double band_sign = (s->bw<0.0) ? -1.0 : 1.0;


    init_chirp_kernel<<<4096,32>>>(s->nchan, s->fft_len, s->freq[0], s->bw, s->chirp_gpu[0], band_sign, dmfac);
    cudaDeviceSynchronize();


    //float2 *chirp_host;
    //chirp_host = (float2 *)malloc(chirp_size);
    //cudaMemcpy(chirp_host, s->chirp_gpu[0], chirp_size, cudaMemcpyDeviceToHost);
    //for (int i=100000; i<101000; i++) {
    //    printf("CHIRP_GPU   %.36f\n",chirp_host[(s->nchan-1)*s->fft_len+i].x);fflush(stdout);
    //    }
}






/* Initialize all necessary memory, etc for doing dedispersion 
 * at the given params.  In the struct, the following MUST be 
 * filled in:
 *   rf, bw, dm, npts_per_block, npol
 * Optionally, fft_len and overlap can be specified as well.  If
 * either of these is set to 0, it will be computed automatically
 * from the input params.
 * TODO: more error checking
 */
extern "C"
void init_dedispersion(dedispersion_setup *s) {

    // Find lowest freq
    int i;
    char strlog[128];
    double f_chan_lo_mhz = s->freq[0];
    for (i=1; i<s->nchan; i++) 
        if (s->freq[i] < f_chan_lo_mhz) 
            f_chan_lo_mhz = s->freq[i];

    printf("rf=%f bw=%f dm=%f freq_lo=%f\n", s->rf, s->bw, s->dm,
            f_chan_lo_mhz);

    // Calc various parameters
    double f_lo_ghz = (f_chan_lo_mhz - fabs(s->bw)/2.0)/1.0e3;
    double f_hi_ghz = (f_chan_lo_mhz + fabs(s->bw)/2.0)/1.0e3;
    double chirp_len_samples = 4150. * s->dm *
        (1.0/(f_lo_ghz*f_lo_ghz) - 1.0/(f_hi_ghz*f_hi_ghz));
    printf("Chirp length = %f us\n", chirp_len_samples);
    chirp_len_samples *= fabs(s->bw);
    printf("Chirp length = %f samples\n", chirp_len_samples);

    if (s->overlap==0 && chirp_len_samples!=0.0) {
        // Do nearest power of 2 for now.  Find out what's optimal
        // Also find out what values don't work.
        s->overlap=1;
        while (s->overlap<chirp_len_samples) s->overlap <<= 1;
    }

    if (s->fft_len==0) {
        // Rough optimization based on testing w/ CUDA 2.3
        // Could make a "dedispersion plan" that tests?
        sprintf(strlog,"s->fft_len==0");
        log_info("init_dedispersion", strlog);
        //s->fft_len = 16*1024;
        //if (s->overlap <= 1024) s->fft_len = 32*1024; // previously 16
        //else if (s->overlap <= 2048) s->fft_len = 64*1024;
        //else if (s->overlap <= 16*1024) s->fft_len = 128*1024;
        //else if (s->overlap <= 64*1024) s->fft_len = 256*1024;
	
        s->fft_len = 262144;
        while (s->fft_len < 2.0*s->overlap) s->fft_len *= 2;
        
        if (s->fft_len > 8*1024*1024) {
            printf("init_dedispersion error: FFT length too large > 16*1024*1024 NOT tested ! (%d)\n",
                    s->fft_len);
            s->fft_len = 16*1024*1024;
        }
        s->fft_len = 4194304;
        
    }
    
    printf("fft_len=%d overlap=%d\n", s->fft_len, s->overlap); fflush(stdout);
    //exit(0);
    // Figure out how many FFTs per block
    s->nfft_per_block = 1;
    int64_t npts_used = s->fft_len;
    while(npts_used <= s->npts_per_block) {
        s->nfft_per_block++;
        npts_used = s->nfft_per_block*(s->fft_len-s->overlap) + s->overlap;
    }
    s->nfft_per_block--;
    s->nfft_per_block=1;
    npts_used = s->nfft_per_block*(s->fft_len-s->overlap) + s->overlap;

    // Allocate memory
    const size_t bytes_per_sample = 4; // 8-bit complex 2-pol
    const size_t bytes_in = bytes_per_sample * s->npts_per_block;
    const size_t bytes_tot = bytes_per_sample * s->fft_len*s->nfft_per_block;
    const size_t bytes_databuf = sizeof(float2)*s->fft_len*s->nfft_per_block;
    const size_t bytes_chirp = sizeof(float2)*s->fft_len*s->nchan;
    size_t total_gpu_mem = 0;
    size_t chirp_gpu_mem = 0;
    size_t overlap_gpu_mem = 0;
    size_t databuf_gpu_mem = 0;

    printf("npts_per_block=%d nfft_per_block=%d npts_used=%lld diff=%d\n",
            s->npts_per_block, s->nfft_per_block, npts_used,
            s->npts_per_block - npts_used);
    printf("s->npts_per_block=%lld\n",(int64_t)s->npts_per_block); 
    printf("bytes_in=%lld\n",(int64_t)bytes_in); 
    fflush(stdout);
    cudaError_t rv = cudaHostAlloc((void**)&(s->tbuf_host), bytes_in, cudaHostAllocWriteCombined);
    cudaMalloc((void**)&s->tbuf_gpu, bytes_in);
    total_gpu_mem += bytes_in;
    cudaMalloc((void**)&s->overlap_gpu, bytes_tot);
    total_gpu_mem += bytes_tot;
    overlap_gpu_mem += bytes_tot;
    cudaMalloc((void**)&s->databuf0_gpu, 2 * bytes_databuf);
    s->databuf1_gpu = s->databuf0_gpu + s->fft_len*s->nfft_per_block;
    total_gpu_mem += 2*bytes_databuf;
    databuf_gpu_mem += 2*bytes_databuf;
    cudaMalloc((void**)&s->chirp_gpu[0], bytes_chirp);
    total_gpu_mem += bytes_chirp;
    chirp_gpu_mem += bytes_chirp;
    for (i=0; i<s->nchan; i++) s->chirp_gpu[i] = s->chirp_gpu[0] + i*s->fft_len;

    //printf("allocated mem\n"); fflush(stdout);
   printf("total_gpu_mem = %lld MB\n", (int64_t)(total_gpu_mem >> 20));
   printf("chirp_gpu_mem = %lld MB\n", (int64_t)(chirp_gpu_mem >> 20));
   sprintf(strlog,"gpu memory created\n total_gpu_mem = %lld MB\n chirp_gpu_mem = %lld MB\n overlap_gpu_mem = %lld MB\n databuf_gpu_mem = %lld MB", (int64_t)(total_gpu_mem >> 20), (int64_t)(chirp_gpu_mem >> 20), (int64_t)(overlap_gpu_mem >> 20), (int64_t)(databuf_gpu_mem >> 20));
   log_info("dedisperse_gpu", strlog);


    cudaThreadSynchronize();
    //printf("init_dedispersion2 cuda_err=\'%s\'\n", cudaGetErrorString(cudaGetLastError()));
    // Init chirp function
    //init_chirp(s);
    init_chirp_GPU(s);

    cudaThreadSynchronize();
    //exit(0);
    //printf("init_dedispersion3 cuda_err=\'%s\'\n", cudaGetErrorString(cudaGetLastError()));

    // Plan FFT
    // nfft_per_block is only for 1 pol, hence the factor of 2 here
    cufftResult fft_rv = cufftPlan1d(&s->plan, s->fft_len, CUFFT_C2C, 2*s->nfft_per_block);
    //printf("fft (%d)\n", fft_rv); fflush(stdout);

    cudaThreadSynchronize();
    //printf("init_dedispersion4 cuda_err=\'%s\'\n", cudaGetErrorString(cudaGetLastError()));
    // Zero out fold buffers (call init_fold() to set them up)
    s->fold_phase = NULL;
    s->fold_step = NULL;
    s->foldbuf_gpu = NULL;

    // Zero out ds buffer
    s->dsbuf_gpu = NULL;

    // Zero out timers
    memset(&s->time, 0, sizeof(dedispersion_times));

    // Check errors
    cudaThreadSynchronize();
    //printf("init_dedispersion cuda_err=\'%s\'\n", cudaGetErrorString(cudaGetLastError()));

}

/* Actually do the dedispersion */
/* TODO: add benchmarking info */
extern "C"
void dedisperse(dedispersion_setup *s, int ichan,
        const unsigned char *in, float *out) {


    /* Various sizes */
    const size_t bytes_per_sample = 4; // 8-bit complex 2-pol
    const size_t bytes_in = bytes_per_sample * s->npts_per_block;
    const size_t npts_tot = s->fft_len*s->nfft_per_block;

    /* Benchmarking stuff 
     * Do we want to create these each time?
     */
#define NT 12
    cudaEvent_t t[NT];
    int it;
    for (it=0; it<NT; it++) cudaEventCreate(&t[it]);
    it=0;

    /* copy input data to transfer buffer */
    memcpy(s->tbuf_host, in, bytes_in);

    cudaEventRecord(t[it], 0); it++;
    cudaEventRecord(t[it], 0); it++;

    /* Copy data to GPU */
    cudaMemcpy(s->tbuf_gpu, s->tbuf_host, bytes_in, cudaMemcpyHostToDevice);
    cudaEventRecord(t[it], 0); it++;

    /* Expand overlap */
    expand_overlap(s);
    cudaEventRecord(t[it], 0); it++;

    /* Convert to floating point */
    byte_to_float_2pol_complex<<<16,1024>>>((unsigned short *)s->overlap_gpu, s->databuf0_gpu, s->databuf1_gpu, npts_tot);
    cudaEventRecord(t[it], 0); it++;

    /* Forward FFT */
    cufftExecC2C(s->plan, s->databuf0_gpu, s->databuf0_gpu, CUFFT_FORWARD);
    cudaEventRecord(t[it], 0); it++;

    /* Multiply by chirp */
    dim3 gd(2*s->nfft_per_block, s->fft_len/4096, 1);

    vector_multiply_complex<<<gd,512>>>(s->databuf0_gpu,
            s->chirp_gpu[ichan], s->fft_len);
    cudaEventRecord(t[it], 0); it++;

    dim3 rd(s->nfft_per_block, s->fft_len/4096, 1);
    if ((float)s->rmfac != 0.0 ) {
        defaraday<<<rd,512>>>(s->databuf0_gpu,
                              (float)s->freq[ichan],
                              (double)s->rmfac,
                              s->fft_len);
        }
	//float2 *chirp_host;
	//chirp_host = (float2 *)malloc(2*bytes_in);
	//cudaMemcpy(chirp_host, s->databuf0_gpu, 2*bytes_in, cudaMemcpyDeviceToHost);
	//for (int i=524288-4; i<524288+4; i++) {
	//    printf(" i = %d  in_gpu X  %.36f  |  in_gpu Y  %.36f\n",i, chirp_host[i].x, chirp_host[i].y);fflush(stdout);
	//    }
	//exit(0);
    cudaEventRecord(t[it], 0); it++;
	

    /* Inverse FFT */
    cufftExecC2C(s->plan, s->databuf0_gpu, s->databuf0_gpu, CUFFT_INVERSE);
    cudaEventRecord(t[it], 0); it++;

#define DETECT_AND_TRANSFER 0
#if DETECT_AND_TRANSFER
    /* Detect */
    detect_4pol<<<32,512>>>(s->databuf0_gpu, s->databuf1_gpu, npts_tot);
    cudaEventRecord(t[it], 0); it++;

    /* Re-quantize to 8 bit?? */
     
    /* Transfer data back, removing non-valid (overlapped) FFT edges */
    transfer_collapse_overlap(s);
    cudaEventRecord(t[it], 0); it++;
#endif

    cudaEventRecord(t[it], 0);
    cudaEventSynchronize(t[it]);

    /* Compute timers */
    float ttmp;
    it=1;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.transfer_to_gpu += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.overlap += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.bit_to_float += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.fft += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.xmult += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.fft += ttmp;
    s->time.total2 += ttmp;
    it++;

#if DETECT_AND_TRANSFER
    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.detect += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.transfer_to_host += ttmp;
    s->time.total2 += ttmp;
    it++;
#endif

    cudaEventElapsedTime(&ttmp, t[0], t[it+1]);
    s->time.total += ttmp;

    int nvalid = s->nfft_per_block*(s->fft_len-s->overlap);
    s->time.nsamp_tot += nvalid;

    for (it=0; it<NT; it++) cudaEventDestroy(t[it]); 
}

/* Actually just unpack the data */
extern "C"
void unpack(dedispersion_setup *s, int ichan,
        const unsigned char *in, float *out) {


    //cufftResult fft_rv;

    /* Various sizes */
    const size_t bytes_per_sample = 4; // 8-bit complex 2-pol
    const size_t bytes_in = bytes_per_sample * s->npts_per_block;
    const size_t npts_tot = s->fft_len*s->nfft_per_block;

    /* Benchmarking stuff 
     * Do we want to create these each time?
     */
#define NT 12
    cudaEvent_t t[NT];
    int it;
    for (it=0; it<NT; it++) cudaEventCreate(&t[it]);
    it=0;

    /* copy input data to transfer buffer */
    memcpy(s->tbuf_host, in, bytes_in);

    cudaEventRecord(t[it], 0); it++;
    cudaEventRecord(t[it], 0); it++;

    /* Copy data to GPU */
    cudaMemcpy(s->tbuf_gpu, s->tbuf_host, bytes_in, cudaMemcpyHostToDevice);
    cudaEventRecord(t[it], 0); it++;

    /* Expand overlap */
    expand_overlap(s);
    cudaEventRecord(t[it], 0); it++;

    /* Convert to floating point */
    byte_to_float_2pol_complex<<<16,128>>>((unsigned short *)s->overlap_gpu, s->databuf0_gpu, s->databuf1_gpu, npts_tot);
    cudaEventRecord(t[it], 0); it++;

    cudaEventRecord(t[it], 0);
    cudaEventSynchronize(t[it]);

    /* Compute timers */
    float ttmp;
    it=1;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.transfer_to_gpu += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.overlap += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.bit_to_float += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.fft += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.xmult += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[it], t[it+1]);
    s->time.fft += ttmp;
    s->time.total2 += ttmp;
    it++;

    cudaEventElapsedTime(&ttmp, t[0], t[it+1]);
    s->time.total += ttmp;

    int nvalid = s->nfft_per_block*(s->fft_len-s->overlap);
    s->time.nsamp_tot += nvalid;

    for (it=0; it<NT; it++) cudaEventDestroy(t[it]); 
}

/* Free any resources associated with dedispersion */
extern "C"
void free_dedispersion(dedispersion_setup *s) {

    cudaThreadSynchronize(); // Need?

    cudaFreeHost(s->tbuf_host);
    cudaFree(s->tbuf_gpu);
    cudaFree(s->overlap_gpu);
    cudaFree(s->databuf0_gpu);
    cudaFree(s->chirp_gpu[0]);

    cudaThreadExit();
}

#define print_percent(var) \
    printf("  %7.2f%% %s\n", 100.0*s->time.var/s->time.total, #var)
#define print_percent_short(var) \
    fprintf(f, "%.3f ", s->time.var/s->time.total)
void print_timing_report(dedispersion_setup *s) {

    /* Print to screen */
    printf("\n");
    printf("Total time  = %6.1f s (%.4f ns/samp)\n", 
            s->time.total/1e3, 1e6*s->time.total/(double)s->time.nsamp_tot);
    printf("Total2 time = %6.1f s (%.4f ns/samp)\n", 
            s->time.total2/1e3, 1e6*s->time.total2/(double)s->time.nsamp_tot);
    //printf("  %f ns/sample\n", 1e6*s->time.total/(double)s->time.nsamp_tot);
    print_percent(transfer_to_gpu);
    print_percent(overlap);
    print_percent(bit_to_float);
    print_percent(fft);
    print_percent(xmult);
#if DETECT_AND_TRANSFER
    print_percent(detect);
#endif
    print_percent(fold_mem);
    print_percent(fold_blocks);
    print_percent(fold_combine);
    print_percent(downsample);
    print_percent(transfer_to_host);

    /* print short version to file */
    FILE *f = fopen("dedisp_timing.dat", "a");
    fprintf(f, "%7d %6d %.4e %.4e ",  s->fft_len, s->overlap,
            s->time.total/(double)s->time.nsamp_tot,
            s->gp->drop_frac_tot);
    print_percent_short(transfer_to_gpu);
    print_percent_short(overlap);
    print_percent_short(bit_to_float);
    print_percent_short(fft);
    print_percent_short(xmult);
    print_percent_short(fold_mem);
    print_percent_short(fold_blocks);
    print_percent_short(fold_combine);
    print_percent_short(downsample);
    print_percent_short(transfer_to_host);
    fprintf(f, "\n");
    fclose(f);
}
