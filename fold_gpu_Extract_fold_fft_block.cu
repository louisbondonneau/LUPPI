
/* Fold each FFT chunk separately.
 * pol0, pol1 are input baseband data
 * Only works for 4 pol output.
 * Call with grid dims (nffts, nbins/BINS_PER_BLOCK)
 * All shared blocks need to fit into shared mem (16kB)
 */
#define BINS_PER_BLOCK 64
#define NTHREAD_FOLD BINS_PER_BLOCK
__global__ void fold_fft_blocks(const float2 *pol0, const float2 *pol1, 
        const double *phase, const double *step, 
        int fftlen, int overlap, int nbin,
        float4 *foldtmp, unsigned *foldtmp_c) {

    // Size params
    const int ifft = blockIdx.x;
    const int ibin = blockIdx.y;
    const int tid = threadIdx.x; // Thread index within the block
    const int nvalid = fftlen - overlap;

    // Pointers to start of valid data in global mem
    const float2 *ptr0 = pol0 + ifft*fftlen + overlap/2;
    const float2 *ptr1 = pol1 + ifft*fftlen + overlap/2;

    // Fold info
    const double bin0 = phase[ifft];
    const double bin_samp = step[ifft];   // bins/sample
    const double samp_bin = 1.0/bin_samp; // samples/bin
    const int bin_lo = ibin*BINS_PER_BLOCK + tid; // assumes 1 thread/bin
    const int nturn = ((double)nvalid*bin_samp)/(double)nbin + 2; // GD : Change +1 into +2

    // Fold results for this thread
    float4 folddata = make_float4(0,0,0,0);
    unsigned foldcount = 0;

    // Loop over number of pulse periods in data block
    for (int iturn=0; iturn<nturn; iturn++) {

        // Determine range of samples needed for this bin, turn
        int samp0 = samp_bin*((double)bin_lo-bin0+(double)iturn*nbin)+0.5;
        int samp1 = samp_bin*((double)bin_lo-bin0+(double)iturn*nbin+1)+0.5;

        // Range checks
        if (samp0<0) { samp0=0; }
        if (samp1<0) { samp1=0; }
        if (samp0>nvalid) { samp0=nvalid; }
        if (samp1>nvalid) { samp1=nvalid; }

        // Read in and add samples
        for (int isamp=samp0; isamp<samp1; isamp++) {
            float2 p0 = ptr0[isamp];
            float2 p1 = ptr1[isamp];
            folddata.x += p0.x*p0.x + p0.y*p0.y;
            folddata.y += p1.x*p1.x + p1.y*p1.y;
            folddata.z += p0.x*p1.x + p0.y*p1.y;
            folddata.w += p0.x*p1.y - p0.y*p1.x;
            foldcount++;
        }
    }

    // Copy results into global mem
    const unsigned prof_offset = ifft * nbin;
    foldtmp[prof_offset + bin_lo].x = folddata.x;
    foldtmp[prof_offset + bin_lo].y = folddata.y;
    foldtmp[prof_offset + bin_lo].z = folddata.z;
    foldtmp[prof_offset + bin_lo].w = folddata.w;
    foldtmp_c[prof_offset + bin_lo] = foldcount;
}
