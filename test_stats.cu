/* ======================================================================== */
/* Program to process data. This program performs spectra
 */

/* ======================================================================== */

//extern "C" {
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
//}

//#define _DEBUG /* Comment out once cuda stuff is working */
#include <cuda.h>

//#define NUM 1048576
#define NUM 2097152
//#define NBLOCKS 512
//#define NTHREADS 512


#include "stats_gpu.cu"
#include "dedisperse_gpu.cu"

int main(int argc, char *argv[]) {


    /* GPU memory sizes */
    int i,j;
    int nblocks=512;
    int nthreads=512;
    int nbiter=1;

    for(i=1;i<argc;i++) {
        //if(strncmp(argv[i],"nblocks=",8) == 0)  sscanf(argv[i],"nblocks=%d",&nblocks);
        //if(strncmp(argv[i],"nthreads=",9) == 0) sscanf(argv[i],"nthreads=%d",&nthreads);
        if(strncmp(argv[i],"nbiter=",7) == 0)   sscanf(argv[i],"nbiter=%d",&nbiter);
    }


    size_t mem_tot, mem_buf, mem_out, mem_result;
    mem_buf = 4 * NUM * sizeof(float);
    mem_out = 4 * NUM / nthreads * sizeof(float);
    mem_result = 128*sizeof(float);


    mem_tot = mem_buf + mem_out + mem_result;
    printf("Need %dMB GPU memory\n", (int) mem_tot>>20); fflush(stdout);

    // -- Init CUDA --
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("%d device(s) detected\n", deviceCount);

    // -- Use GPU 0 or 1 --
    cudaSetDevice(0);

    /* Allocate GPU memory and host transfer buffers */
    float *data_device;
    float *out_device;
    float *result_device;
    cudaMalloc((void**)&data_device, mem_buf);
    cudaMalloc((void**)&out_device, mem_out);
    cudaMalloc((void**)&result_device, mem_result);


    float *data_host;
    float *result_host;
    float *out_host;
    cudaMallocHost((void**)&data_host, mem_buf);
    cudaMallocHost((void**)&out_host, mem_out);
    cudaMallocHost((void**)&result_host, mem_result);

    for(i=0;i<NUM;i++) {
      data_host[4*i    ] = 1.0; data_host[4*i + 2] = 3.0;  
      data_host[4*i + 1] = 2.0; data_host[4*i + 3] = 4.0; 
    }
    //for(i=0;i<4*NUM;i++) data_host[i] = 1.0;

    const unsigned N = 4*NUM;
    const unsigned zero = 0;

    stats_setup s;
    dedispersion_setup ds;
    init_stats(&ds, &s);
    s.npts = N;

    // -- Loop over nbiter iterations --
    for(i=0;i<nbiter;i++) {

	// -- Copy data to GPU --
	cudaMemcpy(data_device, data_host, mem_buf, cudaMemcpyHostToDevice);
#if 0	

	// -- Calc the mean of the data --
	reduce<NTHREADS, float><<<nblocks, nthreads>>> (data_device, out_device, N, zero);
	//cudaThreadSynchronize();
	cudaMemcpy(out_host, out_device, mem_result, cudaMemcpyDeviceToHost);
	printf("temp = %f\n", out_host[0]);
	reduce<NBLOCKS/2, float><<<1, nblocks/2>>> (out_device, result_device, nblocks, N);
	cudaMemcpy(result_host, result_device, mem_result, cudaMemcpyDeviceToHost);
	printf("mean = %f\n", result_host[0]);

        variance<NTHREADS, float><<<nblocks, nthreads>>> (data_device, out_device, result_device, N);
	cudaMemcpy(out_host, out_device, mem_result, cudaMemcpyDeviceToHost);
	printf("temp = %f\n", out_host[0]);
	reduce<NBLOCKS/2, float><<<1, nblocks/2>>> (out_device, result_device, nblocks, zero);
#endif
	s.data_gpu = data_device;
	mean_gpu(&s);
	var_gpu(&s);
	// -- Copy data from GPU --
	//cudaMemcpy(result_host, result_device, mem_result, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_host, &s.tbuf_gpu[0], mem_result, cudaMemcpyDeviceToHost);
	printf("tmp = %f\n", out_host[0]);
	out_host[0]=2.;
	cudaMemcpy(out_host, s.mean_gpu, 4, cudaMemcpyDeviceToHost);
	printf("mean = %f\n", out_host[0]);
	cudaMemcpy(out_host, s.var_gpu, 4, cudaMemcpyDeviceToHost);
	printf("variance = %f\n", sqrt(out_host[0]/N));
    }
    cudaThreadSynchronize();
    //printf("data_host = %f\n", data_host[1]);
    //printf("result = %f\n", out_host[1]);

/*    cudaFree(data_device);
    cudaFree(out_device);
    cudaFree(result_device);
    cudaFreeHost(data_host);
    cudaFreeHost(result_host);
*/
    return(0);
}

