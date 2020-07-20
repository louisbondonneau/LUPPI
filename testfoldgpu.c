#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "polyco.h"
#include "dedisperse_gpu.h"
#include "fold_gpu.h"

int main(int argc, char *argv[]) {
int i;
int imjd;
double fmjd;
double phase,freq;
char polycofile[256],line[256];
polyco *pc=NULL;
int npc=0, ipc;
FILE *pcf;
dedispersion_setup ds;

i=1;
while(i<argc) {
  if( 0==strcmp("-polyco", argv[i]) ) {
    i++; strcpy(polycofile,argv[i]);
    printf("%s> polycofile= %s\n",argv[0],polycofile);
  }
  if( 0==strcmp("-imjd", argv[i]) ) {
    i++; sscanf(argv[i],"%d",&imjd);
  }
  if( 0==strcmp("-fmjd", argv[i]) ) {
    i++; sscanf(argv[i],"%lf",&fmjd);
  }
  i++;
}

if((pcf = fopen(polycofile, "r")) != NULL) {
 npc = read_all_pc(pcf,&pc, NULL);
 fclose(pcf);
 printf("%s> Read %d POLYCO blocs in %s\n",argv[0],npc,polycofile);
} else {
 printf("%s> Unable to open file %s ... Exit.\n",argv[0],polycofile);
 return(-1);
}

ipc = select_pc(pc, npc, NULL, imjd, fmjd);
phase = psr_phase(&pc[ipc], imjd, fmjd, &freq, NULL);
printf("%s> imjd= %d jmjd= %.15lf  ipc= %d phase= %.6lf freq= %.6lf\n",argv[0],imjd,fmjd,ipc,phase,freq);

fmjd+=0.1;
ipc = select_pc(pc, npc, NULL, imjd, fmjd);
phase = psr_phase(&pc[ipc], imjd, fmjd, &freq, NULL);
printf("%s> imjd= %d jmjd= %.15lf  ipc= %d phase= %.6lf freq= %.6lf\n",argv[0],imjd,fmjd,ipc,phase,freq);

printf("%s> Ready to go ? ... will be using GPU !!\n",argv[0]);
gets(line);

// -- Select GPU --
printf("%s> Selecting GPU 0 ...\n",argv[0]);
cudaSetDevice(0);
printf(" selected !\n");

ds.nbins_fold=2048;
ds.bw=4.0;
ds.overlap=8192;
ds.fft_len=262144;
ds.nfft_per_block=8;
ds.imjd=imjd; ds.fmjd=fmjd;
printf("%s> Computing and sending to GPU memory the parameters...\n",argv[0]);
compute_fold_params(&ds, &pc[ipc]);

return(0);
}
