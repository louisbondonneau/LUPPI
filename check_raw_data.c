/* check_raw_data.c
 *
 */

#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <sched.h>
#include <getopt.h>
#include <stdbool.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>

#include "fitshead.h"
#include "psrfits.h"
#include "status.h"
#include "params.h"
#include "cpu_utils.h"
#include "mjdtime.h"


#define MAX_HEADER_SIZE 16*1024

typedef struct {
    int block_count;
    int hdlen;
    int blocksize;
    int filenum;
    int dropped;  /* Number of datablocks dropped */
    double t_read;
    double tsubint;

} timing_stats;

void print_stats(timing_stats *ts) {
    printf("Read %d blocks (%.1lf sec) from %d files (missed %d blocks %.1f%%)\n",
    	ts->block_count, ts->block_count * ts->tsubint, ts->filenum+1, ts->dropped, ts->dropped / (float)(ts->block_count+ts->dropped)*100.);
    float datasize = ts->block_count * ((float)ts->hdlen + (float)ts->blocksize)/ (1024.*1024.);
    printf("Read %f MB in %f s : %f MB/s\n", datasize, ts->t_read, datasize/ts->t_read);
}

void usage() {
    fprintf(stderr,
    "Usage: check_raw_data basefilename\n"
    "Options:\n"
    "  -h, --help        This message\n"
    "  -v, --verbose     Verbose for debugging\n"
   );
}

int main(int argc, char *argv[]) {

    bool verbose=false;
    int opt, opti;

    static struct option long_opts[] = {
        {"help",    0, NULL, 'h'},
        {"verbose", 0, NULL, 'v'},
        {0,0,0,0}
    };

    while ((opt=getopt_long(argc,argv,"vh",long_opts,&opti))!=-1) {
        switch (opt) {
	    case 'v':
	        verbose=true;
		break;
            default:
            case 'h':
                usage();
                exit(0);
                break;
        }
    }

    if (argc == 1) {
	usage();
	exit(0);
    }


    // -- Read in general parameters --
    obs_params gp;
    struct psrfits pf;
    pf.sub.dat_freqs = NULL;
    pf.sub.dat_weights = NULL;
    pf.sub.dat_offsets = NULL;
    pf.sub.dat_scales = NULL;

    /* init timing stats */
    timing_stats ts;
    ts.block_count = 0;
    ts.filenum=0;
    ts.dropped=0;

    /* Init output file */
    char basefilename[128];
    sprintf(basefilename, "%s.%04d.raw", argv[optind], ts.filenum);
    printf("basefilename = %s\n",basefilename);
    FILE *fraw = NULL;
    if((fraw=fopen(basefilename,"r")) == NULL) {
        printf("Could not open %s\n", basefilename);
	exit(-1);
    } else printf("Opened %s\n", basefilename);

    /* Loop */
    int curblock=0, first=1;
    char *ptr=NULL;
    long long last_pktidx=0, pktidx_diff = 0;

    struct timespec t_read_start, t_read_stop;
    
    int rv=0;
    int npacket=0, ndrop=0;
    double tsubint = 0.0;

    /* header */
    char *hdr_buffer;
    hdr_buffer = (char *)malloc(MAX_HEADER_SIZE * sizeof(char));

    /* timing start */
    clock_gettime(CLOCK_REALTIME, &t_read_start);
    while (1) {

	/* Read header */
	memset(hdr_buffer, 0, MAX_HEADER_SIZE);

	//if (first) {
        rv = fread(hdr_buffer, 1, MAX_HEADER_SIZE, fraw);

	/* Go to new file if it exists */
        if (rv != MAX_HEADER_SIZE && feof(fraw)) {
	    fclose(fraw);
            printf("Closed %s\n", basefilename);
	    ts.filenum++;
	    sprintf(basefilename, "%s.%04d.raw", argv[optind], ts.filenum);
            if((fraw=fopen(basefilename,"r")) == NULL) {
        	printf("File %s not found\n", basefilename);
		break;
    	    } else {
	        printf("Opened %s\n", basefilename);
		continue;
	    }	
	}    

	ts.hdlen = gethlength (hdr_buffer);
	if (ts.hdlen < 0) exit(0);

	fseek(fraw, -MAX_HEADER_SIZE+ts.hdlen, SEEK_CUR);

	/* Read the header */
	read_status_params(hdr_buffer, &gp, &pf);
	read_subint_params(hdr_buffer, &gp, &pf);

	if (verbose) {
	    printf("curblock : %d\n", ts.block_count);
	    printf(" SRCNAME : %s\n", pf.hdr.source);
	    printf("     MJD : %Lf\n", pf.hdr.MJD_epoch);
	    printf("  PKTIDX : %Ld\n", gp.packetindex);
	    printf(" TSUBINT : %lf\n", pf.sub.tsubint);
	}    

	if (first) {
	    ts.tsubint = pf.sub.tsubint;
	    hgeti4(hdr_buffer, "BLOCSIZE", &ts.blocksize);
	    ptr = (char *)malloc(ts.blocksize * sizeof(char));
	    first=0;
	}
	else {
	    pktidx_diff = gp.packetindex - last_pktidx;
	    npacket += gp.packets_per_block;
	    ndrop += (gp.packets_per_block - gp.n_packets) + gp.n_dropped;
	    tsubint = pf.hdr.dt * (npacket - ndrop) * gp.packetsize / pf.hdr.nchan / pf.hdr.npol; // Only true for 8-bit data


	    if (pktidx_diff < 0) printf("Warning: Read datablock %Ld blocks delayed\n", -pktidx_diff / (ts.blocksize/gp.packetsize));
	    if (pktidx_diff != ts.blocksize/gp.packetsize) {
	        if (pktidx_diff == 0) printf("Duplicate datablock (pktidx=%Ld)\n", gp.packetindex);
		else {
	            ts.dropped += pktidx_diff / (ts.blocksize/gp.packetsize);
	            printf("Missed %Ld blocs at pktidx=%Ld (%.1lfs) (pktidx_diff=%Ld)\n", pktidx_diff / (ts.blocksize/gp.packetsize), last_pktidx, tsubint, pktidx_diff);
		}    
	    }	 


	    last_pktidx = gp.packetindex;
	}

	/* Read the data */
	rv = fread(ptr, 1, (size_t)ts.blocksize, fraw);

	if (rv != ts.blocksize) {
	    printf("Warning: data block not complete rv=%d blocksize=%d\n", rv, ts.blocksize);
	    break;
	}

	ts.block_count++;
	curblock++;

    }

    /* timing stop */
    clock_gettime(CLOCK_REALTIME, &t_read_stop);
    ts.t_read = (double)timediff(t_read_start, t_read_stop);

    /* Close file */
    if (fraw!=NULL) fclose(fraw);

    /* Print stats */
    print_stats(&ts);


    return 0;
}
