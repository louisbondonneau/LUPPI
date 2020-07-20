
#ifndef _POLYCO_H
#define _POLYCO_H

#include <stdio.h>
#include <stdlib.h>

#include "polyco_struct.h"

#define POLYCODIR "/obs"

#ifdef __cplusplus
extern"C" {
#endif

int read_one_pc(FILE *f, polyco *pc, const char *psr);
int read_pc(FILE *f, polyco *pc, const char *psr, int mjd, double fmjd);
int read_all_pc(FILE *f,  polyco **pc, const char *psr);
int select_pc(const  polyco *pc, int npc, const char *psr,
        int imjd, double fmjd);
double psr_phase(const  polyco *pc, int mjd, double fmjd, double *freq,
        long long *pulsenum);
double psr_fdot(const  polyco *pc, int mjd, double fmjd, double *fdot);
double psr_phase_avg(const  polyco *pc, int mjd, 
        double fmjd1, double fmjd2);
int pc_range_check(const  polyco *pc, int mjd, double fmjd);
int pc_out_of_range(const  polyco *pc, int mjd, double fmjd);
int pc_out_of_range_sloppy(const  polyco *pc, int mjd, double fmjd, double slop);
int polycos_differ(const  polyco *pc1, const  polyco *pc2);

#ifdef __cplusplus
}
#endif

#include "psrfits.h"
int make_polycos(const char *parfile, struct hdrinfo *hdr, char *src, polyco **pc);
int make_const_polyco(double freq, const struct hdrinfo *hdr, polyco **pc);
void make_polycos_cleanup(char *polycofile);

#endif
