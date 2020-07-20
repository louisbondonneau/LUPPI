#ifndef _POLYCO_STRUCT_H
#define _POLYCO_STRUCT_H
typedef struct {
    char psr[15];
    int mjd;
    double fmjd;
    double dm;
    double rm;
    long long rphase_int;
    double rphase;
    double f0;
    double earthz4;
    int nsite;
    int nmin;
    int nc;
    float rf;
    int used;
    double c[15];
} polyco;
#endif
