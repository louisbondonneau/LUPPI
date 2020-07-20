#ifndef _FOLD_H
#define _FOLD_H
#include "polyco.h"

/* Defines ordering of fold buf data.
 * First dim mentioned runs fastest in memory, etc..
 */
enum fold_order {
    chan_pol_bin, // For CPU/SSE folding
    pol_bin_chan, // For GPU folding
    bin_chan_pol  // For PSRFITS output
};

typedef struct {
    int nbin;
    int nchan;
    int npol;
    enum fold_order order;
    float *data;
    unsigned *count;
} foldbuf;

void malloc_foldbuf(foldbuf *f);

void free_foldbuf(foldbuf *f);

void clear_foldbuf(foldbuf *f);

size_t foldbuf_data_size(const foldbuf *f);
size_t foldbuf_count_size(const foldbuf *f);

void scale_counts(foldbuf *f, float fac);

int normalize_transpose_folds(float *out, const foldbuf *f);

typedef struct {
    polyco *pc;
    int imjd;
    double fmjd;
    char *data;
    int nsamp;
    double tsamp;
    int raw_signed;
    foldbuf *fb;
} fold_args;

void *fold_8bit_power_thread(void *_args);

int fold_8bit_power(const polyco *pc, int imjd, double fmjd, 
        const char *data, int nsamp, double tsamp, int raw_signed,
        foldbuf *f);

int accumulate_folds(foldbuf *ftot, const foldbuf *f);

#endif
