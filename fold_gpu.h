#ifndef _FOLD_GPU_H
#define _FOLD_GPU_H
#include "dedisperse_gpu.h"
#include "fold.h"
#ifdef __cplusplus
extern "C" {
#endif
void init_fold(dedispersion_setup *s);
int compute_fold_params(dedispersion_setup *s, const polyco *pc);
void fold(dedispersion_setup *s, int chan, foldbuf *fb_out);
#ifdef __cplusplus
}
#endif
#endif
