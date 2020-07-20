#ifndef _DOWNSAMPLE_GPU_H
#define _DOWNSAMPLE_GPU_H
#include "dedisperse_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif
void init_downsample(dedispersion_setup *s);
void downsample(dedispersion_setup *s);
#ifdef __cplusplus
}
#endif
#endif
