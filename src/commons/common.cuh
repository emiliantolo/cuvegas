#ifndef COMMON_CUH
#define COMMON_CUH

#include <time.h>
#include <stdint.h>
#include <stdio.h>

#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif


double cpuMilliSeconds() {
    struct timespec _t;
    clock_gettime(CLOCK_REALTIME, &_t);
    return (double) _t.tv_sec*1000 + (double) (_t.tv_nsec/1e6);
}

#define checkCudaError(val) check((val), __FILE__, __LINE__)
inline void check(cudaError_t err, const char *const file, int const line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA error: %s %s %d \n", cudaGetErrorString(err), file, line);
    exit(-1);
  }
}

#endif
