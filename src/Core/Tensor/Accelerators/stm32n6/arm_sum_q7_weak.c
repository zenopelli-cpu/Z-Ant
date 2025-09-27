// Minimal weak fallback for CMSIS-DSP arm_sum_q7 to avoid undefined reference
// on toolchains where CMSIS-DSP sources are not linked. If the real CMSIS-DSP
// implementation is linked, it will override this weak symbol.

#include <stdint.h>

__attribute__((weak)) void arm_sum_q7(const int8_t *src, uint32_t blockSize,
                                      int32_t *result) {
  int32_t acc = 0;
  for (uint32_t i = 0; i < blockSize; ++i) {
    acc += (int32_t)src[i];
  }
  *result = acc;
}
