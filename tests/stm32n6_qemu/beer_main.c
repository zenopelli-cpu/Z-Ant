#include "semihost.h"

#include <stddef.h>
#include <stdint.h>

extern int32_t predict(float *input, uint32_t *input_shape, uint32_t shape_len,
                       float **result);

static void fill_input(float *buf, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    buf[i] = 0.0f;
  }
}

int main(void) {
  /* Beer model expects input shape [1, 96, 96, 1] */
  static float input[96u * 96u];
  uint32_t shape[4] = {1u, 96u, 96u, 1u};
  float *output = NULL;

  fill_input(input, (uint32_t)(96u * 96u));

  /* DWT cycle counter setup */
  volatile uint32_t *DEMCR = (uint32_t *)0xE000EDFC;
  volatile uint32_t *DWT_CTRL = (uint32_t *)0xE0001000;
  volatile uint32_t *DWT_CYCCNT = (uint32_t *)0xE0001004;
  const uint32_t DEMCR_TRCENA = (1u << 24);
  const uint32_t DWT_CYCCNTENA = (1u << 0);

  *DEMCR |= DEMCR_TRCENA;
  *DWT_CYCCNT = 0u;
  *DWT_CTRL |= DWT_CYCCNTENA;

  const int32_t status = predict(input, shape, 4u, &output);
  if (status != 0) {
    semihost_write0("beer FAIL\n");
    return (int)status;
  }

  if (output == NULL) {
    semihost_write0("beer FAIL\n");
    return -1;
  }

  /* Read cycles and print both cycles and ms at 600 MHz */
  uint32_t cycles = *DWT_CYCCNT;
  /* Convert cycles to milliseconds with 2 decimal places without libc:
     ms_hundredths = cycles * 100 / 600000  (since 1 ms = 600000 cycles at
     600MHz) */
  uint64_t ms_hundredths = ((uint64_t)cycles * 100ull) / 600000ull;
  uint32_t ms_int = (uint32_t)(ms_hundredths / 100ull);
  uint32_t ms_frac = (uint32_t)(ms_hundredths % 100ull);
  /* fps*100 = (Fclk*100)/cycles to avoid division by zero from ms_hundredths */
  uint32_t fps_hundredths =
      (cycles != 0u) ? (uint32_t)(60000000000ull / (uint64_t)cycles) : 0u;
  uint32_t fps_int = fps_hundredths / 100u;
  uint32_t fps_frac = fps_hundredths % 100u;

  /* Integer to decimal helpers */
  char buf[96];
  uint32_t pos = 0u;
  /* append helpers (C, no stdio) */
  {
    /* append string literal */
    const char *s1 = "beer PASS cycles=";
    while (*s1 && pos + 1u < sizeof(buf)) {
      buf[pos++] = *s1++;
    }
  }
  {
    /* append_u32(cycles) */
    char tmp[16];
    uint32_t i = 0u;
    uint32_t v = cycles;
    if (v == 0u) {
      if (pos + 1u < sizeof(buf))
        buf[pos++] = '0';
    } else {
      while (v > 0u && i < sizeof(tmp)) {
        tmp[i++] = (char)('0' + (v % 10u));
        v /= 10u;
      }
      while (i > 0u && pos + 1u < sizeof(buf)) {
        buf[pos++] = tmp[--i];
      }
    }
  }
  {
    const char *s2 = " time_ms=";
    while (*s2 && pos + 1u < sizeof(buf)) {
      buf[pos++] = *s2++;
    }
  }
  {
    /* append_u32(ms_int) */
    char tmp[16];
    uint32_t i = 0u;
    uint32_t v = ms_int;
    if (v == 0u) {
      if (pos + 1u < sizeof(buf))
        buf[pos++] = '0';
    } else {
      while (v > 0u && i < sizeof(tmp)) {
        tmp[i++] = (char)('0' + (v % 10u));
        v /= 10u;
      }
      while (i > 0u && pos + 1u < sizeof(buf)) {
        buf[pos++] = tmp[--i];
      }
    }
  }
  if (pos + 1u < sizeof(buf))
    buf[pos++] = '.';
  if (pos + 1u < sizeof(buf))
    buf[pos++] = (char)('0' + (ms_frac / 10u));
  if (pos + 1u < sizeof(buf))
    buf[pos++] = (char)('0' + (ms_frac % 10u));
  {
    const char *s3 = " fps=";
    while (*s3 && pos + 1u < sizeof(buf)) {
      buf[pos++] = *s3++;
    }
  }
  {
    /* append_u32(fps_int) */
    char tmp[16];
    uint32_t i = 0u;
    uint32_t v = fps_int;
    if (v == 0u) {
      if (pos + 1u < sizeof(buf))
        buf[pos++] = '0';
    } else {
      while (v > 0u && i < sizeof(tmp)) {
        tmp[i++] = (char)('0' + (v % 10u));
        v /= 10u;
      }
      while (i > 0u && pos + 1u < sizeof(buf)) {
        buf[pos++] = tmp[--i];
      }
    }
  }
  if (pos + 1u < sizeof(buf))
    buf[pos++] = '.';
  if (pos + 1u < sizeof(buf))
    buf[pos++] = (char)('0' + (fps_frac / 10u));
  if (pos + 1u < sizeof(buf))
    buf[pos++] = (char)('0' + (fps_frac % 10u));
  if (pos + 1u < sizeof(buf))
    buf[pos++] = '\n';
  if (pos + 1u < sizeof(buf))
    buf[pos++] = '\0';

  semihost_write0(buf);
  return 0;
}
