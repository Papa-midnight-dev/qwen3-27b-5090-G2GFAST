// Shared bit-identical NVFP4 block-16 fused epilogue helpers for half-warp (1 elem/lane) and quad (4 elems/lane).
#pragma once

#include <cuda_fp8.h>
#include <cstdint>

#include "kernels.h"

namespace q27 {

__device__ __forceinline__ uint8_t epilogue_cvt_e2m1x2(float even, float odd) {
  uint16_t r;
  asm("{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n cvt.u16.u8 %0, t;\n}"
      : "=h"(r) : "f"(odd), "f"(even));
  return uint8_t(r);
}

// 16 consecutive elements held by a half-warp (lane 0..15 or lane 16..31), 1 element per lane.
// Matches quant_block16 in kernels.cu bit for bit.
__device__ __forceinline__ void store_fp4_halfwarp(float y, float in_scale, int row, int kb, int K,
                                                   uint8_t* q4, uint8_t* sf4, int lane) {
  float amax = fabsf(y);
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  const float gs = 1.f / in_scale;
  const __nv_fp8_storage_t sfq = __nv_cvt_float_to_fp8(gs * amax / 6.f, __NV_SATFINITE, __NV_E4M3);
  const float sff = float(__half(__nv_cvt_fp8_to_halfraw(sfq, __NV_E4M3)));
  const float inv = sff != 0.f ? gs / sff : 0.f;
  const float y_odd = __shfl_xor_sync(0xffffffffu, y, 1);
  const uint32_t b0 = epilogue_cvt_e2m1x2(y * inv, y_odd * inv);
  const uint32_t b1 = __shfl_xor_sync(0xffffffffu, b0, 2);
  const uint32_t w0 = b0 | (b1 << 8);
  const uint32_t w1 = __shfl_xor_sync(0xffffffffu, w0, 4);
  const uint32_t word = w0 | (w1 << 16);
  const uint32_t hi = __shfl_xor_sync(0xffffffffu, word, 8);
  if ((lane & 15) == 0) {
    *reinterpret_cast<uint2*>(q4 + int64_t(row) * (K / 2) + int64_t(kb) * 8) = make_uint2(word, hi);
    sf4[sf_offset(row, kb, K / 16)] = sfq;
  }
}

// 16 consecutive elements held by a quad of 4 lanes (4 consecutive elements per lane).
// Matches quant_block16 in kernels.cu bit for bit.
__device__ __forceinline__ void store_fp4_quad(const float y[4], float in_scale, int row, int kb, int K,
                                               uint8_t* q4, uint8_t* sf4, int lane) {
  float amax = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  const float gs = 1.f / in_scale;
  const __nv_fp8_storage_t sfq = __nv_cvt_float_to_fp8(gs * amax / 6.f, __NV_SATFINITE, __NV_E4M3);
  const float sff = float(__half(__nv_cvt_fp8_to_halfraw(sfq, __NV_E4M3)));
  const float inv = sff != 0.f ? gs / sff : 0.f;
  const uint32_t h16 = uint32_t(epilogue_cvt_e2m1x2(y[0] * inv, y[1] * inv)) |
                       (uint32_t(epilogue_cvt_e2m1x2(y[2] * inv, y[3] * inv)) << 8);
  const uint32_t h16_hi = __shfl_xor_sync(0xffffffffu, h16, 1);
  const uint32_t word = h16 | (h16_hi << 16);
  const uint32_t hi = __shfl_xor_sync(0xffffffffu, word, 2);
  if ((lane & 3) == 0) {
    *reinterpret_cast<uint2*>(q4 + int64_t(row) * (K / 2) + int64_t(kb) * 8) = make_uint2(word, hi);
    sf4[sf_offset(row, kb, K / 16)] = sfq;
  }
}

}  // namespace q27
