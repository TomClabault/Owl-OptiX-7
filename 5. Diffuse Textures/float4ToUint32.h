#ifndef FLOAT4_TO_UINT32_H
#define FLOAT4_TO_UINT32_H

#include <cstdint>

#include "vector_types.h"

void cuda_float4_to_uint32(float4* float4_input, unsigned int width, unsigned int height, uint32_t* uint32_output);

#endif
