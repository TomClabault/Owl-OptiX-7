#ifndef FLOAT4_TO_UINT32_H
#define FLOAT4_TO_UINT32_H

#include <cstdint>

#include "vector_types.h"

void hdr_tone_mapping(float4* float4_input, unsigned int width, unsigned int height, uint32_t* uint32_output);

#endif
