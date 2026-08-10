/* Layer Norm gamma/beta - Q8.8 */

#ifndef LAYERNORM_PARAMS_H
#define LAYERNORM_PARAMS_H

#include <stdint.h>

static const int16_t norm1_gamma_q[16] = {   244,   243,   226,   206,   228,   213,   257,   286,   236,   262,   273,   272,   303,   258,   257,   242 };
static const int16_t norm1_beta_q[16] = {     2,     5,     3,    15,     3,     1,     0,     6,    19,    -7,    -8,     7,    -3,    -3,     8,   -12 };
static const int16_t norm2_gamma_q[16] = {   298,   286,   286,   251,   292,   314,   294,   314,   255,   292,   312,   302,   319,   300,   301,   292 };
static const int16_t norm2_beta_q[16] = {    -6,     6,    -5,    -2,    -9,    -6,    -9,     6,     5,    -9,    -9,     8,    -1,     3,     8,    -2 };

#endif /* LAYERNORM_PARAMS_H */
