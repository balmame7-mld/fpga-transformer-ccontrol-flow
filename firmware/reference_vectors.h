/* Reference vectors pour validation numerique ZedBoard */
/* Comparer output_buf[0..15] avec ref_ffn_q_XXX (tolerance +-2 LSB) */

#ifndef REFERENCE_VECTORS_H
#define REFERENCE_VECTORS_H

#include <stdint.h>

/* === IF === Predicted by PyTorch: IF */
static const int16_t ref_tokens_IF[16] = { 1, 50, 60, 20, 61, 60, 21, 61, 51, 52, 7, 61, 54, 53, 0, 0 };
static const int16_t ref_ffn_q_IF[16] = {  -247,   -20,  -418,    49,   -23,   244,   -70,   481,   106,   148,  -264,   169,  -406,    69,   148,     2 };
/* float: [-0.9631, -0.0784, -1.6318, 0.1895, -0.0895, 0.9545, -0.2725, 1.8788, 0.4135, 0.5796, -1.0307, 0.6615, -1.5869, 0.2700, 0.5801, 0.0071] */

/* === WHILE === Predicted by PyTorch: WHILE */
static const int16_t ref_tokens_WHILE[16] = { 9, 60, 40, 61, 54, 3, 50, 60, 23, 60, 51, 52, 60, 45, 54, 53 };
static const int16_t ref_ffn_q_WHILE[16] = {   182,   261,   137,     4,  -246,  -301,   -37,  -239,   -90,  -273,   180,   201,   180,  -354,   172,   199 };
/* float: [0.7099, 1.0196, 0.5366, 0.0166, -0.9595, -1.1777, -0.1433, -0.9330, -0.3515, -1.0674, 0.7023, 0.7841, 0.7016, -1.3831, 0.6720, 0.7758] */

/* === FOR === Predicted by PyTorch: FOR */
static const int16_t ref_tokens_FOR[16] = { 9, 60, 40, 61, 54, 4, 50, 60, 40, 61, 54, 60, 21, 61, 54, 60 };
static const int16_t ref_ffn_q_FOR[16] = {  -204,   118,    -3,   216,   210,  -145,  -469,    71,   267,  -163,  -301,  -108,   495,   280,  -101,  -250 };
/* float: [-0.7972, 0.4621, -0.0111, 0.8431, 0.8184, -0.5647, -1.8316, 0.2769, 1.0420, -0.6360, -1.1777, -0.4215, 1.9326, 1.0937, -0.3944, -0.9772] */

/* === SWITCH === Predicted by PyTorch: SWITCH */
static const int16_t ref_tokens_SWITCH[16] = { 5, 50, 60, 51, 52, 6, 61, 55, 60, 45, 54, 8, 54, 53, 0, 0 };
static const int16_t ref_ffn_q_SWITCH[16] = {   130,  -308,   234,     1,    17,   302,   318,  -306,   -28,   194,   270,  -281,   -51,   -44,  -290,  -176 };
/* float: [0.5091, -1.2031, 0.9136, 0.0051, 0.0682, 1.1789, 1.2433, -1.1968, -0.1092, 0.7567, 1.0531, -1.0983, -0.1987, -0.1702, -1.1336, -0.6874] */

#endif /* REFERENCE_VECTORS_H */
