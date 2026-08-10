#ifndef TRANSFORMER_IMPROVED_H
#define TRANSFORMER_IMPROVED_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>
#include "layernorm_params.h"
// ==================== PARAMÈTRES ====================
#define SEQ_LEN     16
#define D_MODEL     16
#define NUM_HEADS   2
#define D_HEAD      8       // D_MODEL / NUM_HEADS
#define D_FF        32
#define EPSILON     0.00001f
#define SQRT_D_HEAD 2.828f  // sqrt(8), pour scaling attention

// ==================== TYPES ====================
//
// data_t = ap_fixed<16,8> = Q7.8 signé
//   - 1 bit signe + 7 bits entier + 8 bits fraction
//   - Range : [-128.0 , +127.996]
//   - Résolution : 1/256 = 0.0039
//   - Identique au format Q8.8 utilisé pour quantifier les poids PyTorch
//   - AP_RND  : arrondi au plus proche (vs troncature AP_TRN)
//   - AP_SAT  : saturation sur overflow (vs wrap-around qui crée des bugs silencieux)
//
typedef ap_fixed<16, 6, AP_RND, AP_SAT> data_t;

//
// acc_t = ap_fixed<32,24> = Q23.8 signé
//   - Accumulateur pour les produits de matrices
//   - Range entière : [-8,388,608 , +8,388,607]
//   - Vérification overflow :
//     * matmul  : D_MODEL×(128×128) = 16×16384 = 262,144  ✓
//     * FFN W1  : D_MODEL×(128×128) = 262,144             ✓
//     * FFN W2  : D_FF×(128×128)    = 32×16384 = 524,288  ✓
//   - Résolution : 8 bits fraction (1/256), alignée avec data_t
//
typedef ap_fixed<32, 24, AP_TRN, AP_SAT> acc_t;

//
// norm_t = float
//   - Layer Norm nécessite sqrt et divisions → float obligatoire
//   - Synthétisé en DSP flottant par HLS (acceptable, usage limité)
//
typedef float norm_t;

// ==================== FONCTION TOP ====================
void transformer_multihead(
    data_t *mem_in,
    data_t *mem_out,
    int data_offset,
    int wq_offset,
    int wk_offset,
    int wv_offset,
    int wo_offset,
    int w1_offset,
    int w2_offset,
    int out_offset
);

#endif // TRANSFORMER_IMPROVED_H
