#include "transformer_multihead.h"
#include "attention_bias_params.h"
#include "positional_encoding.h"

// ==================== RESIDUAL ADD (SKIP CONNECTION) ====================
void residual_add(
    const data_t Input1[SEQ_LEN][D_MODEL],
    const data_t Input2[SEQ_LEN][D_MODEL],
    data_t Output[SEQ_LEN][D_MODEL]
) {
    #pragma HLS INLINE off
    ADD_I: for(int i = 0; i < SEQ_LEN; i++) {
        ADD_J: for(int j = 0; j < D_MODEL; j++) {
            #pragma HLS PIPELINE II=1
            Output[i][j] = Input1[i][j] + Input2[i][j];
        }
    }
}

// ==================== MATMUL OPTIMISÉE (DSP CONSTRAINED) ====================
void matmul_opt(
    const data_t A[SEQ_LEN][D_MODEL],
    const data_t B[D_MODEL][D_HEAD],
    const data_t bias[D_HEAD],
    data_t C[SEQ_LEN][D_HEAD]
) {
    #pragma HLS INLINE off
    // Partitionnement calqué sur le facteur d'unroll pour un routage propre
    #pragma HLS ARRAY_PARTITION variable=B cyclic factor=4 dim=1

    MATMUL_I: for(int i = 0; i < SEQ_LEN; i++) {
        MATMUL_J: for(int j = 0; j < D_HEAD; j++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = (acc_t)bias[j];   // <-- biais injecté ici, pas de RMW séparé 
            MATMUL_K: for(int k = 0; k < D_MODEL; k++) {
                #pragma HLS UNROLL factor=4 
                // Multiplication native 16x16 -> allocation automatique DSP
                sum += A[i][k] * B[k][j]; 
            }
            C[i][j] = (data_t)sum;
        }
    }
}

// ==================== PYTORCH EXACT ATTENTION HEAD ====================
void attention_head(
    const data_t Q[SEQ_LEN][D_HEAD],
    const data_t K[SEQ_LEN][D_HEAD],
    const data_t V[SEQ_LEN][D_HEAD],
    data_t Output[SEQ_LEN][D_HEAD]
) {
    #pragma HLS INLINE off

    // Plus de partitionnement complet ici, économie massive de LUTs
    float scores[SEQ_LEN][SEQ_LEN]; 

    /* 1. QK^T avec scaling factor de PyTorch (sqrt(8) = 2.828) */
    SCORE_LOOP_I: for(int i = 0; i < SEQ_LEN; i++) {
        SCORE_LOOP_J: for(int j = 0; j < SEQ_LEN; j++) {
            #pragma HLS PIPELINE II=1
            acc_t dot = 0;
            SCORE_LOOP_K: for(int k = 0; k < D_HEAD; k++) {
                #pragma HLS UNROLL factor=4
                dot += Q[i][k] * K[j][k];
            }
            scores[i][j] = (float)dot / SQRT_D_HEAD; 
        }
    }

    /* 2. Softmax Exponentiel + Pondération V */
    APPLY_LOOP_I: for(int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS PIPELINE off // Permet la réutilisation des opérateurs float d'une ligne à l'autre
        
        // Stabilisation numérique (Max-shift)
        float max_val = -99999.0f;
        MAX_LOOP: for(int j = 0; j < SEQ_LEN; j++) {
            #pragma HLS PIPELINE II=1
            if(scores[i][j] > max_val) max_val = scores[i][j];
        }

        // Calcul des exponentielles
        float sum_exp = 0.0f;
        float exp_scores[SEQ_LEN];
        EXP_LOOP: for(int j = 0; j < SEQ_LEN; j++) {
            #pragma HLS PIPELINE II=1
            exp_scores[j] = hls::exp(scores[i][j] - max_val);
            sum_exp += exp_scores[j];
        }

        // OPTIMISATION CRITIQUE : Une seule division lourde ici au lieu de 128
        float inv_sum_exp = 1.0f / sum_exp; 

        /* 3. Somme pondérée de V (transformée en multiplications légères) */
        APPLY_LOOP_K: for(int k = 0; k < D_HEAD; k++) {
            #pragma HLS PIPELINE II=1
            float acc = 0.0f;
            APPLY_LOOP_J: for(int j = 0; j < SEQ_LEN; j++) {
                #pragma HLS UNROLL factor=4
                acc += (exp_scores[j] * inv_sum_exp) * (float)V[j][k];
            }
            Output[i][k] = (data_t)acc;
        }
    }
}

// ==================== LAYER NORMALIZATION ====================
void layer_norm1(
    const data_t Input[SEQ_LEN][D_MODEL],
    data_t Output[SEQ_LEN][D_MODEL]
) {
    #pragma HLS INLINE off
    NORM_SEQ: for(int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS PIPELINE off
        
        norm_t mean = 0.0f;
        MEAN_LOOP: for(int j = 0; j < D_MODEL; j++) {
            #pragma HLS PIPELINE II=1
            mean += (norm_t)Input[i][j];
        }
        mean = mean / (norm_t)D_MODEL;

        norm_t variance = 0.0f;
        VAR_LOOP: for(int j = 0; j < D_MODEL; j++) {
            #pragma HLS PIPELINE II=1
            norm_t diff = (norm_t)Input[i][j] - mean;
            variance += diff * diff;
        }
        variance = variance / (norm_t)D_MODEL;

        norm_t inv_std = 1.0f / hls::sqrt(variance + EPSILON);
        
        NORMALIZE_LOOP: for(int j = 0; j < D_MODEL; j++) {
            #pragma HLS PIPELINE II=1
            norm_t normalized = ((norm_t)Input[i][j] - mean) * inv_std;
            norm_t gamma_f = (norm_t)norm1_gamma_q[j] / 256.0f;
            norm_t beta_f  = (norm_t)norm1_beta_q[j]  / 256.0f;
            Output[i][j] = (data_t)(gamma_f * normalized + beta_f);
        }
    }
}

// ==================== MULTI-HEAD ATTENTION ====================
void multi_head_attention_improved(
    const data_t X[SEQ_LEN][D_MODEL],
    const data_t Wq[NUM_HEADS][D_MODEL][D_HEAD],
    const data_t Wk[NUM_HEADS][D_MODEL][D_HEAD],
    const data_t Wv[NUM_HEADS][D_MODEL][D_HEAD],
    const data_t Wo[D_MODEL][D_MODEL],
    data_t Output[SEQ_LEN][D_MODEL]
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=in_proj_bias_q complete dim=0
    #pragma HLS ARRAY_PARTITION variable=in_proj_bias_k complete dim=0
    #pragma HLS ARRAY_PARTITION variable=in_proj_bias_v complete dim=0
    data_t Q[SEQ_LEN][D_HEAD];
    data_t K[SEQ_LEN][D_HEAD];
    data_t V[SEQ_LEN][D_HEAD];
    data_t head_out[SEQ_LEN][D_HEAD];
    data_t concat[SEQ_LEN][D_MODEL];

    HEAD_LOOP: for(int h = 0; h < NUM_HEADS; h++) {
        matmul_opt(X,Wq[h], in_proj_bias_q[h], Q);
        matmul_opt(X, Wk[h], in_proj_bias_k[h], K);
        matmul_opt(X, Wv[h], in_proj_bias_v[h], V);

        attention_head(Q, K, V, head_out);

        CONCAT_I: for(int i = 0; i < SEQ_LEN; i++) {
            CONCAT_J: for(int j = 0; j < D_HEAD; j++) {
                #pragma HLS PIPELINE II=1
                concat[i][h * D_HEAD + j] = head_out[i][j];
            }
        }
    }

    // Projection finale Wo
    OUT_I: for(int i = 0; i < SEQ_LEN; i++) {
        OUT_J: for(int j = 0; j < D_MODEL; j++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = 0;
            OUT_K: for(int k = 0; k < D_MODEL; k++) {
                #pragma HLS UNROLL factor=4
                sum += concat[i][k] * Wo[k][j];
            }
            sum += (acc_t)out_proj_bias_q[j];
            Output[i][j] = (data_t)sum;
        }
    }
}

// ==================== FEED-FORWARD NETWORK ====================
void feed_forward_improved(
    const data_t Input[SEQ_LEN][D_MODEL],
    const data_t W1[D_MODEL][D_FF],
    const data_t W2[D_FF][D_MODEL],
    data_t Output[SEQ_LEN][D_MODEL]
) {
    #pragma HLS INLINE off

    data_t hidden[SEQ_LEN][D_FF];

    // Couche 1 + ReLU
    FF1_I: for(int i = 0; i < SEQ_LEN; i++) {
        FF1_J: for(int j = 0; j < D_FF; j++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = 0;
            FF1_K: for(int k = 0; k < D_MODEL; k++) {
                #pragma HLS UNROLL factor=4
                sum += Input[i][k] * W1[k][j];
            }
            sum += (acc_t)ffn0_bias_q[j];
            hidden[i][j] = (sum > 0) ? (data_t)sum : (data_t)0;
        }
    }

    // Couche 2
    FF2_I: for(int i = 0; i < SEQ_LEN; i++) {
        FF2_J: for(int j = 0; j < D_MODEL; j++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = 0;
            FF2_K: for(int k = 0; k < D_FF; k++) {
                #pragma HLS UNROLL factor=4
                sum += hidden[i][k] * W2[k][j];
            }
            sum += (acc_t)ffn2_bias_q[j];
            Output[i][j] = (data_t)sum;
        }
    }
}

// LayerNorm 2 (après FFN)
void layer_norm2(
    const data_t Input[SEQ_LEN][D_MODEL],
    data_t Output[SEQ_LEN][D_MODEL]
) {
    #pragma HLS INLINE off
    NORM_SEQ2: for(int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS PIPELINE off

        norm_t mean = 0.0f;
        MEAN_LOOP2: for(int j = 0; j < D_MODEL; j++) {
            #pragma HLS PIPELINE II=1
            mean += (norm_t)Input[i][j];
        }
        mean = mean / (norm_t)D_MODEL;

        norm_t variance = 0.0f;
        VAR_LOOP2: for(int j = 0; j < D_MODEL; j++) {
            #pragma HLS PIPELINE II=1
            norm_t diff = (norm_t)Input[i][j] - mean;
            variance += diff * diff;
        }
        variance = variance / (norm_t)D_MODEL;

        norm_t inv_std = 1.0f / hls::sqrt(variance + EPSILON);
        
        NORMALIZE_LOOP2: for(int j = 0; j < D_MODEL; j++) {
            #pragma HLS PIPELINE II=1
            norm_t normalized = ((norm_t)Input[i][j] - mean) * inv_std;
            norm_t gamma_f = (norm_t)norm2_gamma_q[j] / 256.0f;
            norm_t beta_f  = (norm_t)norm2_beta_q[j]  / 256.0f;
            Output[i][j] = (data_t)(gamma_f * normalized + beta_f);
        }
    }
}

// ==================== TOP FUNCTION ====================
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
) {
    #pragma HLS INTERFACE s_axilite port=return      bundle=control
    #pragma HLS INTERFACE s_axilite port=data_offset bundle=control
    #pragma HLS INTERFACE s_axilite port=wq_offset   bundle=control
    #pragma HLS INTERFACE s_axilite port=wk_offset   bundle=control
    #pragma HLS INTERFACE s_axilite port=wv_offset   bundle=control
    #pragma HLS INTERFACE s_axilite port=wo_offset   bundle=control
    #pragma HLS INTERFACE s_axilite port=w1_offset   bundle=control
    #pragma HLS INTERFACE s_axilite port=w2_offset   bundle=control
    #pragma HLS INTERFACE s_axilite port=out_offset  bundle=control

    #pragma HLS INTERFACE m_axi port=mem_in  offset=slave bundle=gmem0 depth=4096 latency=64
    #pragma HLS INTERFACE m_axi port=mem_out offset=slave bundle=gmem1 depth=256  latency=64

    #pragma HLS INTERFACE s_axilite port=mem_in  bundle=control
    #pragma HLS INTERFACE s_axilite port=mem_out bundle=control

    data_t X[SEQ_LEN][D_MODEL];
    static data_t Wq[NUM_HEADS][D_MODEL][D_HEAD];
    static data_t Wk[NUM_HEADS][D_MODEL][D_HEAD];
    static data_t Wv[NUM_HEADS][D_MODEL][D_HEAD];
    static data_t Wo[D_MODEL][D_MODEL];
    static data_t W1[D_MODEL][D_FF];
    static data_t W2[D_FF][D_MODEL];
    static bool weights_cached = false;   // <-- registre persistant, init a 0 au reset

    // Partitionnement cyclique cohérent avec l'unroll factor de 4
    #pragma HLS ARRAY_PARTITION variable=X cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=Wq cyclic factor=4 dim=3
    #pragma HLS ARRAY_PARTITION variable=Wk cyclic factor=4 dim=3
    #pragma HLS ARRAY_PARTITION variable=Wv cyclic factor=4 dim=3
    #pragma HLS ARRAY_PARTITION variable=Wo cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=W1 cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=W2 cyclic factor=4 dim=2

    data_t MHA_out[SEQ_LEN][D_MODEL];
    data_t MHA_residual[SEQ_LEN][D_MODEL];
    data_t MHA_norm[SEQ_LEN][D_MODEL];
    data_t FFN_out[SEQ_LEN][D_MODEL];
    data_t FFN_residual[SEQ_LEN][D_MODEL];
    data_t FFN_norm[SEQ_LEN][D_MODEL];

    #pragma HLS ARRAY_PARTITION variable=MHA_out cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=MHA_residual cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=MHA_norm cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=FFN_out cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=FFN_residual cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=FFN_norm cyclic factor=4 dim=2

    // --- Burst Reads from AXI ---
    READ_X_I: for(int i = 0; i < SEQ_LEN; i++) {
        READ_X_J: for(int j = 0; j < D_MODEL; j++) {
            #pragma HLS PIPELINE II=1
            X[i][j] = mem_in[data_offset + i * D_MODEL + j];
        }
    }

    ADD_PE_I: for(int i = 0; i < SEQ_LEN; i++) {
        ADD_PE_J: for(int j = 0; j < D_MODEL; j++) {
            #pragma HLS PIPELINE II=1
            X[i][j] = X[i][j] + positional_encoding[i][j];
        }
    }

    if (!weights_cached) {
        READ_HEADS: for(int h = 0; h < NUM_HEADS; h++) {
            READ_H_I: for(int i = 0; i < D_MODEL; i++) {
                READ_H_J: for(int j = 0; j < D_HEAD; j++) {
                    #pragma HLS PIPELINE II=1
                    int idx = h * D_MODEL * D_HEAD + i * D_HEAD + j;
                    Wq[h][i][j] = mem_in[wq_offset + idx];
                    Wk[h][i][j] = mem_in[wk_offset + idx];
                    Wv[h][i][j] = mem_in[wv_offset + idx];
                }
            }
        }

        READ_WO_I: for(int i = 0; i < D_MODEL; i++) {
            READ_WO_J: for(int j = 0; j < D_MODEL; j++) {
                #pragma HLS PIPELINE II=1
                Wo[i][j] = mem_in[wo_offset + i * D_MODEL + j];
            }
        }

        READ_W1_I: for(int i = 0; i < D_MODEL; i++) {
            READ_W1_J: for(int j = 0; j < D_FF; j++) {
                #pragma HLS PIPELINE II=1
                W1[i][j] = mem_in[w1_offset + i * D_FF + j];
            }
        }

        READ_W2_I: for(int i = 0; i < D_FF; i++) {
            READ_W2_J: for(int j = 0; j < D_MODEL; j++) {
                #pragma HLS PIPELINE II=1
                W2[i][j] = mem_in[w2_offset + i * D_MODEL + j];
            }
        }
            weights_cached = true;
    }    
    // ==================== PIPELINE EXECUTION (POST-LN) ====================
    multi_head_attention_improved(X, Wq, Wk, Wv, Wo, MHA_out);
    residual_add(X, MHA_out, MHA_residual);
    layer_norm1(MHA_residual, MHA_norm);

    feed_forward_improved(MHA_norm, W1, W2, FFN_out);
    residual_add(MHA_norm, FFN_out, FFN_residual);
    layer_norm2(FFN_residual, FFN_norm);

    // --- Burst Write to AXI ---
    WRITE_OUT_I: for(int i = 0; i < SEQ_LEN; i++) {
        WRITE_OUT_J: for(int j = 0; j < D_MODEL; j++) {
            #pragma HLS PIPELINE II=1
            mem_out[out_offset + i * D_MODEL + j] = FFN_norm[i][j];
        }
    }
}