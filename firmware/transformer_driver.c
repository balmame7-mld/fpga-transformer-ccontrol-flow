#include "transformer_driver.h"
#include "xil_printf.h"
#include "xil_io.h"
#include "xil_cache.h"
#include "sleep.h"
#include "weights_fpga.h"  // Poids entraînés
#include "embedding_layer.h"

/* Dimensions */
#define INPUT_SIZE (SEQ_LEN * EMBED_DIM)      
#define WEIGHTS_SIZE (EMBED_DIM * EMBED_DIM)  
#define W1_SIZE (EMBED_DIM * FF_DIM)          
#define W2_SIZE (FF_DIM * EMBED_DIM)          
#define OUTPUT_SIZE (SEQ_LEN * EMBED_DIM)     

/* Offsets dans le buffer (en elements) */
#define OFFSET_DATA  0
#define OFFSET_WQ    (OFFSET_DATA + INPUT_SIZE)
#define OFFSET_WK    (OFFSET_WQ + WEIGHTS_SIZE)
#define OFFSET_WV    (OFFSET_WK + WEIGHTS_SIZE)
#define OFFSET_WO    (OFFSET_WV + WEIGHTS_SIZE)
#define OFFSET_W1    (OFFSET_WO + WEIGHTS_SIZE)
#define OFFSET_W2    (OFFSET_W1 + W1_SIZE)
#define OFFSET_OUT   (OFFSET_W2 + W2_SIZE)

#define TOTAL_SIZE   (OFFSET_OUT + OUTPUT_SIZE)

#define GLOBAL_TIMER_BASE 0xF8F00200
static inline uint32_t read_gtimer_lo(void) {
    return Xil_In32(GLOBAL_TIMER_BASE);  /* COUNT_LO, offset +0x00 */
}
/* Timer @ CPU_FREQ/2 = 333 MHz -> 1 tick ~= 3 ns -> ticks*3/1000 = µs */
#define TICKS_TO_US(t) (((uint32_t)(t) * 3) / 1000)

static uint32_t g_last_dispatch_us = 0;
static uint32_t g_last_core_us     = 0;
static uint32_t g_last_readback_us = 0;

uint32_t transformer_get_last_dispatch_us(void) { return g_last_dispatch_us; }
uint32_t transformer_get_last_core_us(void)     { return g_last_core_us; }
uint32_t transformer_get_last_readback_us(void) { return g_last_readback_us; }

/* Fixed-point Q8.8 */
#define FLOAT_TO_FIX(x) ((short)((x) * 256.0f))

/* Buffer DDR global */
static short __attribute__((aligned(64))) transformer_buffer[TOTAL_SIZE];
static int transformer_initialized = 0;

/* Initialiser les poids (une fois) */
static void init_transformer_weights(void) {
    if (transformer_initialized) {
        return;  // Déjà initialisé
    }
    
    xil_printf("[DRIVER] Initializing DEFAULT weights...\n\r");
    
    /* Poids par défaut (identité * 0.25) */
    for(int i = 0; i < WEIGHTS_SIZE; i++) {
        int row = i / EMBED_DIM;
        int col = i % EMBED_DIM;
        short val = (row == col) ? FLOAT_TO_FIX(0.25f) : 0;
        transformer_buffer[OFFSET_WQ + i] = val;
        transformer_buffer[OFFSET_WK + i] = val;
        transformer_buffer[OFFSET_WV + i] = val;
        transformer_buffer[OFFSET_WO + i] = val;
    }
    
    for(int i = 0; i < W1_SIZE; i++) {
        transformer_buffer[OFFSET_W1 + i] = FLOAT_TO_FIX(0.02f);
    }
    
    for(int i = 0; i < W2_SIZE; i++) {
        transformer_buffer[OFFSET_W2 + i] = FLOAT_TO_FIX(0.02f);
    }
    
    transformer_initialized = 1;
    xil_printf("[DRIVER] Default weights initialized\n\r");
}

void transformer_load_trained_weights(void) {
    xil_printf("[DRIVER] Loading trained weights from PyTorch model...\n\r");
    
    // ========== ATTENTION WEIGHTS ==========
    xil_printf("   Loading attention weights (768 params)...\n\r");
    
    for(int i = 0; i < 256; i++) {
        transformer_buffer[OFFSET_WQ + i] = attention_in_proj_W[i];
    }
    
    for(int i = 0; i < 256; i++) {
        transformer_buffer[OFFSET_WK + i] = attention_in_proj_W[256 + i];
    }
    
    for(int i = 0; i < 256; i++) {
        transformer_buffer[OFFSET_WV + i] = attention_in_proj_W[512 + i];
    }
    
    xil_printf("   Loading Wo (256 params)...\n\r");
    for(int i = 0; i < 256; i++) {
        transformer_buffer[OFFSET_WO + i] = attention_out_proj_W[i];
    }
    
    // ========== FFN WEIGHTS ==========
    xil_printf("   Loading FFN weights (1024 params)...\n\r");
    
    for(int i = 0; i < 512; i++) {
        transformer_buffer[OFFSET_W1 + i] = ffn_0_W[i];
    }
    
    for(int i = 0; i < 512; i++) {
        transformer_buffer[OFFSET_W2 + i] = ffn_2_W[i];
    }
    
    // ========== FLUSH CACHE ==========
    Xil_DCacheFlushRange((INTPTR)transformer_buffer, TOTAL_SIZE * sizeof(short));
    
    // ========== MARQUER COMME INITIALISÉ ==========
    transformer_initialized = 1;  // ← AJOUTER CETTE LIGNE
    
    xil_printf("[DRIVER] Trained weights loaded successfully!\n\r");
    xil_printf("   Total loaded: %d parameters\n\r", 256+256+256+256+512+512);

     // ========== DEBUG: AFFICHER QUELQUES POIDS ==========
    xil_printf("\n\r[DEBUG] First 10 weights in Wq:\n\r");
    for(int i = 0; i < 10; i++) {
        xil_printf("  Wq[%d] = %d\n\r", i, transformer_buffer[OFFSET_WQ + i]);
    }
    
    xil_printf("\n\r[DEBUG] First 10 weights in W1:\n\r");
    for(int i = 0; i < 10; i++) {
        xil_printf("  W1[%d] = %d\n\r", i, transformer_buffer[OFFSET_W1 + i]);
    }
    // =====================================================
}

void transformer_init(void) {
    xil_printf("[DRIVER] Initializing transformer at 0x%08X\n\r", 
               TRANSFORMER_BASE);
    
    /* Reset */
    Xil_Out32(TRANSFORMER_BASE + REG_AP_CTRL, 0);
    usleep(1000);
    
    /* Verifier status */
    uint32_t ap_ctrl = Xil_In32(TRANSFORMER_BASE + REG_AP_CTRL);
    xil_printf("[DRIVER] ap_ctrl: 0x%08X\n\r", ap_ctrl);
    
    if(ap_ctrl & AP_CTRL_IDLE) {
        xil_printf("[DRIVER] Transformer ready\n\r");
    } else {
        xil_printf("[DRIVER] Warning: Transformer not idle\n\r");
    }
    
    /* Initialiser les poids */
    init_transformer_weights();
}

static void apply_embedding_layer(int16_t tokens[SEQ_LEN], int16_t embedded[SEQ_LEN][EMBED_DIM]) {
    /* embedding_table est déjà quantifié en Q8.8 (valeurs ±3 typiques)
     * Compatible directement avec data_t = ap_fixed<16,8> côté HLS
     * AUCUNE transformation supplémentaire nécessaire — lookup direct */

    for(int i = 0; i < SEQ_LEN; i++) {
        int16_t token = tokens[i];

        if(token >= 0 && token < VOCAB_SIZE) {
            for(int j = 0; j < EMBED_DIM; j++) {
                embedded[i][j] = embedding_table[token * EMBED_DIM + j];
            }
        } else {
            for(int j = 0; j < EMBED_DIM; j++) {
                embedded[i][j] = 0;
            }
        }
    }

    //xil_printf("[DRIVER] Embedding (Q8.8 direct, no scaling):\n\r");
    /*xil_printf("   Token %d -> [%d, %d, %d, %d]\n\r",
               tokens[0],
               embedded[0][0], embedded[0][1], embedded[0][2], embedded[0][3]); */
}

int transformer_process(int16_t *input_tokens, int16_t *output_embedding) {
    /* 1. Embedding (hors chronométrage — travail logiciel pur) */
    int16_t embedded[SEQ_LEN][EMBED_DIM];
    apply_embedding_layer(input_tokens, embedded);

    for(int i = 0; i < SEQ_LEN; i++) {
        for(int j = 0; j < EMBED_DIM; j++) {
            transformer_buffer[OFFSET_DATA + i * EMBED_DIM + j] = embedded[i][j];
        }
    }

    uint32_t t0 = read_gtimer_lo();

    /* --- DISPATCH : flush + config registres --- */
    Xil_DCacheFlushRange((INTPTR)&transformer_buffer[OFFSET_DATA], INPUT_SIZE * sizeof(short));

    uint32_t base_addr = (uint32_t)transformer_buffer;
    Xil_Out32(TRANSFORMER_BASE + REG_MEM_IN_1, base_addr);
    Xil_Out32(TRANSFORMER_BASE + REG_MEM_IN_2, 0U);
    Xil_Out32(TRANSFORMER_BASE + REG_MEM_OUT_1, base_addr);
    Xil_Out32(TRANSFORMER_BASE + REG_MEM_OUT_2, 0U);
    Xil_Out32(TRANSFORMER_BASE + REG_DATA_OFFSET, OFFSET_DATA);
    Xil_Out32(TRANSFORMER_BASE + REG_WQ_OFFSET, OFFSET_WQ);
    Xil_Out32(TRANSFORMER_BASE + REG_WK_OFFSET, OFFSET_WK);
    Xil_Out32(TRANSFORMER_BASE + REG_WV_OFFSET, OFFSET_WV);
    Xil_Out32(TRANSFORMER_BASE + REG_WO_OFFSET, OFFSET_WO);
    Xil_Out32(TRANSFORMER_BASE + REG_W1_OFFSET, OFFSET_W1);
    Xil_Out32(TRANSFORMER_BASE + REG_W2_OFFSET, OFFSET_W2);
    Xil_Out32(TRANSFORMER_BASE + REG_OUT_OFFSET, OFFSET_OUT);

    uint32_t t1 = read_gtimer_lo();

    /* --- CŒUR : strictement le calcul FPGA --- */
    Xil_Out32(TRANSFORMER_BASE + REG_AP_CTRL, AP_CTRL_START);
    while(!(Xil_In32(TRANSFORMER_BASE + REG_AP_CTRL) & AP_CTRL_DONE));

    uint32_t t2 = read_gtimer_lo();

    /* --- READBACK : invalidation cache + mean pooling --- */
    Xil_DCacheInvalidateRange((INTPTR)&transformer_buffer[OFFSET_OUT], OUTPUT_SIZE * sizeof(short));

    for(int i = 0; i < SEQ_LEN; i++) {
        int32_t sum = 0;
        for(int j = 0; j < EMBED_DIM; j++) {
            sum += transformer_buffer[OFFSET_OUT + i * EMBED_DIM + j];
        }
        output_embedding[i] = (int16_t)sum;
    }

    uint32_t t3 = read_gtimer_lo();

    g_last_dispatch_us = TICKS_TO_US(t1 - t0);
    g_last_core_us     = TICKS_TO_US(t2 - t1);
    g_last_readback_us = TICKS_TO_US(t3 - t2);

    return 0;
}


/* 
int transformer_process(int16_t *input_tokens, int16_t *output_embedding) {
    xil_printf("[DRIVER] Processing sequence: ");
    for(int i = 0; i < SEQ_LEN; i++) {
        xil_printf("%d ", input_tokens[i]);
    }
    xil_printf("\n\r");
    
    // Appliquer embedding layer puis copier dans buffer 
    int16_t embedded[SEQ_LEN][EMBED_DIM];
    apply_embedding_layer(input_tokens, embedded);

    xil_printf("[DRIVER] Embedding applied (first token embeddings):\n\r");
    xil_printf("   Token %d → [%d, %d, %d, %d]\n\r", 
            input_tokens[0],
            embedded[0][0], embedded[0][1], embedded[0][2], embedded[0][3]);

    // Copier dans buffer DDR
    for(int i = 0; i < SEQ_LEN; i++) {
        for(int j = 0; j < EMBED_DIM; j++) {
            transformer_buffer[OFFSET_DATA + i * EMBED_DIM + j] = embedded[i][j];
        }
    }
    
    // Clear output 
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        transformer_buffer[OFFSET_OUT + i] = 0;
    }
    
    // Flush cache 
    Xil_DCacheFlushRange((INTPTR)transformer_buffer, TOTAL_SIZE * sizeof(short));
    
    // Configurer l'IP 
    uint32_t base_addr = (uint32_t)transformer_buffer;
    
    // mem_in address (64-bit split en 2x32-bit) 
    Xil_Out32(TRANSFORMER_BASE + REG_MEM_IN_1, (uint32_t)(base_addr & 0xFFFFFFFF));
    Xil_Out32(TRANSFORMER_BASE + REG_MEM_IN_2, 0U);
    
    // mem_out address
    Xil_Out32(TRANSFORMER_BASE + REG_MEM_OUT_1, (uint32_t)(base_addr & 0xFFFFFFFF));
    Xil_Out32(TRANSFORMER_BASE + REG_MEM_OUT_2, 0U);
    
    // Offsets (en elements, pas en bytes) 
    Xil_Out32(TRANSFORMER_BASE + REG_DATA_OFFSET, OFFSET_DATA);
    Xil_Out32(TRANSFORMER_BASE + REG_WQ_OFFSET, OFFSET_WQ);
    Xil_Out32(TRANSFORMER_BASE + REG_WK_OFFSET, OFFSET_WK);
    Xil_Out32(TRANSFORMER_BASE + REG_WV_OFFSET, OFFSET_WV);
    Xil_Out32(TRANSFORMER_BASE + REG_WO_OFFSET, OFFSET_WO);
    Xil_Out32(TRANSFORMER_BASE + REG_W1_OFFSET, OFFSET_W1);
    Xil_Out32(TRANSFORMER_BASE + REG_W2_OFFSET, OFFSET_W2);
    Xil_Out32(TRANSFORMER_BASE + REG_OUT_OFFSET, OFFSET_OUT);
    
    // Demarrer 
    Xil_Out32(TRANSFORMER_BASE + REG_AP_CTRL, AP_CTRL_START);
    
    // Attendre DONE 
    int timeout = 10000000;
    while(timeout > 0) {
        uint32_t ap_ctrl = Xil_In32(TRANSFORMER_BASE + REG_AP_CTRL);
        
        if(ap_ctrl & AP_CTRL_DONE) {
            xil_printf("[DRIVER] Processing complete (ap_ctrl: 0x%08X)\n\r", ap_ctrl);
            
            // Invalider cache 
            Xil_DCacheInvalidateRange((INTPTR)&transformer_buffer[OFFSET_OUT], 
                                    OUTPUT_SIZE * sizeof(short));
            
            // ========== DEBUG: SIGNATURE COMPLET ==========
            // Calculer une "signature" à partir des 256 valeurs
            int32_t sig_sum = 0;        // Somme totale
            int32_t sig_pos_count = 0;  // Nombre de valeurs positives
            int32_t sig_neg_count = 0;  // Nombre de valeurs négatives
            int32_t sig_zero_count = 0; // Nombre de zéros
            int32_t sig_weighted = 0;   // Somme pondérée par position
            
            for(int i = 0; i < OUTPUT_SIZE; i++) {
                int16_t val = transformer_buffer[OFFSET_OUT + i];
                sig_sum += val;
                if(val > 0) sig_pos_count++;
                else if(val < 0) sig_neg_count++;
                else sig_zero_count++;
                sig_weighted += val * (i + 1);  // Pondérer par position
            }
            
            xil_printf("\n\r[SIGNATURE] Output Transformer:\n\r");
            xil_printf("   Sum:      %d\n\r", (int)sig_sum);
            xil_printf("   Positive: %d\n\r", (int)sig_pos_count);
            xil_printf("   Negative: %d\n\r", (int)sig_neg_count);
            xil_printf("   Zeros:    %d\n\r", (int)sig_zero_count);
            xil_printf("   Weighted: %d\n\r", (int)sig_weighted);
            
            // Afficher les 256 valeurs en ligne compacte
            xil_printf("\n\r[SIGNATURE] Full 256 values:\n\r");
            for(int i = 0; i < OUTPUT_SIZE; i++) {
                int16_t val = transformer_buffer[OFFSET_OUT + i];
                xil_printf("%2d ", val);  // Afficher nombre
                if((i+1) % 32 == 0) xil_printf("\n\r");  // Retour ligne tous les 32
            }
            xil_printf("\n\r");
            // =============================================
            
            // Copier output 
            for(int i = 0; i < SEQ_LEN; i++) {
                int32_t sum = 0;
                for(int j = 0; j < EMBED_DIM; j++) {
                    sum += transformer_buffer[OFFSET_OUT + i * EMBED_DIM + j];
                }
                output_embedding[i] = (int16_t)sum;
            }
            xil_printf("\n\r[DEBUG] Embedding vector (sum over 16 dims):\n\r");
            xil_printf("  [");
            for(int i = 0; i < 8; i++) {
                xil_printf("%d", output_embedding[i]);
                if(i < 7) xil_printf(", ");
            }
            xil_printf(", ...]\n\r");
            return 0;
        }
        
        usleep(10);
        timeout--;
    }
    
    xil_printf("[DRIVER] ERROR: Timeout\n\r");
    return -1;
}
*/

short* transformer_get_output_buffer(void) {
    return &transformer_buffer[OFFSET_OUT];
}

int transformer_is_ready(void) {
    uint32_t ap_ctrl = Xil_In32(TRANSFORMER_BASE + REG_AP_CTRL);
    return (ap_ctrl & (AP_CTRL_IDLE | AP_CTRL_READY)) != 0;
}

uint32_t transformer_get_status(void) {
    return Xil_In32(TRANSFORMER_BASE + REG_AP_CTRL);
}