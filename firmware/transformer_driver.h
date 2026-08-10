#ifndef TRANSFORMER_DRIVER_H
#define TRANSFORMER_DRIVER_H

#include <stdint.h>

/* Configuration */
#define TRANSFORMER_BASE    0x40000000  /* Adresse Vivado confirmee */
// Accès au buffer interne pour signature
extern short* transformer_get_output_buffer(void);
#define TRANSFORMER_OUTPUT_SIZE (16 * 16)  // SEQ_LEN * EMBED_DIM

/* Parametres du transformer */
#define SEQ_LEN             16
#define D_MODEL             16
#define EMBED_DIM           16
#define NUM_HEADS           2
#define D_HEAD              8
#define FF_DIM              32

/* Registres AXI-Lite (selon rapport HLS) */
#define REG_AP_CTRL         0x00
#define REG_GIE             0x04
#define REG_IER             0x08
#define REG_ISR             0x0C
#define REG_MEM_IN_1        0x10
#define REG_MEM_IN_2        0x14
#define REG_MEM_OUT_1       0x1C
#define REG_MEM_OUT_2       0x20
#define REG_DATA_OFFSET     0x28  /* Offset corrige */
#define REG_WQ_OFFSET       0x30
#define REG_WK_OFFSET       0x38
#define REG_WV_OFFSET       0x40
#define REG_WO_OFFSET       0x48
#define REG_W1_OFFSET       0x50
#define REG_W2_OFFSET       0x58
#define REG_OUT_OFFSET      0x60

/* Bits de controle */
#define AP_CTRL_START       0x01
#define AP_CTRL_DONE        0x02
#define AP_CTRL_IDLE        0x04
#define AP_CTRL_READY       0x01

/* Tailles memoire */
#define MEM_INPUT_SIZE      (SEQ_LEN * D_MODEL)
#define MEM_WEIGHTS_SIZE    4096
#define MEM_OUTPUT_SIZE     (SEQ_LEN * D_MODEL)

/* Status bits */
#define STATUS_READY        0x00000002
#define STATUS_DONE         0x00000004

/* Fonctions */
void transformer_init(void);
int transformer_process(int16_t *input_tokens, int16_t *output_embedding);
void transformer_load_trained_weights(void);
uint32_t transformer_get_status(void);
int transformer_is_ready(void);

uint32_t transformer_get_last_dispatch_us(void);
uint32_t transformer_get_last_core_us(void);
uint32_t transformer_get_last_readback_us(void);

#endif /* TRANSFORMER_DRIVER_H */