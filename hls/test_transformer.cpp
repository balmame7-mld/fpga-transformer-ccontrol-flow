
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "transformer_multihead.h"
#include "weights_fpga.h"
#include "embedding_layer.h"

// ================================================================
// VECTEURS DE RÉFÉRENCE PYTORCH (Bloc 12 Colab)
// ================================================================
// Exemple IF #3 : "if (count != max) { count++; }"
// Tokens : 1 50 60 23 60 51 52 60 45 54 53 0 0 0 0 0
static const int16_t REF_TOKENS_IF3[16] = {
    1, 50, 60, 23, 60, 51, 52, 60, 45, 54, 53, 0, 0, 0, 0, 0
};
// Mean-pooled Q8.8 attendu depuis PyTorch :
static const int16_t REF_MEANPOOL_IF3[16] = {
    -184, -132, 266, -319, 29, 188, 81, 7, 41, 304, 38, -307, 205, 73, -209, -83
};
// Classe attendue : 0 = IF
static const int EXPECTED_CLASS_IF3 = 0;

// ================================================================
// DIAGNOSTIC — divergence PyTorch vs materiel sur exemple "prefix"
// "int flag = 0; if (flag == 0) { flag = 1; }"  (idx 20 du dataset IF)
// ================================================================
static const int16_t REF_TOKENS_DIAG[16] = {
    9, 60, 40, 61, 54, 1, 50, 60, 22, 61, 51, 52, 60, 40, 61, 54
};
// Mean-pooled Q8.8 attendu depuis PyTorch (pooled x 256, arrondi) :
static const int16_t REF_MEANPOOL_DIAG[16] = {
    0, -54, -55, 278, 443, 69, -86, 178, -269, 50, -91, -440, -71, -258, 394, -39
};
// Classe attendue : 0 = IF (logits PyTorch : [1.02, -1.34, 1.69, -2.36])
static const int EXPECTED_CLASS_DIAG = 0;


// Exemple WHILE #1 : "while (i < 10) { i++; }"
static const int16_t REF_TOKENS_WHILE1[16] = {
    3, 50, 60, 21, 61, 51, 52, 60, 45, 45, 53, 0, 0, 0, 0, 0
};
static const int16_t REF_MEANPOOL_WHILE1[16] = {
    -62, -156, 218, -497, 101, 321, 260, -451, 330, 3, 26, -291, 20, -27, 113, 36
};
static const int EXPECTED_CLASS_WHILE1 = 1;

// ================================================================
// BUFFER MÉMOIRE PARTAGÉ (simuler la mémoire DDR)
// ================================================================
#define TOTAL_BUFFER_SIZE 8192
static data_t sim_buffer[TOTAL_BUFFER_SIZE];

// Offsets identiques à transformer_driver.c
#define SIM_OFFSET_DATA  0
#define SIM_OFFSET_WQ    256
#define SIM_OFFSET_WK    512
#define SIM_OFFSET_WV    768
#define SIM_OFFSET_WO    1024
#define SIM_OFFSET_W1    1280
#define SIM_OFFSET_W2    1792
#define SIM_OFFSET_OUT   2304

// ================================================================
// FONCTIONS UTILITAIRES
// ================================================================

// Fonction utilitaire : copie les bits d'un int16_t vers data_t sans conversion de valeur
// int16_t value = 9  →  data_t représente 9/256 = 0.035 (correct Q8.8)
// (data_t)9 aurait donné 9.0 = FAUX
inline data_t int16_to_fixed(int16_t v) {
    data_t tmp;
    tmp.range(15, 0) = v;  // copie bit-à-bit : 0x0009 → ap_fixed avec valeur 9/256
    return tmp;
}

// Ajouter cette fonction dans le testbench (symétrique à int16_to_fixed) :
inline int16_t fixed_to_int16(data_t v) {
    ap_uint<16> bits = v.range(15, 0);  // extraire les 16 bits bruts
    return (int16_t)(uint16_t)bits;      // reinterpréter comme int16_t signé
}

void load_weights_to_buffer(void) {
    for(int i = 0; i < 256; i++) {
        sim_buffer[SIM_OFFSET_WQ + i] = int16_to_fixed(attention_in_proj_W[i]);
        sim_buffer[SIM_OFFSET_WK + i] = int16_to_fixed(attention_in_proj_W[256 + i]);
        sim_buffer[SIM_OFFSET_WV + i] = int16_to_fixed(attention_in_proj_W[512 + i]);
    }
    for(int i = 0; i < 256; i++)
        sim_buffer[SIM_OFFSET_WO + i] = int16_to_fixed(attention_out_proj_W[i]);
    for(int i = 0; i < 512; i++)
        sim_buffer[SIM_OFFSET_W1 + i] = int16_to_fixed(ffn_0_W[i]);
    for(int i = 0; i < 512; i++)
        sim_buffer[SIM_OFFSET_W2 + i] = int16_to_fixed(ffn_2_W[i]);

    // Vérification : Wq[0]
    printf("[TB] Poids chargés (valeurs réelles) :\n");
    printf("     Wq[0]=%.4f  Wq[1]=%.4f  Wq[2]=%.4f\n",
           (float)sim_buffer[SIM_OFFSET_WQ],
           (float)sim_buffer[SIM_OFFSET_WQ+1],
           (float)sim_buffer[SIM_OFFSET_WQ+2]);
    // Attendu : ~0.0352  ~-0.363  ~-0.156
}

void load_embedding(const int16_t tokens[16]) {
    for(int i = 0; i < 16; i++) {
        int16_t tok = tokens[i];
        if(tok >= 0 && tok < 100) {
            for(int j = 0; j < 16; j++)
                sim_buffer[SIM_OFFSET_DATA + i*16 + j] =
                    int16_to_fixed(embedding_table[tok*16 + j]);
        } else {
            for(int j = 0; j < 16; j++)
                sim_buffer[SIM_OFFSET_DATA + i*16 + j] = (data_t)0;
        }
    }
}

// Calculer mean pooling sur la sortie (16 positions × 16 dims)
void compute_mean_pool(data_t* out_buf, int16_t result[16]) {
    for(int d = 0; d < 16; d++) {
        int32_t acc = 0;
        for(int pos = 0; pos < 16; pos++)
            acc += (int32_t)fixed_to_int16(out_buf[pos*16 + d]);  // ← fix ici
        result[d] = (int16_t)(acc >> 4);
    }
}

// Comparer deux vecteurs et afficher les écarts
int compare_vectors(const int16_t got[16], const int16_t expected[16],
                    const char* name) {
    int sign_errors = 0;
    float mse = 0.0f;
    printf("\n[TB] Comparaison %s :\n", name);
    printf("     Dim  | FPGA | PyTorch | Erreur | Signe\n");
    printf("     -----+------+---------+--------+------\n");
    for(int i = 0; i < 16; i++) {
        int err = (int)got[i] - (int)expected[i];
        mse += (float)err * err;
        int sign_ok = ((got[i] >= 0) == (expected[i] >= 0)) ||
                      (abs(expected[i]) < 5);  // proche de 0 → signe non significatif
        if(!sign_ok) sign_errors++;
        printf("     [%2d] | %5d | %7d | %6d | %s\n",
       i, (int16_t)got[i], expected[i], 
       (int)(int16_t)got[i] - (int)expected[i],
       sign_ok ? "OK" : "ERREUR");
    }
    mse /= 16.0f;
    float rmse = sqrtf(mse);
    printf("\n     RMSE = %.1f  |  Erreurs de signe = %d/16\n", rmse, sign_errors);
    // Dans compare_vectors et affichages, caster à int16_t pour printf :
    return sign_errors;
}

// Classifieur linéaire simplifié (sans classifier_weights.h pour garder simple)
// Pour le testbench, on affiche juste le vecteur et on compare avec référence
int classify_simple(const int16_t pooled[16]) {
    // Juste pour le testbench : regarder si IF ou WHILE est la classe correcte
    // basé sur la proximité avec les vecteurs de référence
    float dist_if = 0, dist_while = 0;
    for(int i = 0; i < 16; i++) {
        float d_if = (float)(pooled[i] - REF_MEANPOOL_IF3[i]);
        float d_wh = (float)(pooled[i] - REF_MEANPOOL_WHILE1[i]);
        dist_if    += d_if * d_if;
        dist_while += d_wh * d_wh;
    }
    printf("[TB] Distance L2 vers IF    : %.0f\n", sqrtf(dist_if));
    printf("[TB] Distance L2 vers WHILE : %.0f\n", sqrtf(dist_while));
    return (dist_if < dist_while) ? 0 : 1;
}

// ================================================================
// TEST PRINCIPAL
// ================================================================
int main(void) {
    printf("=================================================\n");
    printf("  TESTBENCH Transformer HLS - C Simulation\n");
    printf("=================================================\n\n");

    // Charger les poids
    load_weights_to_buffer();

    int total_tests = 0, passed = 0; 

    // ================================================================
    // TEST 1 — IF #3
    // ================================================================
    printf("\n--- TEST 1 : IF #3 ---\n");
    printf("Code   : if (count != max) { count++; }\n");
    printf("Tokens : ");
    for(int i = 0; i < 16; i++) printf("%d ", REF_TOKENS_IF3[i]);
    printf("\n");

    load_embedding(REF_TOKENS_IF3);

    // Appeler le Transformer HLS
    transformer_multihead(
        sim_buffer, sim_buffer,
        SIM_OFFSET_DATA,
        SIM_OFFSET_WQ, SIM_OFFSET_WK, SIM_OFFSET_WV,
        SIM_OFFSET_WO, SIM_OFFSET_W1, SIM_OFFSET_W2,
        SIM_OFFSET_OUT
    );

    // Afficher les 8 premières valeurs brutes
    // Remplacer la ligne d'affichage "Output brut" par :
    printf("[TB] Output brut (8 premières valeurs) : ");
    for(int i = 0; i < 8; i++)
        printf("%d ", fixed_to_int16(sim_buffer[SIM_OFFSET_OUT + i]));
    printf("\n");

    // Mean pooling
    int16_t pool1[16];
    compute_mean_pool(&sim_buffer[SIM_OFFSET_OUT], pool1);
    printf("[TB] Mean-pooled : ");
    for(int i = 0; i < 16; i++) printf("%d ", pool1[i]);
    printf("\n");

    // Comparer avec référence PyTorch
    int sign_err1 = compare_vectors(pool1, REF_MEANPOOL_IF3, "IF #3");

    total_tests++;
    if(sign_err1 == 0) {
        printf("[TB] TEST 1 : PASS ✅\n");
        passed++;
    } else {
        printf("[TB] TEST 1 : FAIL ❌ (%d erreurs de signe)\n", sign_err1);
    }


    printf("\n--- TEST DIAGNOSTIC : idx 20 (prefix) ---\n");
    printf("Code   : int flag = 0; if (flag == 0) { flag = 1; }\n");

    load_embedding(REF_TOKENS_DIAG);

    transformer_multihead(
        sim_buffer, sim_buffer,
        SIM_OFFSET_DATA,
        SIM_OFFSET_WQ, SIM_OFFSET_WK, SIM_OFFSET_WV,
        SIM_OFFSET_WO, SIM_OFFSET_W1, SIM_OFFSET_W2,
        SIM_OFFSET_OUT
    );

    int16_t pool_diag[16];
    compute_mean_pool(&sim_buffer[SIM_OFFSET_OUT], pool_diag);
    printf("[TB] Mean-pooled : ");
    for(int i = 0; i < 16; i++) printf("%d ", pool_diag[i]);
    printf("\n");

    int sign_err_diag = compare_vectors(pool_diag, REF_MEANPOOL_DIAG, "DIAGNOSTIC idx20");

    // ================================================================
    // TEST 2 — WHILE #1
    // ================================================================
    printf("\n--- TEST 2 : WHILE #1 ---\n");
    printf("Code   : while (i < 10) { i++; }\n");
    printf("Tokens : ");
    for(int i = 0; i < 16; i++) printf("%d ", REF_TOKENS_WHILE1[i]);
    printf("\n");

    load_embedding(REF_TOKENS_WHILE1);

    transformer_multihead(
        sim_buffer, sim_buffer,
        SIM_OFFSET_DATA,
        SIM_OFFSET_WQ, SIM_OFFSET_WK, SIM_OFFSET_WV,
        SIM_OFFSET_WO, SIM_OFFSET_W1, SIM_OFFSET_W2,
        SIM_OFFSET_OUT
    );
    printf("[TB] Output brut WHILE#1 (8 premières valeurs) : ");
    for(int i = 0; i < 8; i++)
        printf("%d ", fixed_to_int16(sim_buffer[SIM_OFFSET_OUT + i]));
    printf("\n"); // ajouté

    int16_t pool2[16];
    compute_mean_pool(&sim_buffer[SIM_OFFSET_OUT], pool2);
    printf("[TB] Mean-pooled : ");
    for(int i = 0; i < 16; i++) printf("%d ", pool2[i]);
    printf("\n");

    int sign_err2 = compare_vectors(pool2, REF_MEANPOOL_WHILE1, "WHILE #1");

    total_tests++;
    if(sign_err2 == 0) {
        printf("[TB] TEST 2 : PASS ✅\n");
        passed++;
    } else {
        printf("[TB] TEST 2 : FAIL ❌ (%d erreurs de signe)\n", sign_err2);
    }

    // ================================================================
    // RÉSUMÉ
    // ================================================================
    printf("\n=================================================\n");
    printf("  RÉSULTATS : %d/%d tests passés\n", passed, total_tests);
    printf("=================================================\n");
    printf("\nInterprétation des RMSE :\n");
    printf("  RMSE < 20  : Excellent (erreur Q8.8 normale)\n");
    printf("  RMSE < 50  : Acceptable\n");
    printf("  RMSE > 100 : Problème structurel\n");
    printf("  Erreurs signe = 0 : Classification correcte probable\n");
    printf("  Erreurs signe > 2 : Classification incorrecte probable\n");

    return (passed == total_tests) ? 0 : 1;
}
