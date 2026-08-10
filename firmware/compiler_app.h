#ifndef COMPILER_APP_H
#define COMPILER_APP_H

#include <stdint.h>

// ==================== TYPES DE PATTERNS ====================

typedef enum {
    PATTERN_UNKNOWN = 0,     // Pattern non reconnu
    PATTERN_IF_SIMPLE,       // if (condition) { ... }
    PATTERN_WHILE_LOOP,      // while (condition) { ... }
    PATTERN_FOR_LOOP,        // for (init; cond; incr) { ... }
    PATTERN_SWITCH_CASE      // switch (var) { case ... }
} PatternType;

// ==================== FONCTIONS PUBLIQUES ====================

/**
 * Compile du code C simple vers Assembly ARM
 * Pipeline complet: Tokenization → FPGA → Classification → Code Gen
 * @param code Source code C (string)
 */
void compile_code(const char* code);

/**
 * Classifie le pattern de code à partir de l'embedding Transformer
 * @param embedding Vecteur d'embedding de 16 valeurs (sortie Transformer)
 * @return Type de pattern détecté
 */
PatternType classify_pattern(int16_t* embedding);

/**
 * Génère le code Assembly ARM pour un pattern donné
 * @param pattern Type de pattern (IF, WHILE, FOR, SWITCH)
 * @param code Code source original (pour commentaires)
 */
void generate_assembly(PatternType pattern, const char* code);

/**
 * Menu interactif avec exemples prédéfinis
 */
void compiler_menu(void);

#endif // COMPILER_APP_H