#include <stdint.h>
#include <string.h>
#include "tokenizer.h"

static int is_alnum_(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

void tokenize(const char* code, int16_t* tokens, int max_tokens) {
    int token_idx = 0;
    int i = 0;
    int len = strlen(code);

    while(i < len && token_idx < max_tokens) {
        char c = code[i];

        if(c == ' ' || c == '\t' || c == '\n' || c == '\r') { i++; continue; }

        /* Mots-clés — plus longs d'abord, AVEC vérification de frontière de mot */
        if(strncmp(&code[i], "switch", 6) == 0 && !is_alnum_(code[i+6])) { tokens[token_idx++] = TOKEN_SWITCH; i += 6; }
        else if(strncmp(&code[i], "while", 5) == 0 && !is_alnum_(code[i+5])) { tokens[token_idx++] = TOKEN_WHILE; i += 5; }
        else if(strncmp(&code[i], "break", 5) == 0 && !is_alnum_(code[i+5])) { tokens[token_idx++] = TOKEN_BREAK; i += 5; }
        else if(strncmp(&code[i], "case", 4) == 0 && !is_alnum_(code[i+4])) { tokens[token_idx++] = TOKEN_CASE; i += 4; }
        else if(strncmp(&code[i], "else", 4) == 0 && !is_alnum_(code[i+4])) { tokens[token_idx++] = TOKEN_ELSE; i += 4; }
        else if(strncmp(&code[i], "void", 4) == 0 && !is_alnum_(code[i+4])) { tokens[token_idx++] = TOKEN_VOID; i += 4; }
        else if(strncmp(&code[i], "for", 3) == 0 && !is_alnum_(code[i+3])) { tokens[token_idx++] = TOKEN_FOR; i += 3; }
        else if(strncmp(&code[i], "int", 3) == 0 && !is_alnum_(code[i+3])) { tokens[token_idx++] = TOKEN_INT; i += 3; }
        else if(strncmp(&code[i], "if", 2) == 0 && !is_alnum_(code[i+2])) { tokens[token_idx++] = TOKEN_IF; i += 2; }

        /* Opérateurs composés (2 caractères) — AVANT les versions à 1 caractère */
        else if(strncmp(&code[i], "==", 2) == 0) { tokens[token_idx++] = TOKEN_EQUAL; i += 2; }
        else if(strncmp(&code[i], "!=", 2) == 0) { tokens[token_idx++] = TOKEN_NOT_EQUAL; i += 2; }
        else if(strncmp(&code[i], ">=", 2) == 0) { tokens[token_idx++] = TOKEN_GEQ; i += 2; }
        else if(strncmp(&code[i], "<=", 2) == 0) { tokens[token_idx++] = TOKEN_LEQ; i += 2; }
        else if(strncmp(&code[i], "++", 2) == 0) { tokens[token_idx++] = TOKEN_INCREMENT; i += 2; }
        else if(strncmp(&code[i], "--", 2) == 0) { tokens[token_idx++] = TOKEN_DECREMENT; i += 2; }
        else if(strncmp(&code[i], "+=", 2) == 0) { tokens[token_idx++] = TOKEN_PLUS_ASSIGN; i += 2; }
        else if(strncmp(&code[i], "-=", 2) == 0) { tokens[token_idx++] = TOKEN_MINUS_ASSIGN; i += 2; }
        else if(strncmp(&code[i], "*=", 2) == 0) { tokens[token_idx++] = TOKEN_MULT_ASSIGN; i += 2; }
        else if(strncmp(&code[i], "&&", 2) == 0) { i += 2; } /* toujours ignoré — limite connue, cf. point I passation */
        else if(strncmp(&code[i], "||", 2) == 0) { i += 2; }

        /* Opérateurs simples */
        else if(c == '>') { tokens[token_idx++] = TOKEN_GREATER; i++; }
        else if(c == '<') { tokens[token_idx++] = TOKEN_LESS; i++; }
        else if(c == '=') { tokens[token_idx++] = TOKEN_ASSIGN; i++; }
        else if(c == '+') { tokens[token_idx++] = TOKEN_PLUS; i++; }
        else if(c == '-') { tokens[token_idx++] = TOKEN_MINUS; i++; }
        else if(c == '*') { tokens[token_idx++] = TOKEN_MULT; i++; }
        else if(c == '/') { tokens[token_idx++] = TOKEN_DIV; i++; }

        /* Délimiteurs — inchangés */
        else if(c == '(') { tokens[token_idx++] = TOKEN_LPAREN; i++; }
        else if(c == ')') { tokens[token_idx++] = TOKEN_RPAREN; i++; }
        else if(c == '{') { tokens[token_idx++] = TOKEN_LBRACE; i++; }
        else if(c == '}') { tokens[token_idx++] = TOKEN_RBRACE; i++; }
        else if(c == ';') { tokens[token_idx++] = 54; i++; }
        else if(c == ':') { tokens[token_idx++] = 55; i++; }
        //else if(c == ';') { i++; }
        else if(c >= '0' && c <= '9') {
            tokens[token_idx++] = TOKEN_NUMBER;
            while(i < len && code[i] >= '0' && code[i] <= '9') i++;
        }
        else if((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
            tokens[token_idx++] = TOKEN_VARIABLE;
            while(i < len && is_alnum_(code[i])) i++;
        }
        else { i++; }
    }
    while(token_idx < max_tokens) tokens[token_idx++] = 0;
}