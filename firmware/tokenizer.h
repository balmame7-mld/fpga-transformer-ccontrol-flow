
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>



// Token IDs
#define TOKEN_IF            1
#define TOKEN_ELSE          2
#define TOKEN_WHILE         3
#define TOKEN_FOR           4
#define TOKEN_SWITCH        5
#define TOKEN_CASE          6
#define TOKEN_RETURN        7
#define TOKEN_BREAK         8
#define TOKEN_INT           9
#define TOKEN_VOID          10

#define TOKEN_GREATER       20
#define TOKEN_LESS          21
#define TOKEN_EQUAL         22
#define TOKEN_NOT_EQUAL     23
#define TOKEN_GEQ           24
#define TOKEN_LEQ           25

#define TOKEN_ASSIGN        40
#define TOKEN_PLUS          41
#define TOKEN_MINUS         42
#define TOKEN_MULT          43
#define TOKEN_DIV           44
#define TOKEN_INCREMENT     45
#define TOKEN_DECREMENT     46
#define TOKEN_PLUS_ASSIGN   47
#define TOKEN_MINUS_ASSIGN  48
#define TOKEN_MULT_ASSIGN   49

#define TOKEN_LPAREN        50
#define TOKEN_RPAREN        51
#define TOKEN_LBRACE        52
#define TOKEN_RBRACE        53

#define TOKEN_VARIABLE      60
#define TOKEN_NUMBER        61

void tokenize(const char* code, int16_t* tokens, int max_tokens);
const char* get_example_code(int index);

#endif
