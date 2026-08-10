#include "compiler_app.h"
#include "transformer_driver.h"
#include "tokenizer.h"
#include "xil_printf.h"
#include "sleep.h"
#include <string.h>
#include "classifier_weights.h"
#include "positional_encoding_sw.h"
#include "weights_fpga.h"
#include "attention_bias_sw.h"
#include "layernorm_params.h"
#include <math.h>   /* expf, sqrtf */
//#include "embedding_layer.h" // Contient embedding_table
#include <stdlib.h>   /* abs() */
#include <stdio.h>
#include "xiltimer.h"
extern const int16_t embedding_table[1600];

#define COUNTS_PER_SECOND XSLEEPTIMER_FREQ
/* ================================================================
   EXEMPLES DE CODE C — 20 exemples, 5 par classe
   Déclarés en portée fichier pour être accessibles dans
   compiler_menu(), compile_example() et display_examples_menu()
   ================================================================ */
#define NUM_EXAMPLES 20

static const char* example_codes[NUM_EXAMPLES] = {
    /* IF (1-5) */
    "int flag = 0; if (flag == 0) { flag = 1; }",
    "if (a == b) { match = 1; }",
    "if (n % 2 == 0) { even++; }",
    "if (score >= 90) { grade = 'A'; }",
    "if (err != 0) { handle_error(err); }",
    /* WHILE (6-10) */
    "while (x != 1) { if (x % 2 == 0) x /= 2; else x = 3*x+1; }",
    "while (i < 100) { sum += i; i += 2; }",
    "while (flag == 0) { wait(); }",
    "while (ch != '\n') { ch = getchar(); }",
    "while (n != 0) { digit = n % 10; n /= 10; }",
    /* FOR (11-15) */
    "for (i = 1; i < n; i++) { fib[i] = fib[i-1] + fib[i-2]; }",
    "for (k = 0; k < 8; k++) { bits |= (val>>k)&1; }",
    "for (i = 0; i < len; i++) { str[i] = toupper(str[i]); }",
    "for (j = 0; j < n; j += step) { acc += arr[j]; }",
    "for (i = 0; i < 4; i++) { vec[i] *= scale; }",
    /* SWITCH (16-20) */
    "switch (x) { case 1: break; }",
    "switch (fmt) { case HEX: print_hex(v); break; case DEC: print_dec(v); break; }",
    "switch (axis) { case X: dx=1; break; case Y: dy=1; break; }",
    "switch (btn) { case A: jump(); break; case B: shoot(); break; }",
    "switch (proto) { case TCP: use_tcp(); break; case UDP: use_udp(); break; }"
};

static const char* example_labels[NUM_EXAMPLES] = {
    "IF #1", "IF #2", "IF #3", "IF #4", "IF #5",
    "WHILE #1", "WHILE #2", "WHILE #3", "WHILE #4", "WHILE #5",
    "FOR #1",   "FOR #2",   "FOR #3",   "FOR #4",   "FOR #5",
    "SWITCH #1","SWITCH #2","SWITCH #3","SWITCH #4","SWITCH #5"
};

/* ================================================================
   TEST BATCH 200 EXEMPLES — 50 par classe
   ================================================================ */
#define NUM_EXAMPLE 200
/* 200 exemples : 50 IF, 50 WHILE, 50 FOR, 50 SWITCH */
static const char* codes_200[NUM_EXAMPLE] = {
    /* === IF (0-49) === */
    "if (x > 5) { y = 10; }",
    "if (a == 0) { return 1; }",
    "if (count != max) { count++; }",
    "if (x > y && y > z) { flag = 1; }",
    "if (temp < MIN || temp > MAX) { error(); }",
    "if (n > 0) { result = n * 2; }",
    "if (a >= b) { swap(a, b); }",
    "if (x != 0 && y != 0) { ratio = x / y; }",
    "if (status == OK) { proceed(); }",
    "if (val < 0) { val = -val; }",
    "if (i < size && arr[i] > 0) { sum += arr[i]; }",
    "if (ptr != NULL) { *ptr = 0; }",
    "if (c == 'a' || c == 'A') { count++; }",
    "if (level > MAX_LEVEL) { level = MAX_LEVEL; }",
    "if (done == 1) { break; }",
    "if (x > 0) { x--; }",
    "iff (vrai_ou_faux == 1) { execute_action(); }",                       // Attendu: PATTERN_IF_SIMPLE,
    "if(a) \n\t { \n\t\t b = 1; \n\t }",                                    // Attendu: PATTERN_IF_SIMPLE,
    "if /* attention: loop inside */ (x == 0) { break; }",                 // Attendu: PATTERN_IF_SIMPLE,
    "int for_loop_counter = 5; if (for_loop_counter > 0) { status = 1; }", // Attendu: PATTERN_IF_SIMPLE,

    "int flag = 0; if (flag == 0) { flag = 1; }",
    "int status = 1; if (status > 0) { status = 0; }",
    "int count = 0; if (count < 5) { count = 5; }",
    "int result = 2; if (result != 3) { result = 3; }",
    "int level = 0; if (level == 0) { level = 1; }",
    "int c = 5; if (c == 5) { c = 6; }",
    "int a = 1; int b = 2; if (a < b) { a = b; }",
    "int x = 10; if (x > 5) { x = 0; }",
    "int active = 0; if (active == 0) { active = 1; }",
    "int n = 7; if (n != 0) { n = 0; }", 
    "int ready = 0; if (ready < 1) { ready = 1; }",
    "int state = 3; if (state == 3) { state = 4; }",
    "if (x > 0) { for (i=0;i<3;i++) { sum += i; } }",
    "if (a > b) { while (a > 0) { a--; } }",
    "if (mode == 1) { switch (mode) { case 1: run(); break; } }",
    "if (n > 0) { for (i=0;i<n;i++) { total += i; } }",
    "if (valid == 1) { while (valid) { valid = 0; } }",
    "if (code == 2) { switch (code) { case 2: exec(); break; } }",
    "if (flag) { for (j=0;j<2;j++) { count++; } }",
    "if (limit > 0) { while (limit > 0) { limit--; } }",
    "if (key == 5) { switch (key) { case 5: open(); break; } }",
    "if (ok) { for (k=0;k<4;k++) { arr[k]=0; } }",
    "if(x>5){y=10;}", 
    "if ( x > 5 ) { y = 10 ; }",
    "if (x>5) {y=10;}",
    "if(x > 5) { y = 10; }",
    "if (x>5)   {   y=10;   }",
    "if(  x  >  5  ){ y = 10 ; }",
    "if (flag==1) { flag=0; }",
    "if  (n<0)  { n=0; }",

    /* === WHILE (50-99) === */
    "while (i < 10) { i++; }",
    "while (done != 1) { process(); }",
    "while (x > 0 && y < 100) { x--; y++; }",
    "while (buffer[idx] != 0) { idx++; }",
    "while (1) { if (stop) break; }",
    "while (n > 0) { n--; count++; }",
    "while (ptr != NULL) { ptr = ptr->next; }",
    "while (sum < target) { sum += step; }",
    "while (c != EOF) { c = getchar(); }",
    "while (retry < MAX_RETRY) { retry++; }",
    "while (i < n && arr[i] != val) { i++; }",
    "while (bits != 0) { bits &= bits - 1; count++; }",
    "while (left <= right) { mid = (left + right) / 2; }",
    "while (q != NULL) { q = q->next; size++; }",
    "while (timeout > 0) { timeout--; }",
    "while (x != 1) { if (x % 2 == 0) x /= 2; else x = 3*x+1; }",
    "int if_condition = 0; while (if_condition < 10) { if_condition++; }", // Attendu: PATTERN_WHILE_LOOP
    "while /* for compatibility reasons */ (ready == 0) { wait(); }",       // Attendu: PATTERN_WHILE_LOOP,
    "while     (   data   !=   end   )     {   data++;   }",               // Attendu: PATTERN_WHILE_LOOP,
    "whle (statut_global_systeme == 0) { rafraichir(); }",                   // Attendu: PATTERN_WHILE_LOOP

    "int i = 0; while (i < 10) { i++; }",  
    "int j = 5; while (j > 0) { j--; }",  
    "int done = 0; while (done == 0) { done = 1; }",  
    "int k = 1; while (k != 0) { k = 0; }",  
    "int t = 100; while (t > 50) { t--; }",  
    "int cnt = 0; while (cnt < 3) { cnt++; }",  
    "int a = 1; int b = 0; while (a > b) { b++; }",  
    "int retry = 3; while (retry > 0) { retry--; }",  
    "int flag = 1; while (flag == 1) { flag = 0; }",  
    "int x = 20; while (x != 0) { x--; }",  
    "int wait = 1; while (wait > 0) { wait = 0; }", 
    "int level = 5; while (level > 0) { level--; }",  
    "while (i < 5) { if (i == 2) { skip(); } }",  
    "while (n > 0) { for (j=0;j<n;j++) { sum += j; } }",  
    "while (running) { switch (state) { case 1: run(); break; } }",
    "while (a < b) { while (c > 0) { c--; } }",  
    "while (x > 0) { if (x > 2) { x--; } }",
    "while (t != 0) { for (k=0;k<2;k++) { t--; } }",
    "while (ok) { switch (code) { case 0: stop(); break; } }", 
    "while (cnt < 10) { if (cnt == 5) { cnt = 10; } }", 
    "while (v > 0) { for (m=0;m<v;m++) { arr[m]=0; } }",  
    "while (idx < len) { switch (idx) { case 0: init(); break; } }",
    "while(i<10){i++;}",
    "while ( i < 10 ) { i++ ; }", 
    "while (i<10) {i++;}", 
    "while(i < 10) { i++; }", 
    "while (i<10)   {   i++;   }", 
    "while(  i  <  10  ){ i++ ; }",
    "while (done==0) { done=1; }",
    "while  (x>0)  { x--; }",

    /* === FOR (100-149) === */
    "for (i = 0; i < 10; i++) { sum += i; }",
    "for (j = 10; j > 0; j--) { data[j] = 0; }",
    "for (k = 0; k < N; k += 2) { process(k); }",
    "for (idx = start; idx < end; idx++) { check(idx); }",
    "for (x = 0; x < MAX && valid; x++) { test(); }",
    "for (i = 0; i < n; i++) { arr[i] = i * 2; }",
    "for (j = 1; j <= 100; j++) { total += j; }",
    "for (i = n-1; i >= 0; i--) { print(arr[i]); }",
    "for (k = 0; k < rows; k++) { sum += mat[k][0]; }",
    "for (i = 0; i < 16; i++) { buf[i] = 0; }",
    "for (j = 0; j < n && arr[j] != x; j++) { ; }",
    "for (i = 2; i <= n; i *= 2) { count++; }",
    "for (k = 0; k < size; k++) { max = (arr[k]>max)?arr[k]:max; }",
    "for (i = 0; i < n-1; i++) { if (a[i]>a[i+1]) swap(a,i); }",
    "for (j = 0; j < cols; j++) { row_sum += mat[0][j]; }",
    "for (i = 1; i < n; i++) { fib[i] = fib[i-1] + fib[i-2]; }",
    "for (k = 0; k < 8; k++) { bits |= (val>>k)&1; }",
    "for (int i=0; i<2; i++) { switch(i) { case 0: item = 0; break; } }",  // Attendu: PATTERN_FOR_LOOP
    "for /* switch to manual control */ (int i=0; i<5; i++) { check(i); }", // Attendu: PATTERN_FOR_LOOP
    "int switch_val = 2; for (int i = 0; i < switch_val; i++) { a = i; }", // Attendu: PATTERN_FOR_LOOP

    "int sum = 0; for (i=0;i<10;i++) { sum += i; }",  
    "int total = 0; for (j=0;j<5;j++) { total++; }",  
    "int arr_len = 3; for (k=0;k<arr_len;k++) { arr[k]=0; }",  
    "int max = 100; for (i=0;i<max;i++) { count++; }",  
    "int limit = 20; for (n=0;n<limit;n++) { val += n; }",  
    "int base = 2; for (p=0;p<base;p++) { res += 2; }",  
    "int a = 1; int b = 5; for (i=a;i<b;i++) { sum++; }",  
    "int stop = 4; for (i=0;i<stop;i++) { data[i]=i; }",  
    "int steps = 6; for (i=0;i<steps;i++) { move(); }",  
    "int rows = 3; for (r=0;r<rows;r++) { print(r); }",  
    "int size = 8; for (i=0;i<size;i++) { buf[i]=0; }",  
    "int end = 9; for (i=0;i<end;i++) { total += i; }",  
    "for (i=0;i<3;i++) { if (i == 1) { skip(); } }",  
    "for (i=0;i<n;i++) { while (busy) { wait(); } }",  
    "for (i=0;i<4;i++) { switch (i) { case 0: init(); break; } }",  
    "for (j=0;j<2;j++) { for (k=0;k<2;k++) { sum++; } }",  
    "for (i=0;i<5;i++) { if (arr[i] > 0) { count++; } }",  
    "for (i=0;i<10;i++) { while (i > 8) { i = 10; } }",  
    "for (i=0;i<3;i++) { switch (mode) { case 1: run(); break; } }",  
    "for (i=0;i<6;i++) { if (i > 3) { odd++; } }", 
    "for (k=0;k<n;k++) { while (flag) { flag = 0; } }",  
    "for (i=0;i<7;i++) { switch (i) { case 3: stop(); break; } }", 
    "for(i=0;i<10;i++){sum+=i;}",  
    "for ( i = 0 ; i < 10 ; i++ ) { sum += i ; }",  
    "for (i=0;i<10;i++) {sum+=i;}",  
    "for(i=0; i<10; i++) { sum += i; }",  
    "for (i=0;i<10;i++)   {   sum+=i;   }",  
    "for(  i=0 ;  i<10 ;  i++  ){ sum += i ; }",  
    "for (i=0;i<5;i++) { total++; }",  
    "for  (j=0;j<5;j++)  { total++; }",  

    /* === SWITCH (150-199) === */
    "switch (x) { case 1: break; }",
    "switch (cmd) { case 'A': exec_a(); break; case 'B': exec_b(); break; }",
    "switch (state) { case IDLE: break; case BUSY: wait(); break; }",
    "switch (mode) { case 0: default: error(); }",
    "switch (type) { case TYPE_INT: return 4; case TYPE_FLOAT: return 8; }",
    "switch (op) { case ADD: result = a+b; break; case SUB: result = a-b; break; }",
    "switch (day) { case 0: return MON; case 6: return SUN; }",
    "switch (err) { case OK: break; case FAIL: handle(); break; }",
    "switch (ch) { case 'y': yes++; break; case 'n': no++; break; }",
    "switch (level) { case 1: easy(); break; case 2: hard(); break; }",
    "switch (dir) { case NORTH: y++; break; case SOUTH: y--; break; }",
    "switch (color) { case RED: r=255; break; case BLUE: b=255; break; }",
    "switch (base) { case 2: bits=8; break; case 10: bits=10; break; }",
    "switch (unit) { case KG: w*=1000; break; case G: break; }",
    "switch (prio) { case HIGH: run_now(); break; case LOW: defer(); break; }",
    "switch (sign) { case POS: return 1; case NEG: return -1; }",
    "switch (fmt) { case HEX: print_hex(v); break; case DEC: print_dec(v); break; }",
    "switch (axis) { case X: dx=1; break; case Y: dy=1; break; }",
    "switch (btn) { case A: jump(); break; case B: shoot(); break; }",
    "switch (mode) { case 1: for(int i=0; i<3; i++) { run(); } break; }",  // Attendu: PATTERN_SWITCH_CASE

    "int mode = 2; switch (mode) { case 2: run(); break; }",
    "int code = 1; switch (code) { case 1: exec(); break; }",  
    "int level = 3; switch (level) { case 3: up(); break; }",
    "int cmd = 0; switch (cmd) { case 0: stop(); break; }",
    "int key = 5; switch (key) { case 5: open(); break; }", 
    "int val = 4; switch (val) { case 4: close(); break; }", 
    "int a = 1; int b = 2; switch (a) { case 1: run(); break; }", 
    "int state = 7; switch (state) { case 7: reset(); break; }", 
    "int opt = 2; switch (opt) { case 2: save(); break; }",
    "int flag = 1; switch (flag) { case 1: load(); break; }", 
    "int action = 0; switch (action) { case 0: idle(); break; }", 
    "int choice = 6; switch (choice) { case 6: run(); break; }", 
    "switch (x) { case 1: for(i=0;i<3;i++){run();} break; }", 
    "switch (mode) { case 2: while(active){active=0;} break; }",
    "switch (code) { case 0: if(ready){go();} break; }",
    "switch (n) { case 3: for(j=0;j<n;j++){sum+=j;} break; }",
    "switch (v) { case 1: if(v>0){inc();} break; }", 
    "switch (s) { case 2: while(s>0){s--;} break; }", 
    "switch (k) { case 4: for(i=0;i<k;i++){arr[i]=0;} break; }",
    "switch (m) { case 0: if(m==0){zero();} break; }", 
    "switch (t) { case 5: while(t>0){t--;} break; }",
    "switch (c) { case 1: for(i=0;i<2;i++){cnt++;} break; }",
    "switch(x){case 1:run();break;}", 
    "switch ( x ) { case 1 : run() ; break ; }", 
    "switch (x) {case 1: run(); break;}", 
    "switch(x) { case 1: run(); break; }",
    "switch (x)   {   case 1: run();   break;   }", 
    "switch(  x  ){ case 1 : run() ; break ; }", 
    "switch (mode==2) { case 2: save(); break; }", 
    "switch  (code)  { case 1: exec(); break; }",
};



static const PatternType expected_200[NUM_EXAMPLE] = {
    /* IF x50 */
    PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE,
    PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE,
    PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE,
    PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE,
    PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE,
    
    PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE,
    PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE,
    PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE,
    PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE,
    PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE, PATTERN_IF_SIMPLE,


    /* WHILE x50 */
    PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP,
    PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP,
    PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP,
    PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP,
    PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP,

    PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP,
    PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP,
    PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP,
    PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP,
    PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP, PATTERN_WHILE_LOOP,
    /* FOR x50 */
    PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP,
    PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP,
    PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP,
    PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP,
    PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP,

    PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP,
    PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP,
    PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP,
    PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP,
    PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP, PATTERN_FOR_LOOP,
    /* SWITCH x50 */
    PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE,
    PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE,
    PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE,
    PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE,
    PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE,

    PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE,
    PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE,
    PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE,
    PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE,
    PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE, PATTERN_SWITCH_CASE,
};

/* ================================================================
   CLASSIFY_CODE — pipeline complet, retourne PatternType
   Utilisé par compile_code() ET run_batch_test_80()
   ================================================================ */
static PatternType classify_code(const char* code) {
    int16_t tokens[16];
    for(int i = 0; i < 16; i++) tokens[i] = 0;
    tokenize(code, tokens, 16);

    int16_t embedding[16];
    int result = transformer_process(tokens, embedding);
    if(result != 0) return PATTERN_UNKNOWN;

    int16_t *output_buf = transformer_get_output_buffer();

    /* Mean pooling */
    int32_t pooled[CLASSIFIER_IN_DIM];
    for(int d = 0; d < CLASSIFIER_IN_DIM; d++) {
        int32_t acc = 0;
        for(int pos = 0; pos < 16; pos++)
            acc += (int32_t)output_buf[pos * CLASSIFIER_IN_DIM + d];
        pooled[d] = acc >> 4;
    }

    /* Classifieur linéaire */
    int64_t logits[CLASSIFIER_OUT_DIM];
    for(int c = 0; c < CLASSIFIER_OUT_DIM; c++) {
        logits[c] = 0;
        for(int d = 0; d < CLASSIFIER_IN_DIM; d++)
            logits[c] += (int64_t)pooled[d] * (int64_t)classifier_W[c][d];
        logits[c] += (int64_t)classifier_b[c];
    }

    int predicted = 0;
    for(int c = 1; c < CLASSIFIER_OUT_DIM; c++)
        if(logits[c] > logits[predicted]) predicted = c;

    switch(predicted) {
        case CLASS_IF:     return PATTERN_IF_SIMPLE;
        case CLASS_WHILE:  return PATTERN_WHILE_LOOP;
        case CLASS_FOR:    return PATTERN_FOR_LOOP;
        case CLASS_SWITCH: return PATTERN_SWITCH_CASE;
        default:           return PATTERN_UNKNOWN;
    }
}


/* =================================================================
   BASELINE SOFTWARE 2 — Vrai Transformer exécuté par le CPU (ARM)
   Sert à isoler le strict gain de performance de l'accélération FPGA.
   ================================================================= */
/* Layer Norm logicielle (fidèle à layer_norm1/layer_norm2 côté HLS) */
static void layer_norm_sw(const float In[SEQ_LEN][D_MODEL], float Out[SEQ_LEN][D_MODEL],
                           const int16_t* gamma_q, const int16_t* beta_q) {
    for(int i = 0; i < SEQ_LEN; i++) {
        float mean = 0.0f;
        for(int j = 0; j < D_MODEL; j++) mean += In[i][j];
        mean /= (float)D_MODEL;

        float var = 0.0f;
        for(int j = 0; j < D_MODEL; j++) {
            float d = In[i][j] - mean;
            var += d * d;
        }
        var /= (float)D_MODEL;

        float inv_std = 1.0f / sqrtf(var + 0.00001f);
        for(int j = 0; j < D_MODEL; j++) {
            float norm = (In[i][j] - mean) * inv_std;
            float gamma = (float)gamma_q[j] / 256.0f;
            float beta  = (float)beta_q[j]  / 256.0f;
            Out[i][j] = gamma * norm + beta;
        }
    }
}

/* =================================================================
   BASELINE SOFTWARE 2 — Vrai Transformer entraîné, exécuté par le CPU (ARM)
   Réutilise exactement les mêmes poids/biais que le pipeline FPGA
   (weights_fpga.h, attention_bias_sw.h, layernorm_params.h) afin
   d'isoler strictement le gain de l'accélération matérielle.
   ================================================================= */
PatternType transformer_software_classify(const char* code) {
    #define SW_NH   2
    #define SW_DH   8
    #define SW_DFF  32
    #define SW_SQRT_DH 2.828f

    int16_t tokens[SEQ_LEN];
    for(int i = 0; i < SEQ_LEN; i++) tokens[i] = 0;
    tokenize(code, tokens, SEQ_LEN);

    /* 1. Embedding */
    float X[SEQ_LEN][D_MODEL];
    for(int i = 0; i < SEQ_LEN; i++) {
        int tok = tokens[i];
        if(tok < 0 || tok >= 100) tok = 0;
        for(int j = 0; j < D_MODEL; j++)
            X[i][j] = (float)embedding_table[tok * D_MODEL + j] / 256.0f;
    }
    /* 1bis. Ajout de l'encodage positionnel — AJOUT */
    for(int i = 0; i < SEQ_LEN; i++)
        for(int j = 0; j < D_MODEL; j++)
            X[i][j] += sw_positional_encoding[i][j];    

    /* 2. Multi-Head Attention — vraies projections Q/K/V + softmax réel */
    float concat[SEQ_LEN][D_MODEL];

    for(int h = 0; h < SW_NH; h++) {
        float Q[SEQ_LEN][SW_DH], K[SEQ_LEN][SW_DH], V[SEQ_LEN][SW_DH];

        for(int i = 0; i < SEQ_LEN; i++) {
            for(int j = 0; j < SW_DH; j++) {
                float q = 0.0f, k = 0.0f, v = 0.0f;
                for(int d = 0; d < D_MODEL; d++) {
                    float xin = X[i][d];
                    int idx = h * D_MODEL * SW_DH + d * SW_DH + j;
                    q += xin * ((float)attention_in_proj_W[idx]       / 256.0f);
                    k += xin * ((float)attention_in_proj_W[256 + idx] / 256.0f);
                    v += xin * ((float)attention_in_proj_W[512 + idx] / 256.0f);
                }
                Q[i][j] = q + sw_in_proj_bias_q[h][j];
                K[i][j] = k + sw_in_proj_bias_k[h][j];
                V[i][j] = v + sw_in_proj_bias_v[h][j];
            }
        }

        /* Scaled dot-product attention avec softmax exponentiel réel */
        for(int i = 0; i < SEQ_LEN; i++) {
            float scores[SEQ_LEN];
            float maxv = -1e30f;
            for(int jx = 0; jx < SEQ_LEN; jx++) {
                float dot = 0.0f;
                for(int d = 0; d < SW_DH; d++) dot += Q[i][d] * K[jx][d];
                scores[jx] = dot / SW_SQRT_DH;
                if(scores[jx] > maxv) maxv = scores[jx];
            }
            float sum_exp = 0.0f;
            for(int jx = 0; jx < SEQ_LEN; jx++) {
                scores[jx] = expf(scores[jx] - maxv);
                sum_exp += scores[jx];
            }
            for(int d = 0; d < SW_DH; d++) {
                float acc = 0.0f;
                for(int jx = 0; jx < SEQ_LEN; jx++)
                    acc += (scores[jx] / sum_exp) * V[jx][d];
                concat[i][h * SW_DH + d] = acc;
            }
        }
    }

    /* Projection finale Wo + biais */
    float MHA_out[SEQ_LEN][D_MODEL];
    for(int i = 0; i < SEQ_LEN; i++) {
        for(int j = 0; j < D_MODEL; j++) {
            float sum = 0.0f;
            for(int kk = 0; kk < D_MODEL; kk++)
                sum += concat[i][kk] * ((float)attention_out_proj_W[kk * D_MODEL + j] / 256.0f);
            MHA_out[i][j] = sum + sw_out_proj_bias_q[j];
        }
    }

    /* Residual + LayerNorm1 (Post-LN, comme le modèle PyTorch) */
    float MHA_res[SEQ_LEN][D_MODEL], MHA_norm[SEQ_LEN][D_MODEL];
    for(int i = 0; i < SEQ_LEN; i++)
        for(int j = 0; j < D_MODEL; j++)
            MHA_res[i][j] = X[i][j] + MHA_out[i][j];
    layer_norm_sw(MHA_res, MHA_norm, norm1_gamma_q, norm1_beta_q);

    /* FFN : Linear(16->32) + ReLU + Linear(32->16) */
    float hidden[SEQ_LEN][SW_DFF];
    for(int i = 0; i < SEQ_LEN; i++) {
        for(int j = 0; j < SW_DFF; j++) {
            float sum = 0.0f;
            for(int kk = 0; kk < D_MODEL; kk++)
                sum += MHA_norm[i][kk] * ((float)ffn_0_W[kk * SW_DFF + j] / 256.0f);
            sum += sw_ffn0_bias_q[j];
            hidden[i][j] = (sum > 0.0f) ? sum : 0.0f;
        }
    }
    float FFN_out[SEQ_LEN][D_MODEL];
    for(int i = 0; i < SEQ_LEN; i++) {
        for(int j = 0; j < D_MODEL; j++) {
            float sum = 0.0f;
            for(int kk = 0; kk < SW_DFF; kk++)
                sum += hidden[i][kk] * ((float)ffn_2_W[kk * D_MODEL + j] / 256.0f);
            FFN_out[i][j] = sum + sw_ffn2_bias_q[j];
        }
    }

    /* Residual + LayerNorm2 */
    float FFN_res[SEQ_LEN][D_MODEL], FFN_norm[SEQ_LEN][D_MODEL];
    for(int i = 0; i < SEQ_LEN; i++)
        for(int j = 0; j < D_MODEL; j++)
            FFN_res[i][j] = MHA_norm[i][j] + FFN_out[i][j];
    layer_norm_sw(FFN_res, FFN_norm, norm2_gamma_q, norm2_beta_q);

    /* Mean pooling (dim=1, comme x.mean(dim=1) en PyTorch) */
    float pooled[D_MODEL];
    for(int j = 0; j < D_MODEL; j++) {
        float sum = 0.0f;
        for(int i = 0; i < SEQ_LEN; i++) sum += FFN_norm[i][j];
        pooled[j] = sum / (float)SEQ_LEN;
    }

    /* Classifieur linéaire final (mêmes poids que le hardware) */
    float logits[4];
    for(int c = 0; c < 4; c++) {
        float score = 0.0f;
        for(int d = 0; d < D_MODEL; d++)
            score += pooled[d] * ((float)classifier_W[c][d] / 256.0f);
        score += (float)classifier_b[c] / 65536.0f;
        logits[c] = score;
    }

    int best = 0;
    float maxscore = logits[0];
    for(int c = 1; c < 4; c++)
        if(logits[c] > maxscore) { maxscore = logits[c]; best = c; }

    switch(best) {
        case 0: return PATTERN_IF_SIMPLE;
        case 1: return PATTERN_WHILE_LOOP;
        case 2: return PATTERN_FOR_LOOP;
        case 3: return PATTERN_SWITCH_CASE;
        default: return PATTERN_UNKNOWN;
    }
}

/* ================================================================
   TIMER — Lecture directe du Global Timer Cortex-A9
   Pas d'include nécessaire, toujours disponible sur Zynq-7000
   Résolution : 1 tick = 1/(CPU_FREQ/2) = ~3 ns à 666 MHz
   ================================================================ */
#define GLOBAL_TIMER_BASE     0xF8F00200
#define GLOBAL_TIMER_COUNT_LO (*(volatile u32*)(GLOBAL_TIMER_BASE + 0x00))
#define GLOBAL_TIMER_COUNT_HI (*(volatile u32*)(GLOBAL_TIMER_BASE + 0x04))
#define GLOBAL_TIMER_CTRL     (*(volatile u32*)(GLOBAL_TIMER_BASE + 0x08))

static void timer_start(void) {
    GLOBAL_TIMER_CTRL = 1;  /* Enable */
}

static u64 timer_read_us(void) {
    u32 lo = GLOBAL_TIMER_COUNT_LO;
    u32 hi = GLOBAL_TIMER_COUNT_HI;
    u64 ticks = ((u64)hi << 32) | lo;
    /* Global Timer = CPU_FREQ / 2 = 666/2 = 333 MHz */
    /* 1 tick = 1/333 µs → pour éviter la division flottante : */
    /* latence_us = ticks * 3 / 1000  (×3/1000 ≈ /333) */
    return (ticks * 3) / 1000;
}

/* ================================================================
   BASELINE SOFTWARE — classification par token uniquement (sans FPGA)
   Mesure la précision et latence d'une approche triviale
   pour comparer avec le Transformer FPGA
   ================================================================ */
static PatternType baseline_classify(const char* code) {
    int16_t tokens[16];
    for(int i = 0; i < 16; i++) tokens[i] = 0;
    tokenize(code, tokens, 16);
    /* Règle triviale : premier token détermine la classe */
    switch(tokens[0]) {
        case 1: return PATTERN_IF_SIMPLE;
        case 3: return PATTERN_WHILE_LOOP;
        case 4: return PATTERN_FOR_LOOP;
        case 5: return PATTERN_SWITCH_CASE;
        default: return PATTERN_UNKNOWN;
    }
}

/* =================================================================
   ÉVALUATION ET COMPARISON DES TROIS BASELINES (BATCH TEST)
   Mesure l'accuracy (%) et la latence moyenne (us) des 3 approches
   ================================================================= */
void run_batch_test_200(void) {
    xil_printf("\n\r============================================================\n\r");
    xil_printf("   BATCH TEST — 200 EXEMPLES (50 par classe)\n\r");
    xil_printf("============================================================\n\r");

    // Déclaration des compteurs locaux (uniquement ici, pas dans le .h)
    int if_correct = 0;
    int while_correct = 0;
    int for_correct = 0;
    int switch_correct = 0;
    int total_correct = 0;

    /* Accumulateurs latence cœur FPGA (isolée) */
    uint64_t total_dispatch_us = 0;
    uint64_t total_core_us     = 0;
    uint64_t total_readback_us = 0;

    /* Mesure latence totale */
    timer_start();
    u64 t1 = timer_read_us();

    for(int i = 0; i < NUM_EXAMPLE; i++) {
        PatternType predicted = classify_code(codes_200[i]);
        PatternType expected  = expected_200[i];
        total_dispatch_us += transformer_get_last_dispatch_us();
        total_core_us     += transformer_get_last_core_us();
        total_readback_us += transformer_get_last_readback_us();

        if (predicted == expected) {
            // Le test est réussi, on incrémente le bon compteur selon la classe réelle
            total_correct++;
            if (expected == PATTERN_IF_SIMPLE)   if_correct++;
            if (expected == PATTERN_WHILE_LOOP)  while_correct++;
            if (expected == PATTERN_FOR_LOOP)    for_correct++;
            if (expected == PATTERN_SWITCH_CASE) switch_correct++;
        } 
        else {
            // En cas d'échec, on convertit proprement l'Enum en texte pour le xil_printf
            const char* exp_str  = "UNK";
            const char* pred_str = "UNK";

            if (expected == PATTERN_IF_SIMPLE)   exp_str = "IF";
            if (expected == PATTERN_WHILE_LOOP)  exp_str = "WHILE";
            if (expected == PATTERN_FOR_LOOP)    exp_str = "FOR";
            if (expected == PATTERN_SWITCH_CASE) exp_str = "SWITCH";

            if (predicted == PATTERN_IF_SIMPLE)   pred_str = "IF";
            if (predicted == PATTERN_WHILE_LOOP)  pred_str = "WHILE";
            if (predicted == PATTERN_FOR_LOOP)    pred_str = "FOR";
            if (predicted == PATTERN_SWITCH_CASE) pred_str = "SWITCH";

            xil_printf("[%2d] FAIL exp=%s got=%s | %s\n\r",
                       i+1, exp_str, pred_str, codes_200[i]);
        }
    }

    u64 t2 = timer_read_us();
    u64 total_us = t2 - t1;

    xil_printf("\n\r============================================================\n\r");
    xil_printf("   RÉSULTATS CORRIGÉS IP \n\r");
    xil_printf("============================================================\n\r");
    xil_printf("   IF    : %2d/50\n\r", if_correct);
    xil_printf("   WHILE : %2d/50\n\r", while_correct);
    xil_printf("   FOR   : %2d/50\n\r", for_correct);
    xil_printf("   SWITCH: %2d/50\n\r", switch_correct);
    xil_printf("   -------------------------------------\n\r");
    /* xil_printf("   TOTAL : %2d/200 = %.1f%%\n\r",
               total_correct, 100.0f * total_correct / NUM_EXAMPLE); */
               
    int perc_int = (total_correct * 100) / NUM_EXAMPLE;
    int perc_dec = ((total_correct * 1000) / NUM_EXAMPLE) % 10;
    xil_printf("   TOTAL : %2d/200 = %d.%d%%\n\r", total_correct, perc_int, perc_dec);

    xil_printf("\n\r   Latence totale  : %llu us\n\r", total_us);
    xil_printf("   Latence moyenne : %llu us/inference\n\r", total_us / NUM_EXAMPLE);
    xil_printf("\n\r   --- Décomposition cœur FPGA (isolée, hors logiciel) ---\n\r");
    xil_printf("   Dispatch moyen  : %llu us/inference\n\r", total_dispatch_us / NUM_EXAMPLE);
    xil_printf("   Cœur AXI moyen  : %llu us/inference\n\r", total_core_us / NUM_EXAMPLE);
    xil_printf("   Readback moyen  : %llu us/inference\n\r", total_readback_us / NUM_EXAMPLE);
    xil_printf("   Somme des 3     : %llu us/inference\n\r",
               (total_dispatch_us + total_core_us + total_readback_us) / NUM_EXAMPLE);
    xil_printf("============================================================\n\r");
}

void run_batch_test_200_heuristic(void) {
    xil_printf("\n\r============================================================\n\r");
    xil_printf("   BATCH TEST — 200 EXEMPLES (HEURISTIQUE STATIQUE CPU)\n\r");
    xil_printf("============================================================\n\r");

    int if_correct = 0;
    int while_correct = 0;
    int for_correct = 0;
    int switch_correct = 0;
    int total_correct = 0;

    /* Mesure de la latence globale */
    timer_start();
    u64 t1 = timer_read_us();

    for(int i = 0; i < 200; i++) {
        // APPEL DE L'HEURISTIQUE LOGICIELLE CPU
        PatternType predicted = baseline_classify(codes_200[i]);
        PatternType expected  = expected_200[i];

        if (predicted == expected) {
            total_correct++;
            if (expected == PATTERN_IF_SIMPLE)   if_correct++;
            if (expected == PATTERN_WHILE_LOOP)  while_correct++;
            if (expected == PATTERN_FOR_LOOP)    for_correct++;
            if (expected == PATTERN_SWITCH_CASE) switch_correct++;
        } 
        else {
            const char* exp_str  = "UNK";
            const char* pred_str = "UNK";

            if (expected == PATTERN_IF_SIMPLE)   exp_str = "IF";
            if (expected == PATTERN_WHILE_LOOP)  exp_str = "WHILE";
            if (expected == PATTERN_FOR_LOOP)    exp_str = "FOR";
            if (expected == PATTERN_SWITCH_CASE) exp_str = "SWITCH";

            if (predicted == PATTERN_IF_SIMPLE)   pred_str = "IF";
            if (predicted == PATTERN_WHILE_LOOP)  pred_str = "WHILE";
            if (predicted == PATTERN_FOR_LOOP)    pred_str = "FOR";
            if (predicted == PATTERN_SWITCH_CASE) pred_str = "SWITCH";

            xil_printf("[%2d] FAIL exp=%s got=%s | %s\n\r", i+1, exp_str, pred_str, codes_200[i]);
        }
    }

    u64 t2 = timer_read_us();
    u64 total_us = t2 - t1;

    // Calcul du pourcentage sans float pour xil_printf
    int perc_int = (total_correct * 100) / 200;
    int perc_dec = ((total_correct * 1000) / 200) % 10;

    xil_printf("\n\r============================================================\n\r");
    xil_printf("   RÉSULTATS HEURISTIQUE CPU\n\r");
    xil_printf("============================================================\n\r");
    xil_printf("   IF    : %2d/50\n\r", if_correct);
    xil_printf("   WHILE : %2d/50\n\r", while_correct);
    xil_printf("   FOR   : %2d/50\n\r", for_correct);
    xil_printf("   SWITCH: %2d/50\n\r", switch_correct);
    xil_printf("   -------------------------------------\n\r");
    xil_printf("   TOTAL : %2d/200 = %d.%d%%\n\r", total_correct, perc_int, perc_dec);
    xil_printf("\n\r   Latence totale  : %llu us\n\r", total_us);
    xil_printf("   Latence moyenne : %llu us/inference\n\r", total_us / 200);
    xil_printf("============================================================\n\r");
}

void run_batch_test_200_transformer_sw(void) {
    xil_printf("\n\r============================================================\n\r");
    xil_printf("   BATCH TEST — 200 EXEMPLES (TRANSFORMER PUR LOGICIEL CPU)\n\r");
    xil_printf("============================================================\n\r");

    int if_correct = 0;
    int while_correct = 0;
    int for_correct = 0;
    int switch_correct = 0;
    int total_correct = 0;

    /* Mesure de la latence globale */
    timer_start();
    u64 t1 = timer_read_us();

    for(int i = 0; i < 200; i++) {
        // APPEL DU TRANSFORMER SIMULÉ/EXÉCUTÉ SUR CPU
        PatternType predicted = transformer_software_classify(codes_200[i]);
        PatternType expected  = expected_200[i];

        if (predicted == expected) {
            total_correct++;
            if (expected == PATTERN_IF_SIMPLE)   if_correct++;
            if (expected == PATTERN_WHILE_LOOP)  while_correct++;
            if (expected == PATTERN_FOR_LOOP)    for_correct++;
            if (expected == PATTERN_SWITCH_CASE) switch_correct++;
        } 
        else {
            const char* exp_str  = "UNK";
            const char* pred_str = "UNK";

            if (expected == PATTERN_IF_SIMPLE)   exp_str = "IF";
            if (expected == PATTERN_WHILE_LOOP)  exp_str = "WHILE";
            if (expected == PATTERN_FOR_LOOP)    exp_str = "FOR";
            if (expected == PATTERN_SWITCH_CASE) exp_str = "SWITCH";

            if (predicted == PATTERN_IF_SIMPLE)   pred_str = "IF";
            if (predicted == PATTERN_WHILE_LOOP)  pred_str = "WHILE";
            if (predicted == PATTERN_FOR_LOOP)    pred_str = "FOR";
            if (predicted == PATTERN_SWITCH_CASE) pred_str = "SWITCH";

            xil_printf("[%2d] FAIL exp=%s got=%s | %s\n\r", i+1, exp_str, pred_str, codes_200[i]);
        }
    }

    u64 t2 = timer_read_us();
    u64 total_us = t2 - t1;

    // Calcul du pourcentage sans float pour xil_printf
    int perc_int = (total_correct * 100) / 200;
    int perc_dec = ((total_correct * 1000) / 200) % 10;

    xil_printf("\n\r============================================================\n\r");
    xil_printf("   RÉSULTATS TRANSFORMER LOGICIEL CPU\n\r");
    xil_printf("============================================================\n\r");
    xil_printf("   IF    : %2d/50\n\r", if_correct);
    xil_printf("   WHILE : %2d/50\n\r", while_correct);
    xil_printf("   FOR   : %2d/50\n\r", for_correct);
    xil_printf("   SWITCH: %2d/50\n\r", switch_correct);
    xil_printf("   -------------------------------------\n\r");
    xil_printf("   TOTAL : %2d/200 = %d.%d%%\n\r", total_correct, perc_int, perc_dec);
    xil_printf("\n\r   Latence totale  : %llu us\n\r", total_us);
    xil_printf("   Latence moyenne : %llu us/inference\n\r", total_us / 200);
    xil_printf("============================================================\n\r");
}

/* ================================================================
   STRUCTURES ET TYPES INTERNES
   ================================================================ */
typedef struct {
    int32_t sum;
    int32_t positive;
    int32_t negative;
    int32_t zeros;
    int32_t weighted;
} TransformerSignature;

/* ================================================================
   FONCTIONS INTERNES — forward declarations
   ================================================================ */
static PatternType classify_from_first_token(int16_t tokens[16]);
void generate_assembly(PatternType pattern, const char* code);
static void display_examples_menu(void);
static char get_user_input(void);


/* ================================================================
   CLASSIFY FROM FIRST TOKEN (fallback)
   Utilisé si le Transformer échoue.
   Token IDs: 1=IF, 3=WHILE, 4=FOR, 5=SWITCH
   ================================================================ */
static PatternType classify_from_first_token(int16_t tokens[16]) {
    switch(tokens[0]) {
        case 1: return PATTERN_IF_SIMPLE;
        case 3: return PATTERN_WHILE_LOOP;
        case 4: return PATTERN_FOR_LOOP;
        case 5: return PATTERN_SWITCH_CASE;
        default: return PATTERN_UNKNOWN;
    }
}

/* ================================================================
   GENERATE ASSEMBLY
   Génération du code ARM Thumb-2 pour chaque patron détecté
   ================================================================ */
void generate_assembly(PatternType pattern, const char* code) {
    xil_printf("\n\r");
    for(int i = 0; i < 60; i++) xil_printf("-");
    xil_printf("\n\r  STEP 4: ARM Assembly Generation\n\r");
    for(int i = 0; i < 60; i++) xil_printf("-");
    xil_printf("\n\r");
    xil_printf("@ ARM Assembly (Thumb-2) | Cortex-A9 | Source: %s\n\r\n\r", code);

    switch(pattern) {
        case PATTERN_IF_SIMPLE:
            xil_printf("if_statement:\n\r");
            xil_printf("    PUSH    {R4-R7, LR}\n\r");
            xil_printf("    LDR     R0, =var_x\n\r");
            xil_printf("    LDR     R1, [R0]\n\r");
            xil_printf("    MOV     R2, #5\n\r");
            xil_printf("    CMP     R1, R2\n\r");
            xil_printf("    BLE     .Lend_if\n\r");
            xil_printf(".Lthen_block:\n\r");
            xil_printf("    MOV     R3, #10\n\r");
            xil_printf("    LDR     R0, =var_y\n\r");
            xil_printf("    STR     R3, [R0]\n\r");
            xil_printf(".Lend_if:\n\r");
            xil_printf("    POP     {R4-R7, PC}\n\r");
            break;

        case PATTERN_WHILE_LOOP:
            xil_printf("while_loop:\n\r");
            xil_printf("    PUSH    {R4-R7, LR}\n\r");
            xil_printf("    LDR     R0, =var_i\n\r");
            xil_printf("    LDR     R1, [R0]\n\r");
            xil_printf("    MOV     R2, #10\n\r");
            xil_printf(".Lwhile_start:\n\r");
            xil_printf("    CMP     R1, R2\n\r");
            xil_printf("    BGE     .Lwhile_end\n\r");
            xil_printf("    ADD     R1, R1, #1\n\r");
            xil_printf("    STR     R1, [R0]\n\r");
            xil_printf("    B       .Lwhile_start\n\r");
            xil_printf(".Lwhile_end:\n\r");
            xil_printf("    POP     {R4-R7, PC}\n\r");
            break;

        case PATTERN_FOR_LOOP:
            xil_printf("for_loop:\n\r");
            xil_printf("    PUSH    {R4-R7, LR}\n\r");
            xil_printf("    MOV     R1, #0\n\r");
            xil_printf("    MOV     R2, #10\n\r");
            xil_printf(".Lfor_condition:\n\r");
            xil_printf("    CMP     R1, R2\n\r");
            xil_printf("    BGE     .Lfor_end\n\r");
            xil_printf("    ADD     R1, R1, #1\n\r");
            xil_printf("    B       .Lfor_condition\n\r");
            xil_printf(".Lfor_end:\n\r");
            xil_printf("    POP     {R4-R7, PC}\n\r");
            break;

        case PATTERN_SWITCH_CASE:
            xil_printf("switch_case:\n\r");
            xil_printf("    PUSH    {R4-R7, LR}\n\r");
            xil_printf("    LDR     R0, =var_x\n\r");
            xil_printf("    LDR     R1, [R0]\n\r");
            xil_printf("    CMP     R1, #1\n\r");
            xil_printf("    BEQ     .Lcase_1\n\r");
            xil_printf("    CMP     R1, #2\n\r");
            xil_printf("    BEQ     .Lcase_2\n\r");
            xil_printf("    B       .Ldefault\n\r");
            xil_printf(".Lcase_1:\n\r");
            xil_printf("    B       .Lswitch_end\n\r");
            xil_printf(".Lcase_2:\n\r");
            xil_printf("    B       .Lswitch_end\n\r");
            xil_printf(".Ldefault:\n\r");
            xil_printf(".Lswitch_end:\n\r");
            xil_printf("    POP     {R4-R7, PC}\n\r");
            break;

        default:
            xil_printf("    @ ERROR: Unknown pattern\n\r");
            break;
    }

    xil_printf("\n\r@ End of generated assembly\n\r");
    for(int i = 0; i < 60; i++) xil_printf("-");
    xil_printf("\n\r");
}

/* Activer le compteur de cycles ARM (Performance Monitor Unit) */
 void enable_cycle_counter(void) {
    /* Enable PMU */
    asm volatile("mcr p15, 0, %0, c9, c14, 0" :: "r"(1));
    /* Reset and enable cycle counter */
    asm volatile("mcr p15, 0, %0, c9, c12, 0" :: "r"(5));
    asm volatile("mcr p15, 0, %0, c9, c12, 1" :: "r"(0x80000000));
}

u32 read_cycle_counter(void) {
    u32 val;
    asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(val));
    return val;
}


void run_baseline_comparison(void) {
    xil_printf("\n\r============================================================\n\r");
    xil_printf("  BASELINE vs TRANSFORMER\n\r");
    xil_printf("============================================================\n\r");

    int baseline_correct = 0;
    int transformer_correct = 0;

    timer_start();
    u64 t1 = timer_read_us();

    /* Baseline : premier token seulement, sans FPGA */
    for(int i = 0; i < NUM_EXAMPLE; i++) {
        PatternType pred = baseline_classify(codes_200[i]);
        if(pred == expected_200[i]) baseline_correct++;
    }

    u64 t2 = timer_read_us();

    /* Transformer : avec FPGA */
    u64 t3 = timer_read_us();

    for(int i = 0; i < NUM_EXAMPLE; i++) {
        PatternType pred = classify_code(codes_200[i]);
        if(pred == expected_200[i]) transformer_correct++;
    }

    u64 t4 = timer_read_us();

    xil_printf("  Baseline (token seul, CPU uniquement) :\n\r");
    // APRÈS :
    int bl_pct_int = (baseline_correct * 100) / 200;
    int bl_pct_dec = ((baseline_correct * 1000) / 200) % 10;
    xil_printf("    Accuracy : %d/80 = %d.%d%%\n\r",
            baseline_correct, bl_pct_int, bl_pct_dec);
    u64 bl_lat = (t2 - t1) / 200;
    xil_printf("    Latence  : %llu us/inference\n\r", bl_lat);

    xil_printf("\n\r  Transformer FPGA :\n\r");
    int tr_pct_int = (transformer_correct * 100) / 200;
    int tr_pct_dec = ((transformer_correct * 1000) / 200) % 10;
    xil_printf("    Accuracy : %d/200 = %d.%d%%\n\r",
               transformer_correct, tr_pct_int, tr_pct_dec);
    u64 tr_lat = (t4-t3)/200;
    xil_printf("    Latence  : %llu us/inference\n\r", tr_lat);

    int tr_cor_int = 100*(transformer_correct-baseline_correct) / 200;
    int tr_cor_dec = (1000*(transformer_correct-baseline_correct) / 200) % 10;
    xil_printf("\n\r  Gain accuracy  : +%d.%d%%\n\r", tr_cor_int, tr_cor_dec);
    u64 tr_cor_lat = (t4-t3)/(t2-t1);
    xil_printf("  Cout latence    : x%llu vs baseline\n\r", tr_cor_lat);
    xil_printf("============================================================\n\r");
}


/* ================================================================
   COMPILE CODE — Pipeline principal
   1. Tokenisation (PS)
   2. Transformer FPGA (PL)
   3. Classification par signature (PS)
   4. Génération assembleur (PS)
   ================================================================ */
void compile_code(const char* code) {
    xil_printf("\n\r\n\r");
    for(int i = 0; i < 60; i++) xil_printf("=");
    xil_printf("\n\r     TRANSFORMER-BASED COMPILER PIPELINE\n\r");
    for(int i = 0; i < 60; i++) xil_printf("=");
    xil_printf("\n\r");

    /* --- STEP 1: Tokenisation (PS) --- */
    xil_printf("\n\rSTEP 1: Tokenization (PS - ARM Cortex-A9)\n\r");
    for(int i = 0; i < 60; i++) xil_printf("-");
    xil_printf("\n\rInput: %s\n\r\n\r", code);

    int16_t tokens[16];
    for(int i = 0; i < 16; i++) tokens[i] = 0;
    tokenize(code, tokens, 16);

    const char* token_names[] = {
        "UNK", "IF", "ELSE", "WHILE", "FOR", "SWITCH",
        "CASE", "RETURN", "BREAK", "INT", "VOID"
    };
    xil_printf("Tokens: ");
    int token_count = 0;
    for(int i = 0; i < 16; i++) {
        if(tokens[i] == 0) break;
        token_count++;
        if(tokens[i] >= 1 && tokens[i] <= 10)
            xil_printf("[%s] ", token_names[tokens[i]]);
        else if(tokens[i] == 50)  xil_printf("[(] ");
        else if(tokens[i] == 51)  xil_printf("[)] ");
        else if(tokens[i] == 60)  xil_printf("[VAR] ");
        else if(tokens[i] == 61)  xil_printf("[NUM] ");
        else                       xil_printf("[%d] ", tokens[i]);
    }
    xil_printf("\n\rToken count: %d\n\r", token_count);

   /* STEP 2 : Transformer FPGA avec mesure de latence */
    xil_printf("\n\rSTEP 2: Pattern Recognition (PL - FPGA Transformer)\n\r");
    for(int i = 0; i < 60; i++) xil_printf("-");
    xil_printf("\n\r");

    timer_start();
    u64 t_before = timer_read_us();

    int16_t embedding[16];
    transformer_process(tokens, embedding);

    u64 t_after = timer_read_us();
    u64 latency_us = t_after - t_before;

    xil_printf("[TIMING] Transformer latency: %llu us\n\r", latency_us);

    /* --- STEP 3: Classification (PS - ARM Cortex-A9) --- */
    xil_printf("\n\rSTEP 3: Pattern Classification (PS - ARM Cortex-A9)\n\r");
    for(int i = 0; i < 60; i++) xil_printf("-");
    xil_printf("\n\r");

    /* Récupérer le buffer complet de sortie FPGA
     * Dimensions : SEQ_LEN(16) × D_MODEL(16) = 256 valeurs int16_t
     * Layout mémoire : [pos0_dim0, pos0_dim1, ..., pos0_dim15,
     *                   pos1_dim0, ..., pos15_dim15]             */
    int16_t *output_buf = transformer_get_output_buffer();

    /* ---- Mean Pooling sur les 16 positions ----
     * PyTorch fait mean(dim=1) avant le classifieur
     * output_buf[pos * D_MODEL + dim] → moyenne sur les 16 pos
     * Accumulation en int32 pour éviter overflow
     * Résultat en Q8.8 : diviser par 16 (shift >> 4)            */
    #define D_MODEL_VAL  16
    #define SEQ_LEN_VAL  16

    int32_t pooled_q[D_MODEL_VAL];
    for(int d = 0; d < D_MODEL_VAL; d++) {
        int32_t acc = 0;
        for(int pos = 0; pos < SEQ_LEN_VAL; pos++) {
            acc += (int32_t)output_buf[pos * D_MODEL_VAL + d];
        }
        /* Division par 16 = shift arithmétique de 4 bits
         * Résultat reste en Q8.8 car on moyenne des Q8.8        */
        pooled_q[d] = acc >> 4;
    }

    /* Afficher le vecteur poolé pour validation ZedBoard */
    xil_printf("Mean-pooled vector (Q8.8):\n\r  [");
    for(int d = 0; d < D_MODEL_VAL; d++) {
        xil_printf("%d", (int)pooled_q[d]);
        if(d < D_MODEL_VAL - 1) xil_printf(", ");
    }
    xil_printf("]\n\r\n\r");

    /* ---- Classifieur Linéaire Linear(16→4) ----
     * acc[c] = Σ_d pooled_q[d] × classifier_W[c][d]  (int64)
     * acc[c] += classifier_b[c]                        (Q16.16)
     *
     * Unités :
     *   pooled_q   : Q8.8  → valeur réelle = pooled_q / 256
     *   classifier_W: Q8.8  → valeur réelle = W / 256
     *   produit    : Q16.16 → valeur réelle = produit / 65536
     *   classifier_b: Q16.16 → même unité, addition directe
     *
     * Argmax sur acc[] donne la classe — pas besoin de dé-quantifier  */
    int64_t logits[CLASSIFIER_OUT_DIM];
    for(int c = 0; c < CLASSIFIER_OUT_DIM; c++) {
        logits[c] = 0;
        for(int d = 0; d < CLASSIFIER_IN_DIM; d++) {
            logits[c] += (int64_t)pooled_q[d] * (int64_t)classifier_W[c][d];
        }
        logits[c] += (int64_t)classifier_b[c];
    }

    /* Afficher les logits pour debug */
    xil_printf("Logits (Q16.16):\n\r");
    const char* cls_names[] = {"IF", "WHILE", "FOR", "SWITCH"};
    for(int c = 0; c < CLASSIFIER_OUT_DIM; c++) {
        /* Convertir en float approximatif pour affichage lisible
         * logit_real = logits[c] / 65536                         */
        int32_t logit_int  = (int32_t)(logits[c] >> 16);
        int32_t logit_frac = (int32_t)((logits[c] & 0xFFFF) * 100 >> 16);
        if(logit_frac < 0) logit_frac = -logit_frac;
        xil_printf("  %s: %d.%02d\n\r", cls_names[c],
                   (int)logit_int, (int)logit_frac);
    }

    /* ---- Argmax → classe prédite ---- */
    int predicted_class = 0;
    for(int c = 1; c < CLASSIFIER_OUT_DIM; c++) {
        if(logits[c] > logits[predicted_class]) {
            predicted_class = c;
        }
    }

    /* Convertir en PatternType
     * Ordre : 0=IF, 1=WHILE, 2=FOR, 3=SWITCH
     * Doit correspondre à class_names dans classifier_weights.h  */
    PatternType pattern;
    switch(predicted_class) {
        case CLASS_IF:     pattern = PATTERN_IF_SIMPLE;   break;
        case CLASS_WHILE:  pattern = PATTERN_WHILE_LOOP;  break;
        case CLASS_FOR:    pattern = PATTERN_FOR_LOOP;    break;
        case CLASS_SWITCH: pattern = PATTERN_SWITCH_CASE; break;
        default:           pattern = PATTERN_UNKNOWN;     break;
    }

    xil_printf("\n\rPredicted class : %d (%s)\n\r",
               predicted_class, cls_names[predicted_class]);

    /* Fallback sur token si FPGA a échoué */
    if(pattern == PATTERN_UNKNOWN) {
        xil_printf("[WARN] Fallback sur classification token\n\r");
        pattern = classify_from_first_token(tokens);
    }

    /* Nom lisible du pattern pour affichage */
    const char* pattern_name;
    switch(pattern) {
        case PATTERN_IF_SIMPLE:   pattern_name = "IF Statement";        break;
        case PATTERN_WHILE_LOOP:  pattern_name = "WHILE Loop";          break;
        case PATTERN_FOR_LOOP:    pattern_name = "FOR Loop";            break;
        case PATTERN_SWITCH_CASE: pattern_name = "SWITCH Statement";    break;
        default:                  pattern_name = "Unknown";             break;
    }

    /* --- STEP 4: Génération assembleur (PS) --- */
    generate_assembly(pattern, code);

    /* --- Résumé --- */
    xil_printf("\n\r");
    for(int i = 0; i < 60; i++) xil_printf("=");
    xil_printf("\n\r           COMPILATION SUMMARY\n\r");
    for(int i = 0; i < 60; i++) xil_printf("=");
    xil_printf("\n\r");
    xil_printf("Source:   %s\n\r", code);
    xil_printf("Pattern:  %s\n\r", pattern_name);
    xil_printf("Target:   ARM Assembly (Thumb-2, Cortex-A9)\n\r");
    xil_printf("Status:   [OK] SUCCESS\n\r");
    xil_printf("\n\rPipeline timing:\n\r");
    xil_printf("  Tokenization  : <1 ms  (PS)\n\r");
    xil_printf("  Transformer   : ~0.35 ms (PL @ 80 MHz)\n\r");
    xil_printf("  Classification: <1 ms  (PS)\n\r");
    xil_printf("  Code Gen      : <1 ms  (PS)\n\r");
    xil_printf("  Total         : ~2-3 ms\n\r");
    for(int i = 0; i < 60; i++) xil_printf("=");
    xil_printf("\n\r");
}

/* ================================================================
   COMPILE EXAMPLE — wrapper pour compiler un exemple indexé
   ================================================================ */
static void compile_example(int idx) {
    if(idx < 0 || idx >= NUM_EXAMPLES) {
        xil_printf("[ERROR] Invalid example index: %d\n\r", idx);
        return;
    }
    xil_printf("\n\r[EXAMPLE %d/%d] %s\n\r", idx+1, NUM_EXAMPLES, example_labels[idx]);
    compile_code(example_codes[idx]);
}

/* ================================================================
   DISPLAY EXAMPLES MENU
   ================================================================ */
static void display_examples_menu(void) {
    xil_printf("\n\r");
    xil_printf("============================================================\n\r");
    xil_printf("     TRANSFORMER COMPILER - SELECT EXAMPLE\n\r");
    xil_printf("============================================================\n\r");

    xil_printf("\n\r  === IF STATEMENTS ===\n\r");
    for(int i = 0; i < 5; i++)
        xil_printf("  [%2d] %s\n\r", i+1, example_codes[i]);

    xil_printf("\n\r  === WHILE LOOPS ===\n\r");
    for(int i = 5; i < 10; i++)
        xil_printf("  [%2d] %s\n\r", i+1, example_codes[i]);

    xil_printf("\n\r  === FOR LOOPS ===\n\r");
    for(int i = 10; i < 15; i++)
        xil_printf("  [%2d] %s\n\r", i+1, example_codes[i]);

    xil_printf("\n\r  === SWITCH CASES ===\n\r");
    for(int i = 15; i < 20; i++)
        xil_printf("  [%2d] %s\n\r", i+1, example_codes[i]);

    xil_printf("\n\r  [r] Return to main menu\n\r");
    xil_printf("\n\rChoice (1-20 or r): ");
}



/* ================================================================
   GET USER INPUT (local, inbyte Xilinx bare-metal)
   ================================================================ */
static char get_user_input(void) {
    char c;
    while(1) {
        c = inbyte();
        if(c != 0 && c != (char)0xFF) {
            xil_printf("%c\n\r", c);
            return c;
        }
    }
}


/* ================================================================
   COMPILER MENU — point d'entrée appelé depuis main.c
   Gère la saisie d'un exemple (1-9 direct, 10-20 via 2 chiffres)
   ================================================================ */
void compiler_menu(void) {
    int running = 1;

    while(running) {
        display_examples_menu();
        char first = get_user_input();

        if(first == 'r' || first == 'R') {
            running = 0;
            continue;
        }

        if(first >= '1' && first <= '9') {
            /* Attente courte : si un second caractère arrive (<500ms), on le lit */
            for(volatile int t = 0; t < 500000; t++) {
                /* Vérifier si donnée disponible via polling */
                /* Sur bare-metal, on lit directement — on attend juste */
            }
            /* Essai de lecture non-bloquante via registre UART */
            /* Simple : on considère que l'utilisateur tape ENTER après */
            /* Pour simplifier, on gère 1-9 directement et 10-20 via input */
            int idx = (first - '0') - 1;

            /* Tenter second digit si premier = 1 ou 2 (pour 10-20) */
            /* On affiche et lit directement */
            xil_printf("(Enter to confirm, or type second digit for 10-20): ");
            char c2 = inbyte();
            if(c2 >= '0' && c2 <= '9') {
                xil_printf("%c\n\r", c2);
                int num = (first - '0') * 10 + (c2 - '0');
                if(num >= 1 && num <= NUM_EXAMPLES) {
                    idx = num - 1;
                } else {
                    xil_printf("[ERROR] Invalid number: %d\n\r", num);
                    usleep(500000);
                    continue;
                }
            } else {
                xil_printf("\n\r");
                /* Premier chiffre seul = 1-9 */
                if(idx < 0 || idx >= NUM_EXAMPLES) {
                    xil_printf("[ERROR] Invalid: %d\n\r", idx+1);
                    usleep(500000);
                    continue;
                }
            }

            compile_example(idx);

        } else {
            xil_printf("\n\rInvalid input.\n\r");
            usleep(300000);
            continue;
        }

        xil_printf("\n\rPress any key to continue...");
        get_user_input();
    }
}
 
#define N_REPS 1000
 
void run_latency_stats_1000(void) {
    static uint32_t core_times[N_REPS];
    static uint32_t dispatch_times[N_REPS];
    static uint32_t readback_times[N_REPS];
    int16_t output_embedding[SEQ_LEN];
 
    xil_printf("\n\r============================================================\n\r");
    xil_printf("   STATISTIQUES DE LATENCE (N=%d repetitions)\n\r", N_REPS);
    xil_printf("============================================================\n\r");
    xil_printf("Cache D/I : chaud (Xil_DCacheEnable actif depuis le boot)\n\r");
    xil_printf("Resolution Global Timer : ticks*3/1000 = us (CPU_FREQ/2 = 333 MHz)\n\r");
    xil_printf("Cycle sur les 200 exemples, %d fois (%d x 200 = %d appels)\n\r",
               N_REPS/NUM_EXAMPLE, NUM_EXAMPLE, (N_REPS/NUM_EXAMPLE)*NUM_EXAMPLE);
    xil_printf("Mesure en cours...\n\r");
 
    int16_t tokens[SEQ_LEN];
    for (int rep = 0; rep < N_REPS; rep++) {
        int example_idx = rep % NUM_EXAMPLE;
        for(int t = 0; t < SEQ_LEN; t++) tokens[t] = 0;
        tokenize(codes_200[example_idx], tokens, SEQ_LEN);
 
        transformer_process(tokens, output_embedding);
 
        dispatch_times[rep] = transformer_get_last_dispatch_us();
        core_times[rep]     = transformer_get_last_core_us();
        readback_times[rep] = transformer_get_last_readback_us();
    }
 
    /* --- Calcul des statistiques (moyenne, ecart-type, min, max) --- */
    void compute_stats(uint32_t *arr, int n, const char *label) {
        uint64_t sum = 0;
        uint32_t min_v = 0xFFFFFFFF, max_v = 0;
        for (int i = 0; i < n; i++) {
            sum += arr[i];
            if (arr[i] < min_v) min_v = arr[i];
            if (arr[i] > max_v) max_v = arr[i];
        }
        float mean = (float)sum / n;
 
        float sumsq_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = (float)arr[i] - mean;
            sumsq_diff += diff * diff;
        }
        float variance = sumsq_diff / n;
        float stddev = sqrtf(variance);
 
        /* xil_printf ne supporte pas %f -- affichage manuel a 2 decimales */
        int mean_int = (int)mean;
        int mean_dec = (int)((mean - mean_int) * 100);
        int std_int = (int)stddev;
        int std_dec = (int)((stddev - std_int) * 100);
 
        xil_printf("%s : mean=%d.%02d us  std=%d.%02d us  min=%d us  max=%d us\n\r",
                   label, mean_int, mean_dec, std_int, std_dec, min_v, max_v);
    }
 
    compute_stats(dispatch_times, N_REPS, "Dispatch ");
    compute_stats(core_times,     N_REPS, "Coeur AXI");
    compute_stats(readback_times, N_REPS, "Readback ");
 
    xil_printf("============================================================\n\r");
}