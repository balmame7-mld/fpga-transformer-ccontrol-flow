/* Classifier weights - Linear(16->4) */
/* PyTorch Test Accuracy: 100.00 % */
/* W : int16 Q8.8   b : int32 Q16.16 */

#ifndef CLASSIFIER_WEIGHTS_H
#define CLASSIFIER_WEIGHTS_H

#include <stdint.h>

#define CLASSIFIER_IN_DIM  16
#define CLASSIFIER_OUT_DIM  4

#define CLASS_IF       0
#define CLASS_WHILE    1
#define CLASS_FOR      2
#define CLASS_SWITCH   3

/* W[class][dim] shape (4,16) Q8.8 */
static const int16_t classifier_W[4][16] = {
    {    -18,    -12,    -72,     18,    -32,     41,     12,     98,     24,     33,    -43,     66,    -72,     15,     49,      0 },  /* IF */
    {     48,     66,     38,     20,    -84,    -64,     30,    -52,     -6,    -58,     33,     75,     53,    -51,     40,     64 },  /* WHILE */
    {    -41,     19,    -22,     52,     33,    -30,    -82,    -33,     46,    -71,    -87,    -20,     96,     89,    -35,    -82 },  /* FOR */
    {     26,    -91,     30,     21,     80,     61,     88,    -77,    -12,     82,     20,    -43,     38,     26,    -82,    -50 }  /* SWITCH */
};

/* b[class] Q16.16 */
static const int32_t classifier_b[4] = {
            7418,  /* IF: 0.113194 */
            1104,  /* WHILE: 0.016848 */
          -13857,  /* FOR: -0.211435 */
            3166  /* SWITCH: 0.048304 */
};

#endif /* CLASSIFIER_WEIGHTS_H */
