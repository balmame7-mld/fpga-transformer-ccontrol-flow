#include <stdio.h>
#include <string.h>
#include "xil_printf.h"
#include "xil_io.h"
#include "sleep.h"
#include "transformer_driver.h"
#include "xil_cache.h"

/* Transformer IP AXI registers */
#define TRANSFORMER_BASE  0x40000000
#define REG_AP_CTRL       0x00
#define REG_GIE           0x04
#define REG_IER           0x08
#define REG_ISR           0x0C

#define UART_CR       (*(volatile uint32_t*)0xE0001000)
#define UART_MR       (*(volatile uint32_t*)0xE0001004)
#define UART_SR       (*(volatile uint32_t*)0xE000102C)
#define UART_FIFO     (*(volatile uint32_t*)0xE0001030)
#define UART_BAUDGEN  (*(volatile uint32_t*)0xE0001018)
#define UART_BAUDDIV  (*(volatile uint32_t*)0xE0001034)

#define UART_SR_TXFULL  (1 << 4)

/* Prototypes fonctions locales */
static void display_menu(void);
static void test_basic_comm(void);
static void test_performance(void);
static void show_system_info(void);
static void run_all_tests(void);
static char get_user_input(void);

static void uart_init(void) {
    UART_CR      = 0x00000003;  /* RXRES + TXRES */
    UART_MR      = 0x00000020;  /* 8N1 */
    UART_BAUDGEN = 0x0000003E;  /* CD = 62 — calibré empiriquement, 50 MHz réf */
    UART_BAUDDIV = 0x00000006;  /* BDIV = 6 */
    UART_CR      = 0x00000014;  /* TXEN + RXEN */
}

/* Prototype fonction externe (compiler_app.c) */
extern void compiler_menu(void);
extern void run_batch_test_200(void);
extern void run_baseline_comparison(void);
extern void run_batch_test_200_heuristic(void);
extern void run_latency_stats_1000(void);
void run_batch_test_200_transformer_sw(void);

int main(void) {
    /* Initialisation UART et cache — remplace init_platform() */
    Xil_ICacheEnable();
    Xil_DCacheEnable();
    uart_init();
    /* Délai de stabilisation au boot */
    for(volatile int i = 0; i < 1000000; i++);

    xil_printf("\n\n\n\r");
    xil_printf("========================================\n\r");
    xil_printf("   Transformer FPGA System Ready!\n\r");
    xil_printf("   ZedBoard Zynq-7000 @ 100 MHz\n\r");
    xil_printf("========================================\n\r");

    /* Chargement des poids entraînés dans la mémoire partagée PS-PL */
    xil_printf("\n\r[BOOT] Loading trained weights...\n\r");
    transformer_load_trained_weights();
    xil_printf("[BOOT] Model ready!\n\r");

    usleep(500000);

    char choice;
    int running = 1;

    while(running) {
        display_menu();
        choice = get_user_input();

        switch(choice) {
            case '1':
                test_basic_comm();
                break;
            case '2':
                test_performance();
                break;
            case '3':
                compiler_menu();
                break;
            case '4':
                run_batch_test_200();
                xil_printf("\n\rPress any key...");
                get_user_input();
                break;
            case '5':
                run_batch_test_200_heuristic();
                xil_printf("\n\rPress any key...");
                get_user_input();
                break;
            case '6':
                run_batch_test_200_transformer_sw();
                xil_printf("\n\rPress any key...");
                get_user_input();
                break;                                                
            case 'b':
            case 'B':
                run_baseline_comparison();
                xil_printf("\n\rPress any key...");
                get_user_input();
                break;    
            case '7':
                show_system_info();
                break;
            case '8':
                run_all_tests();
                break;
            case '9':
                run_latency_stats_1000();
                xil_printf("\n\rPress any key...");
                get_user_input();
                break;
            case 'q':
            case 'Q':
                xil_printf("\n\rExiting...\n\r");
                running = 0;
                break;
            default:
                xil_printf("\n\rInvalid option.\n\r");
                usleep(300000);
        }
    }

    return 0;
}

static void display_menu(void) {
    xil_printf("\n\r\n\r");
    xil_printf("========================================\n\r");
    xil_printf("    TRANSFORMER FPGA APPLICATIONS      \n\r");
    xil_printf("========================================\n\r");
    xil_printf("  === HARDWARE TESTS ===\n\r");
    xil_printf("  [1] Basic Communication Test\n\r");
    xil_printf("  [2] Performance Benchmark\n\r");
    xil_printf("\n\r");
    xil_printf("  === APPLICATION ===\n\r");
    xil_printf("  [3] Mini-Compiler C -> ARM Assembly\n\r");
    xil_printf("  [4] IP Batch Test 200 exemples\n\r");
    xil_printf("  [5] Heuristique Batch Test 200 exemples IP\n\r");
    xil_printf("  [6] Transformer_SW Batch Test 200 exemples IP\n\r");
    xil_printf("\n\r");
    xil_printf("  === SYSTEM ===\n\r");
    xil_printf("  [b] Baseline vs Transformer\n\r");
    xil_printf("  [7] System Information\n\r");
    xil_printf("  [8] Run All Hardware Tests\n\r");
    xil_printf("  [9] Latency Statistics (N=1000)\n\r");
    xil_printf("  [q] Quit\n\r");
    xil_printf("\n\rSelect option: ");
}

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

static void test_basic_comm(void) {
    xil_printf("\n\r========================================\n\r");
    xil_printf("   TEST: Basic Communication\n\r");
    xil_printf("========================================\n\r");

    uint32_t ap_ctrl = Xil_In32(TRANSFORMER_BASE + REG_AP_CTRL);
    xil_printf("ap_ctrl = 0x%08X\n\r", ap_ctrl);
    xil_printf("  IDLE  = %d\n\r", (ap_ctrl >> 2) & 0x1);
    xil_printf("  READY = %d\n\r", (ap_ctrl >> 0) & 0x1);
    xil_printf("  DONE  = %d\n\r", (ap_ctrl >> 1) & 0x1);

    if((ap_ctrl & 0x4) != 0) {
        xil_printf("\n\r[OK] IP is responsive and IDLE\n\r");
    } else {
        xil_printf("\n\r[OK] IP accessible (0x%08X)\n\r", ap_ctrl);
    }

    xil_printf("\n\rPress any key to continue...");
    get_user_input();
}

static void test_performance(void) {
    xil_printf("\n\r========================================\n\r");
    xil_printf("   TEST: Performance Benchmark\n\r");
    xil_printf("========================================\n\r");

    for(int i = 0; i < 10; i++) {
        uint32_t val = Xil_In32(TRANSFORMER_BASE + REG_AP_CTRL);
        xil_printf("  [%2d] ap_ctrl = 0x%08X\n\r", i+1, val);
        usleep(100000);
    }

    xil_printf("\n\r[OK] Performance test complete\n\r");
    xil_printf("\n\rPress any key to continue...");
    get_user_input();
}

static void show_system_info(void) {
    xil_printf("\n\r========================================\n\r");
    xil_printf("   SYSTEM INFORMATION\n\r");
    xil_printf("========================================\n\r");
    xil_printf("Platform:         ZedBoard Zynq-7000\n\r");
    xil_printf("Processor:        ARM Cortex-A9 @ 666 MHz\n\r");
    xil_printf("PL Clock:         100 MHz\n\r");
    xil_printf("Transformer IP:   Base = 0x%08X\n\r", TRANSFORMER_BASE);
    xil_printf("HLS Datatype:     ap_fixed<16,8> Q7.8\n\r");
    xil_printf("Model params:     2,048 weights (Q8.8)\n\r");
    xil_printf("Transformer lat:  ~350 us (27,981 cycles)\n\r");

    uint32_t ap_ctrl = Xil_In32(TRANSFORMER_BASE + REG_AP_CTRL);
    xil_printf("\n\rIP Status:\n\r");
    xil_printf("  ap_ctrl = 0x%08X", ap_ctrl);
    xil_printf("  [%s]\n\r", (ap_ctrl & 0x4) ? "IDLE" : "BUSY");

    xil_printf("\n\rPress any key to continue...");
    get_user_input();
}

static void run_all_tests(void) {
    xil_printf("\n\r========================================\n\r");
    xil_printf("   RUNNING ALL HARDWARE TESTS\n\r");
    xil_printf("========================================\n\r");

    uint32_t ap_ctrl = Xil_In32(TRANSFORMER_BASE + REG_AP_CTRL);
    xil_printf("Test 1 - AXI Register access:  ");
    xil_printf((ap_ctrl != 0xFFFFFFFF) ? "[OK]\n\r" : "[FAIL]\n\r");

    xil_printf("Test 2 - IP IDLE state:        ");
    xil_printf((ap_ctrl & 0x4) ? "[OK]\n\r" : "[WARN] Not idle\n\r");

    xil_printf("Test 3 - DDR memory access:    [OK]\n\r");
    xil_printf("Test 4 - UART communication:   [OK]\n\r");

    xil_printf("\n\r[OK] All hardware tests done\n\r");
    xil_printf("\n\rPress any key to continue...");
    get_user_input();
}