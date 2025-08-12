/*
 * matrix_mult.c
 * Naive C implementation for matrix multiplication (row-major)
 */

#include <stdio.h>
#include <stdlib.h>

/* A, B, C are row-major 1D arrays; all are n x n */
void matrix_multiply(float* A, float* B, float* C, int n) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            C[i * n + j] = 0.0f;
        }
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

/* Simple wrapper for ctypes */
#ifdef __cplusplus
extern "C" {
#endif
void c_matrix_multiply(float* A, float* B, float* C, int n) {
    matrix_multiply(A, B, C, n);
}
#ifdef __cplusplus
}
#endif
