/** @file test.c
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Unit Test File
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

#include <stdio.h>
#include <assert.h> // assert
#include <math.h>   // fabsf

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

#include "linalg.h"
#include "navtoolbox.h"
#include "kalman_takasu.h"
#include "kalman_udu.h"
#include "benchmark/benchmark.h"

/******************************************************************************
 * DEFINES
 ******************************************************************************/

#define TEST_FLOAT_WITHIN(delta, expected, actual, message)                                        \
    assert((fabsf((expected) - (actual)) <= delta) && message)

/******************************************************************************
 * TYPEDEFS
 ******************************************************************************/

/******************************************************************************
 * LOCAL DATA DEFINITIONS
 ******************************************************************************/

/******************************************************************************
 * LOCAL FUNCTION PROTOTYPES
 ******************************************************************************/

/** @brief Fill array with a Hilbert matrix.
 * @param[out] H Output Hilbert matrix (n x n).
 * @param[in] n Dimension of H. */
static void hilbert(float* H, int n);

/** @brief Print a matrix to stdout
 * @param[in] R column-major n x m matrix
 * @param[in] n rows
 * @param[in] m cols
 * @param[in] fmt printf format string, e.g. "%.3f"
 * @param[in] name Pretty print with the name of the matrix, can be NULL */
static void matprint(const float* R, const int n, const int m, const char* fmt, const char* name);

/******************************************************************************
 * FUNCTION BODIES
 ******************************************************************************/

static void testlinalg(void)
{
    printf("Running linalg (linear algebra) tests...\n");
    /* Note: all matrices in column-major order */

    // Test Matrix Multiplication
    {
        const float A[4]    = { 1, 4, 3, 2 };             // A = 2 rows x 2 columns
        const float B[6]    = { 2, 3, 5, 6, 3, 9 };       // B= 2 rows x 3 columns
        float       C[6]    = { 1, 1, 1, 1, 1, 1 };       // Output: A*B --> 2 rows x 3 columns
        const float Cexp[6] = { 35, 44, 71, 98, 92, 92 }; // C = 3*A*B + 2*C
        const float alpha   = 3.0f;
        const float beta    = 2.0f;
        matmul("N", "N", 2, 3, 2, alpha, A, B, beta, C);
        for (int i = 0; i < 6; i++)
        {
            TEST_FLOAT_WITHIN(1.0e-08f, Cexp[i], C[i], "Error in matrix multiplication");
        }
        printf("[x] test C = alpha*A*B + beta*C (matmul)\n");
    }
    {
        const float A[6]    = { 9, 6, -5, 10, -3, 9 }; // A = 2 rows x 3 columns
        float       B[4]    = { -1, -1, -1, -1 };      // Output: A*A' --> 2 rows x 2 columns
        const float Bexp[4] = { 115, -23, -23, 217 };  // B = 1*A*A' + 0*B
        const float alpha   = 1.0f;
        const float beta    = 0.0f;
        matmul("N", "T", 2, 2, 3, alpha, A, A, beta, B);
        for (int i = 0; i < 4; i++)
        {
            TEST_FLOAT_WITHIN(1.0e-08f, Bexp[i], B[i], "Error in matrix multiplication");
        }
        printf("[x] test C = A*A' (matmul)\n");
    }
    {
        const float A[4]    = { 1, 4, 3, 2 };       // A = 2 rows x 2 columns
        const float B[6]    = { 2, 3, 5, 6, 3, 9 }; // B= 2 rows x 3 columns
        float       C[6]    = { 2, 2, 2, 2, 2, 2 }; // Output: A*B --> 2 rows x 3 columns
        const float Cexp[6] = { 21, 18, 43.5f, 40.5f, 58.5f, 40.5f }; // C = 1.5*A'*B + 0*C
        const float alpha   = 1.5f;
        const float beta    = 0.0f;
        matmul("T", "N", 2, 3, 2, alpha, A, B, beta, C);
        for (int i = 0; i < 6; i++)
        {
            TEST_FLOAT_WITHIN(1.0e-08f, Cexp[i], C[i], "Error in matrix multiplication");
        }
        printf("[x] test C = alpha*A'*B (matmul)\n");
    }
    {
        const float A[]    = { 1, -10, 5, 3, -20, 7 }; // A = 3 rows x 2 columns
        const float B[]    = { -1, 4, -2, 5, -3, 6 };  // B= 2 rows x 3 columns
        float       C[]    = { 3, 3, 3, 3 };
        const float Cexp[] = { 4, 16, -16, -46 }; // C = 1*A'*B'
        const float alpha  = 1.0f;
        const float beta   = 0.0f;

        matmul("T", "T", 2, 2, 3, alpha, A, B, beta, C);
        for (int i = 0; i < 4; i++)
        {
            TEST_FLOAT_WITHIN(1.0e-08f, Cexp[i], C[i], "Error in matrix multiplication");
        }
        printf("[x] test C = A'*B' (matmul)\n");
    }
    // Test cholesky decomposition
    {
#define CHOLESKY_TEST_N 8
        const int n = CHOLESKY_TEST_N;
        float     L[CHOLESKY_TEST_N * CHOLESKY_TEST_N];
        hilbert(L, n); // L = hilbert(n) put hilbert matrix into L
        // matprint(L, n, n, "%10.8f", "H");
        const int result = cholesky(L, n, 0); // L = chol(L) inplace calc.
        assert(result == 0 && "Cholesky calculation test failed");
        // matprint(L, n, n, "%10.8f", "L (L*L'=H)");
        // test if L*L' actually is equal to H:
        float LLt[CHOLESKY_TEST_N * CHOLESKY_TEST_N];
        matmul("N", "T", n, n, n, 1.0f, L, L, 0.0, LLt); // LLt = L*L'
        float H[CHOLESKY_TEST_N * CHOLESKY_TEST_N];
        hilbert(H, n); // recreate expected result
        matprint(H, n, n, "%8.6f", "H");
        const float threshold = 1.5e-08f; // comparable error of independent MATLAB test
        for (int i = 0; i < n * n; i++)
        {
            TEST_FLOAT_WITHIN(threshold, H[i], LLt[i], "cholesky decomp. of Hilbert matrix failed");
        }
        printf("[x] Cholesky decomposition on close to singular matrix "
               "(cholesky)\n");
    }
    // Right-hand side triangular solve
    {
        const float L[]    = { 2, 3, 0, 1 };         // 2 x 2 matrix
        float       B[]    = { 8, 18, 28, 2, 4, 6 }; // 3 x 2 matrix
        float       Xexp[] = { 1, 3, 5, 2, 4, 6 };   // X*L = B
        trisolveright(L, B, 2, 3, "N");
        const float threshold = 1.0e-08f;
        for (int i = 0; i < 2 * 3; i++)
        {
            TEST_FLOAT_WITHIN(threshold, B[i], Xexp[i], "trisolveright failed");
        }
        printf("[x] Right-hand side triangular solve (trisolveright)\n");
    }
    {
        const float L[]    = { 2, 3, 0, 1 };            // 2 x 2 matrix
        float       B[]    = { 12, 10, 8, 21, 17, 13 }; // 3 x 2 matrix
        float       Xexp[] = { 6, 5, 4, 3, 2, 1 };      // X*L' = B
        trisolveright(L, B, 2, 3, "T");
        const float threshold = 1.0e-08f;
        for (int i = 0; i < 2 * 3; i++)
        {
            TEST_FLOAT_WITHIN(threshold, B[i], Xexp[i], "trisolveright with transpose failed");
        }
        printf("[x] Right-hand side triangular solve with transpose (trisolveright)\n");
    }
    {
        float L[3 * 3];
        float Xexp[3 * 3];
        float B[3 * 3];
        hilbert(L, 3);
        hilbert(Xexp, 3);
        cholesky(L, 3, 0);
        matmul("N", "N", 3, 3, 3, 1.0f, Xexp, L, 0.0f, B);
        // matprint(Xexp, 3, 3, "%8.6f", "X");
        // matprint(L, 3, 3, "%8.6f", "L");
        // matprint(B, 3, 3, "%8.6f", "B");
        trisolveright(L, B, 3, 3, "N");
        const float threshold = 1.0e-07f;
        for (int i = 0; i < 3 * 3; i++)
        {
            TEST_FLOAT_WITHIN(threshold, B[i], Xexp[i], "trisolveright with transpose failed");
        }
        printf("[x] Right-hand side triangular solve test case #2 (trisolveright)\n");
    }
    /* FIXME: add test for cases were trisolveright could fail */
    {
        const float A[3 * 3] = { 9, 6, 8, 6, 6, 7, 8, 7, 9 };
        float       U[3 * 3] = { -1, 0, 0, -1, -1, 0, -1, -1, -1 };
        float       d[3]     = { -1, -1, -1 };
        udu(A, U, d, 3);

        // matprint(A, 3, 3, "%6.4f", "A");
        // matprint(U, 3, 3, "%6.4f", "U");
        // matprint(d, 3, 1, "%6.4f", "d");

        const float Uexp[3 * 3] = { 1, 0, 0, -0.4f, 1, 0, 0.8889f, 0.7778f, 1 };
        const float dexp[3]     = { 1.8f, 0.5556f, 9 };
        const float threshold   = 1.0e-03f;
        for (int i = 0; i < 3 * 3; i++)
        {
            TEST_FLOAT_WITHIN(threshold, U[i], Uexp[i], "UDU: U test failed");
        }
        for (int i = 0; i < 3; i++)
        {
            TEST_FLOAT_WITHIN(threshold, d[i], dexp[i], "UDU: d test failed");
        }
    }
    // Test magnetometer yaw
    {
        const float roll_rad         = DEG2RAD(45.0f);
        const float pitch_rad        = DEG2RAD(70.0f);
        const float yaw_expected_rad = DEG2RAD(170.0f);
        const float mb[3]     = { 40.2481f, -27.63536f, -22.7238f };
        const float yaw_rad = nav_mag_heading(mb, roll_rad, pitch_rad);
        const float threshold = 0.001f;

        TEST_FLOAT_WITHIN(threshold, yaw_rad, yaw_expected_rad, "Magnetometer heading test failed (nav_mag_heading)");
        printf("[x] Yaw from magnetometer (nav_mag_heading)\n");
    }
}

static void testnavtoolbox(void)
{
    printf("Running navtoolbox tests...\n");

    // Test Roll Pitch From Accelerometer, body2nav
    {
        const float f_body[3] = { 0.0f, 0.0f, -0.01f }; /* close to free fall */
        float       roll_rad, pitch_rad;
        nav_roll_pitch_from_accelerometer(f_body, &roll_rad, &pitch_rad);
        TEST_FLOAT_WITHIN(DEG2RAD(1.0e-06f), DEG2RAD(0.0f), roll_rad,
                          "Roll angle calculation incorrect");
        TEST_FLOAT_WITHIN(DEG2RAD(1.0e-06f), DEG2RAD(0.0f), pitch_rad,
                          "Pitch angle calculation incorrect");
    }
    {
        const float f_nav[3] = { 0.0f, 0.0f, -GRAVITY };
        float       R[9];
        float       f_body[3];
        nav_matrix_body2nav(DEG2RAD(10.0f), DEG2RAD(20.0f), 0.0f, R);
        // matprint(R, 3, 3, "%6.3f", "R");
        matmul("T", "N", 3, 1, 3, 1.0f, R, f_nav, 0.0f, f_body);
        // printf("f_body = %.2f %.2f %.2f\n", f_body[0], f_body[1], f_body[2]);
        float roll_rad, pitch_rad;
        nav_roll_pitch_from_accelerometer(f_body, &roll_rad, &pitch_rad);
        // printf("%.1f %.1f\n", RAD2DEG(roll_rad), RAD2DEG(pitch_rad));
        TEST_FLOAT_WITHIN(DEG2RAD(1.0e-06f), DEG2RAD(10.0f), roll_rad,
                          "Roll angle calculation incorrect");
        TEST_FLOAT_WITHIN(DEG2RAD(1.0e-06f), DEG2RAD(20.0f), pitch_rad,
                          "Pitch angle calculation incorrect");
        printf("[x] Body to navigation frame transformation "
               "(nav_matrix_body2nav)\n");
        printf("[x] Initial alignment from accelerometer "
               "(nav_roll_pitch_from_accelerometer)\n");
    }
    // Kalman Filter Tests
    {
        const float R[3 * 3]  = { 0.25f, 0, 0, 0, 0.25f, 0, 0, 0, 0.25f };
        const float dz[3]     = { 0.2688f, 0.9169f, -1.1294f };
        const float Ht[4 * 3] = { 8, 1, 6, 1, 3, 5, 7, 2, 4, 9, 2, 3 };
        float       x[4]      = { 1, 1, 1, 1 };
        float       P[4 * 4]  = { 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f };
        int         result    = kalman_takasu(x, P, dz, R, Ht, 4, 3, 0.0f, NULL);
        assert(result == 0);
        const float xexp[4]     = { 0.9064f, 0.9046f, 1.2017f, 0.9768f };
        const float threshold   = 1.0e-04f;
        const float Pexp[4 * 4] = {
            0.0081f,  0.0000f,  0.0000f, 0.0000f, -0.0006f, 0.0063f,  0.0000f,  0.0000f,
            -0.0056f, -0.0006f, 0.0081f, 0.0000f, -0.0021f, -0.0102f, -0.0021f, 0.0367f
        }; /* upper triangular part is valid */
        // matprint(x, 4, 1, "%6.3f", "x");
        // matprint(P, 4, 4, "%6.3f", "P");
        for (int i = 0; i < 4; i++)
        {
            TEST_FLOAT_WITHIN(threshold, x[i], xexp[i],
                              "nav_kalman state vector calculation failed");
        }
        for (int i = 0; i < 4 * 4; i++)
        {
            TEST_FLOAT_WITHIN(threshold, P[i], Pexp[i],
                              "nav_kalman covariance matrix calculation failed");
        }
        printf("[x] Kalman Filter Update (nav_kalman)\n");
    }
    // Test same kalman filter but now with an outlier
    {
        const float R[3 * 3]  = { 0.25f, 0, 0, 0, 0.25f, 0, 0, 0, 0.25f };
        const float dz[3]     = { 0.2688f, 0.9169f, -100.1294f }; // adding outlier to 3rd measurement
        const float Ht[4 * 3] = { 8, 1, 6, 1, 3, 5, 7, 2, 4, 9, 2, 3 };
        float       x[4]      = { 1, 1, 1, 1 };
        float       P[4 * 4]  = { 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f };
        float       chi2;
        int         result    = kalman_takasu(x, P, dz, R, Ht, 4, 3, 7.8147f, &chi2);
        assert(result == -2);
        const float xexp[4]     = { 1, 1, 1, 1 };
        float       Pexp[4 * 4] = { 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f };
        const float threshold   = 1.0e-04f;
        const float chi2exp     = 1622.8f; // from testcasegen.m kalman_takasu_robust()
        TEST_FLOAT_WITHIN(0.1f, chi2, chi2exp, "outlier test chi2 result incorrect");
        for (int i = 0; i < 4; i++)
        {
            TEST_FLOAT_WITHIN(threshold, x[i], xexp[i],
                              "nav_kalman failed to reject outlier");
        }
        for (int i = 0; i < 4 * 4; i++)
        {
            TEST_FLOAT_WITHIN(threshold, P[i], Pexp[i],
                              "nav_kalman outlier modified covariance matrix");
        }
        printf("[x] Kalman Filter Outlier Test (nav_kalman)\n");
    }
    {
        const float R[3 * 3]  = { 0.25f, 0, 0, 0, 0.25f, 0, 0, 0, 0.25f };
        const float z[3]      = { 16.2688f, 17.9169f, 16.8706f };
        const float Ht[4 * 3] = { 8, 1, 6, 1, 3, 5, 7, 2, 4, 9, 2, 3 };
        float       x[4]      = { 1, 1, 1, 1 };
        float       P[4 * 4]  = { 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f };
        float       U[4 * 4];
        float       d[4];
        int         result;

        result = udu(P, U, d, 4);
        assert(result == 0);

        result = kalman_udu(x, U, d, z, R, Ht, 4, 3, 0.0f, 0);
        assert(result == 0);

        const float xexp[4]     = { 0.906426012f, 0.904562052f, 1.201702724f, 0.976775052f };
        const float threshold   = 1.0e-06f;
        const float Uexp[4 * 4] = { 1.000000000000000f,
                                    0.0f,
                                    0.0f,
                                    0.0f,
                                    -0.619422572178478f,
                                    1.000000000000000f,
                                    0.0f,
                                    0.0f,
                                    -0.717109934386682f,
                                    -0.147377605926585f,
                                    1.000000000000000f,
                                    0.0f,
                                    -0.055997010835721f,
                                    -0.277195167517748f,
                                    -0.055997010835721f,
                                    1.000000000000000f };
        const float dexp[4]     = { 2.62467e-03f, 3.259279e-03f, 7.977724e-03f, 3.67391e-02f };
        // matprint(x, 4, 1, "%6.3f", "x");
        // matprint(U, 4, 4, "%6.3f", "U");
        // matprint(d, 4, 1, "%6.3f", "d");
        for (int i = 0; i < 4; i++)
        {
            TEST_FLOAT_WITHIN(threshold, x[i], xexp[i],
                              "kalman_udu state vector calculation failed");
            TEST_FLOAT_WITHIN(threshold, d[i], dexp[i], "kalman_udu d[] calculation failed");
        }
        for (int i = 0; i < 4 * 4; i++)
        {
            TEST_FLOAT_WITHIN(threshold, U[i], Uexp[i],
                              "kalman_udu U matrix calculation failed");
        }
        printf("[x] Kalman Filter Update (kalman_udu)\n");
    }
    // Temporal Update Test (source: predict_test())
    {
        const float Q[] = {0.1f, 0.2f};
        const float G[] = {1, 0, 0.5f, 0, 1, 0.5f};
        float x[] = {1, 2, 3};
        const float Phi[] = { 1, 0, 0, 0.5, 1, 0, 0.25, 0.1, 1 };
        float P[3*3] = {
            1.050000f,  0.170000f, -0.180000f,
            0.170000f,  1.260000f,  0.420000f,
            -0.180000f,  0.420000f,  1.040000f
        };
        int n=3;
        int r=2;

        kalman_predict(x, P, Phi, G, Q, n, r);

        float x_exp[3*1] = {
            2.750000f,  2.300000f,  3.000000f
        };
        float P_exp[3*3] = {
            1.715000f,  0.934000f,  0.340000f,
            0.934000f,  1.554400f,  0.624000f,
            0.340000f,  0.624000f,  1.115000f
        };

        const float threshold = 0.001f;
        for (int i = 0; i < n; i++)
        {
            TEST_FLOAT_WITHIN(threshold, x[i], x_exp[i],
                              "kalman_predict x calculation failed");
        }
        for (int i = 0; i < n * n; i++)
        {
            TEST_FLOAT_WITHIN(threshold, P[i], P_exp[i],
                              "kalman_udu_predict U matrix calculation failed");
        }
        printf("[x] Kalman Filter Prediction Test\n");
    }
    // decorr Test
    {
        float R[4] = { 1.328125f, 8.45f, 8.45f, 56.2525f };
        float z[]  = { 16.25f, -11.0f };
        float Ht[] = { 1.0f, -0.5f, 0.25f, 0.1f, 5.0f, -2.0f };

        const float zexp[]    = { 14.100479758212652f, -72.481609669099413f };
        const float Hexp[]    = { 0.867721831274625f,  -0.433860915637312f, 0.216930457818656f,
                                  -3.968112807452599f, 5.183967022924173f,  -2.275160677878139f };
        const float threshold = 1.0e-04f;

        int result = decorrelate(z, Ht, R, 3, 2);

        assert(result == 0);
        // matprint(z, 2, 1, "%9.7f", "zdecorr");
        // matprint(Ht, 3, 2, "%9.7f", "Hdecorr'");

        for (int i = 0; i < 2; i++)
        {
            TEST_FLOAT_WITHIN(threshold, z[i], zexp[i], "decorrelate z calculation failed");
        }
        for (int i = 0; i < 2 * 2; i++)
        {
            TEST_FLOAT_WITHIN(threshold, Ht[i], Hexp[i], "decorrelate H calculation failed");
        }
        printf("[x] Measurement decorrelation test (decorrelate)\n");
    }
    // Robust UDU Kalman Filter Test
    {
        int         result;
        const float P[]  = { 144.010f, 120.0120f, 120.012f, 100.0144f };
        float       Ht[] = { 1.0f, -0.5f, 0.1f, 5.0f };
        float       x[]  = { 10.0f, -5.0f };
        float       z[]  = { 1.250000000000000e+01f, -1.240000000000000e+02f };
        float       R[]  = { 1.328125000000000e+00f, 8.449999999999999e+00f, 8.449999999999999e+00f,
                             5.625250000000000e+01f };

        float U[2 * 2];
        float d[2];
        result = udu(P, U, d, 2);
        assert(result == 0);
        const float chi2_threshold = 3.8415f;

        decorrelate(z, Ht, R, 2, 2);
        mateye(R, 2); // set R to eye(2)
        result = kalman_udu(x, U, d, z, R, Ht, 2, 2, chi2_threshold, 1);
        assert(result == 0);

        const float x_robust_exp[] = { 9.918531929653216f, -5.068338573737113f };
        const float U_exp[]        = { 1.0f, 0, 1.198931661436124f, 1.0f };
        const float d_exp[]        = { 1.932847767499027e-03f, 2.641859030819114e+00f };
        const float threshold      = 1.0e-04f;
        for (int i = 0; i < 2; i++)
        {
            TEST_FLOAT_WITHIN(threshold, x[i], x_robust_exp[i],
                              "kalman_udu robust state vector calculation failed");
            TEST_FLOAT_WITHIN(threshold, d[i], d_exp[i],
                              "kalman_udu robust d[] calculation failed");
        }
        for (int i = 0; i < 2 * 2; i++)
        {
            TEST_FLOAT_WITHIN(threshold, U[i], U_exp[i],
                              "kalman_udu robust U matrix calculation failed");
        }
        printf("[x] Robust UDU Kalman Filter Test with Outlier\n");
    }
    // UDU Temporal Update Test (source: thornton_test())
    {
        const float Q[] = {0.1f, 0.2f};
        const float G[] = {1, 0, 0.5f, 0, 1, 0.5f};
        float x[] = {1, 2, 3};
        const float Phi[] = { 1, 0, 0, 0.5, 1, 0, 0.25, 0.1, 1 };
        float U[] = {1, 0, 0, 0.2226f, 1, 0, -0.1731f, 0.4038f, 1};
        float d[] = { 0.9648f, 1.0904f, 1.0400f };
        int n=3;
        int r=2;

        const float x_exp[] = { 2.7500f, 2.3000f, 3.0000 };
        const float U_exp[] = {1, 0, 0, 0.6171f, 1, 0, 0.3049, 0.5596, 1};
        const float d_exp[] = { 1.1524f, 1.2052f, 1.1150f };

        kalman_udu_predict(x, U, d, Phi, G, Q, n, r);

        const float threshold = 0.001f;
        for (int i = 0; i < n; i++)
        {
            TEST_FLOAT_WITHIN(threshold, x[i], x_exp[i],
                              "kalman_udu_predict x calculation failed");
            TEST_FLOAT_WITHIN(threshold, d[i], d_exp[i],
                              "kalman_udu_predict d calculation failed");
        }
        for (int i = 0; i < n * n; i++)
        {
            TEST_FLOAT_WITHIN(threshold, U[i], U_exp[i],
                              "kalman_udu_predict U matrix calculation failed");
        }
        printf("[x] UDU Kalman Filter Prediction Test\n");
    }
}

int main(int argc, char** argv)
{
    testlinalg();
    testnavtoolbox();

    printf("\n[OK] All tests completed.\n");

    benchmark();

    return 0;
}

static void hilbert(float* H, int n)
{
    /*
     *   Hilbert test matrix, lines are _almost_ linear dependent
     *
     *   [  1       1/2     1/3     ...  1/nA      ]
     *   [  1/2     1/3     1/4     ...  1/(n+1)   ]
     *   [  1/3     1/4     1/5     ...  1/(n+2)   ]
     *   [               ...                       ]
     *   [  1/n     1/(n+1) 1/(n+2) ...  1/(2*n-1) ]
     */

    int start = 1;
    for (int i = 0; i < n; i++) /* row */
    {
        int rowstart = start;
        for (int j = 0; j < n; j++) /* col */
        {
            MAT_ELEM(H, i, j, n, n) = 1.0f / rowstart;
            rowstart++;
        }
        start++;
    }
}

static void matprint(const float* R, const int n, const int m, const char* fmt, const char* name)
{
    if (name)
    {
        printf(" %s =\n", name);
        printf("\t");
    }
    for (int i = 0; i < n; i++) /* row */
    {
        for (int j = 0; j < m; j++) /* col */
        {
            printf(fmt, (double)MAT_ELEM(R, i, j, n, m));
            printf(" ");
        }
        printf("\n");
        if (name && i < (n - 1))
        {
            printf("\t");
        }
    }
}

/* @} */

