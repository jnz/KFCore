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
#include <string.h> // memcpy

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

#include "linalg.h"
#include "miniblas.h"
#include "kalman_takasu.h"
#include "kalman_udu.h"
#include "benchmark/benchmark.h"

/******************************************************************************
 * DEFINES
 ******************************************************************************/

#define TEST_FLOAT_WITHIN(delta, expected, actual, message)                                        \
    assert((fabsf((expected) - (actual)) <= delta) && message)

#define DEG2RAD(x)       (((x) * (float)M_PI / 180.0f))
#define RAD2DEG(x)       ((x) * 180.0f / (float)M_PI)

#define GRAVITY          (9.80665f)

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
static void matprint(const float* R, const int n, const int m, const char* fmt,
                     const char* name);

/** @brief Calculate an approximate orientation from accelerometer data,
 * assuming that the accelerometer measurement is mainly gravity.
 * Beware that the equations become nearly singular near 90 degrees pitch.
 *
 * @param[in] f Specific force measurement x,y,z component (m/s^2)
 * @param[out] roll_rad Output roll angle (rad)
 * @param[out] pitch_rad Output pitch angle (rad) */
static void nav_roll_pitch_from_accelerometer(const float f[3], float*
                                              roll_rad, float* pitch_rad);

/** @brief Calculate a matrix R that transforms from
 * the body-frame (b) to the navigation-frame (n): R^n_b.
 * @param[in] roll_rad Roll angle in (rad)
 * @param[in] pitch_rad Pitch angle in (rad)
 * @param[in] yaw_rad Yaw angle in (rad)
 * @param[out] R_output Output 3x3 matrix in column-major format */
static void nav_matrix_body2nav(const float roll_rad, const float pitch_rad,
                                const float yaw_rad, float R_output[9]);

/** @brief Calculate the magnetic heading from magnetometer measurements.
 * The orientation (roll/pitch) of the magnetometer measurements must be known.
 * The magnetometer measurements are given in the body frame.
 *
 * Note: Beware of pitch angles near +/- 90 degrees.
 *
 * @param[in] mb (3x1) Magnetometer measurement in body frame (Tesla or Gauss).
 *                     x (mb[0]) pointing to forward/roll-axis of the vehicle,
 *                     y (mb[1]) pointing to the right of the vehicle (pitch axis)
 *                     z (mb[2]) pointing down (yaw-axis)
 * @param[in] roll_rad Roll angle (rad) of the vehicle relative to Earth tangent plane n-frame.
 * @param[in] pitch_rad Pitch angle (rad) of the vehicle relative to Earth tangent plane n-frame.
 *
 * @return Calculated output yaw angle (rad) of the vehicle relative to magnetic North.
 *
 * Note: This is not the geodetic heading, no declination correction is applied. */
static float nav_mag_heading(const float mb[3],
                             float roll_rad, float pitch_rad);

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

/* ---------------------------------------------------------------------------
 * miniblas.c branch coverage: linalg.c's wrappers (matmul/trisolve/
 * trisolveright/matmulsym/symmetricrankupdate) each only ever call the
 * underlying strsm_/sgemm_/ssyrk_/ssymm_/strmm_ with ONE fixed set of
 * flags, so most of these reference-BLAS routines' side/uplo/transa/
 * diag/alpha/beta branches (and their parameter-check "info" paths)
 * are otherwise never exercised. These tests call miniblas.h's
 * functions directly to close that gap, ahead of a variable-renaming
 * pass on miniblas.c (fortran2c-style names -> readable ones, no
 * algorithm changes).
 *
 * Where possible the check is a property of the routine itself (e.g.
 * strsm_'s defining equation op(A)*X = alpha*B, verified against the
 * already-tested matmul()) rather than a hand-transcribed expected
 * matrix -- less error-prone and exercises every side/uplo/transa/diag
 * combination through one loop instead of 16 hand-written cases.
 * sgemm_ itself can't use matmul() as an oracle (matmul() *is* sgemm_),
 * so those cases use alpha=0/beta=0/1/other and A=identity, whose
 * results are trivial to state analytically.
 * ---------------------------------------------------------------------------
 */

static void teststrsm(void)
{
    printf("Running strsm_ (triangular solve) branch coverage tests...\n");

    const int          m = 3, n = 2;
    /* Deliberately not fully dense: idx1=(1,0) and idx6=(0,2) are 0, so
       both the "skip if zero" and "accumulate if nonzero" outcomes of
       the a[...] != 0.f checks get exercised for whichever triangle a
       given uplo keeps (the other triangle is zeroed unconditionally
       below regardless of these particular entries). */
    const float         base_a3[9] = { 4, 0, 2, 1, 5, 1, 0, 1, 6 }; /* 3x3, col-major */
    /* idx1=(1,0) is 0 too, for the same reason as base_a3 above but on
       the 2x2 side=R matrix (whose only off-diagonal slot per triangle
       otherwise never sees a zero, since uplo=U/L each only reference
       one of the two off-diagonal-adjacent entries). */
    const float         base_a2[4] = { 4, 0, 1, 3 };                /* 2x2, col-major */
    /* idx0=(0,0) and idx2=(2,0) of column 0 are 0: side=L's zero-skip
       loops over B run in *either* direction depending on uplo
       (increasing for uplo=L, decreasing for uplo=U), and this in-place
       back-substitution overwrites already-visited B entries as it
       goes -- so only the position visited *first* in a given direction
       still holds its original value when checked. idx0 covers the
       increasing (uplo=L) case, idx2 the decreasing (uplo=U) case. */
    const float         base_b[6]  = { 0, -2, 0, -4, 5, -1 };       /* 3x2, col-major */
    const char* const   sides[2]   = { "L", "R" };
    const char* const   uplos[2]   = { "U", "L" };
    const char* const   transas[3] = { "N", "T", "C" }; /* T and C are equivalent for real matrices */
    const char* const   diags[2]   = { "N", "U" };
    const float         alphas[2]  = { 1.0f, 2.0f }; /* 1.0 hits the "skip the pre-scale" branches */

    for (int si = 0; si < 2; si++)
    {
        for (int ui = 0; ui < 2; ui++)
        {
            for (int ti = 0; ti < 3; ti++)
            {
                for (int di = 0; di < 2; di++)
                {
                    for (int ai = 0; ai < 2; ai++)
                    {
                        const char* side   = sides[si];
                        const char* uplo   = uplos[ui];
                        const char* transa = transas[ti];
                        const char* diag   = diags[di];
                        const int   nrowa  = (side[0] == 'L') ? m : n;

                        float A[9];
                        memcpy(A, (nrowa == 3) ? base_a3 : base_a2,
                              sizeof(float) * (size_t)(nrowa * nrowa));
                        for (int c = 0; c < nrowa; c++)
                        {
                            for (int r = 0; r < nrowa; r++)
                            {
                                if ((uplo[0] == 'U' && r > c) || (uplo[0] == 'L' && r < c))
                                {
                                    A[r + c * nrowa] = 0.0f;
                                }
                            }
                        }
                        if (diag[0] == 'U')
                        {
                            for (int k = 0; k < nrowa; k++) { A[k + k * nrowa] = 1.0f; }
                        }

                        float Borig[6];
                        memcpy(Borig, base_b, sizeof(Borig));
                        float B[6];
                        memcpy(B, base_b, sizeof(B));

                        int       mi = m, ni = n, lda = nrowa, ldb = m;
                        float     alpha = alphas[ai];
                        const int rc =
                            strsm_(side, uplo, transa, diag, &mi, &ni, &alpha, A, &lda, B, &ldb);
                        assert(rc == 0 && "strsm_ unexpected error code");

                        /* Defining property: op(A)*X = alpha*B (side=L), or
                           X*op(A) = alpha*B (side=R), checked via matmul(). */
                        const char* transa_for_matmul = (transa[0] == 'C') ? "T" : transa;
                        float       result[6];
                        if (side[0] == 'L')
                        {
                            matmul(transa_for_matmul, "N", m, n, m, 1.0f, A, B, 0.0f, result);
                        }
                        else
                        {
                            matmul("N", transa_for_matmul, m, n, n, 1.0f, B, A, 0.0f, result);
                        }
                        for (int k = 0; k < m * n; k++)
                        {
                            TEST_FLOAT_WITHIN(1.0e-03f, alpha * Borig[k], result[k],
                                              "strsm_ residual check (op(A)*X == alpha*B) failed");
                        }
                    }
                }
            }
        }
    }
    printf("[x] strsm_ all side/uplo/transa/diag/alpha combinations (residual check)\n");

    /* alpha == 0: B must be zeroed regardless of A/side/uplo (quick
       special-case ahead of the actual solve). */
    {
        float     A[9] = { 1, 0.5f, 0.3f, 0, 1, 0.2f, 0, 0, 1 };
        float     B[6] = { 1, 2, 3, 4, 5, 6 };
        int       mi = 3, ni = 2, lda = 3, ldb = 3;
        float     alpha = 0.0f;
        const int rc = strsm_("L", "L", "N", "U", &mi, &ni, &alpha, A, &lda, B, &ldb);
        assert(rc == 0);
        for (int k = 0; k < 6; k++)
        {
            TEST_FLOAT_WITHIN(0.0f, 0.0f, B[k], "strsm_ alpha=0 must zero B");
        }
        printf("[x] strsm_ alpha=0 zeroes B\n");
    }

    /* Quick return: m==0 or n==0 must be a no-op (B untouched). */
    {
        float     A[1] = { 1.0f };
        float     B[1] = { 42.0f };
        int       mi = 0, ni = 1, lda = 1, ldb = 1;
        float     alpha = 1.0f;
        const int rc = strsm_("L", "L", "N", "U", &mi, &ni, &alpha, A, &lda, B, &ldb);
        assert(rc == 0);
        TEST_FLOAT_WITHIN(0.0f, 42.0f, B[0], "strsm_ m=0 quick return must not touch B");
        printf("[x] strsm_ m=0/n=0 quick return\n");
    }
    /* Same quick return, reached via n==0 instead of m==0. */
    {
        float     A[1] = { 1.0f };
        float     B[1] = { 42.0f };
        int       mi = 1, ni = 0, lda = 1, ldb = 1;
        float     alpha = 1.0f;
        const int rc = strsm_("L", "L", "N", "U", &mi, &ni, &alpha, A, &lda, B, &ldb);
        assert(rc == 0);
        TEST_FLOAT_WITHIN(0.0f, 42.0f, B[0], "strsm_ n=0 quick return must not touch B");
        printf("[x] strsm_ n=0 quick return\n");
    }

    /* side=R's zero-skip checks (a[...] != 0.f) only ever see a single
       array element per side/uplo/transa combination in the 2x2 matrix
       above (n=2 leaves just one off-diagonal slot per triangle, so
       one fixed value can't show both the "skip" and "take" outcome
       for that one element) -- a 3x3 matrix with a genuine mix of zero
       and nonzero off-diagonal entries covers both, for uplo=U AND
       uplo=L (each uplo reaches a *different* one of these checks). */
    {
        const float A3_upper[9] = { 2, 0, 0, 0, 4, 0, 3, 0, 5 }; /* (0,1)=0, (0,2)=3, (1,2)=0 */
        const float A3_lower[9] = { 2, 0, 3, 0, 4, 0, 0, 0, 5 }; /* (1,0)=0, (2,0)=3, (2,1)=0 */
        const float B3[6]       = { 1, 4, -2, -5, 3, 6 };        /* 2x3 */
        for (int ui = 0; ui < 2; ui++)
        {
            for (int ti = 0; ti < 2; ti++)
            {
                const char* uplo   = (ui == 0) ? "U" : "L";
                const char* transa = (ti == 0) ? "N" : "T";
                float       A[9];
                memcpy(A, (ui == 0) ? A3_upper : A3_lower, sizeof(A3_upper));
                float Borig[6];
                memcpy(Borig, B3, sizeof(B3));
                float B[6];
                memcpy(B, B3, sizeof(B3));

                int       mi = 2, ni = 3, lda = 3, ldb = 2;
                float     alpha = 2.0f;
                const int rc = strsm_("R", uplo, transa, "N", &mi, &ni, &alpha, A, &lda, B, &ldb);
                assert(rc == 0);

                float result[6];
                matmul("N", transa, 2, 3, 3, 1.0f, B, A, 0.0f, result);
                for (int k = 0; k < 6; k++)
                {
                    TEST_FLOAT_WITHIN(1.0e-03f, alpha * Borig[k], result[k],
                                      "strsm_ side=R zero/nonzero mix residual check failed");
                }
            }
        }
        printf("[x] strsm_ side=R zero-entry skip branch (uplo=U/L, transa=N/T)\n");
    }

    /* info/error codes: one invalid parameter at a time (xerbla-style),
       matching reference BLAS's parameter-checking convention. */
    {
        float     A[4]    = { 1, 0, 0, 1 };
        float     B[4]    = { 1, 2, 3, 4 };
        int       mi = 2, ni = 2, lda = 2, ldb = 2, mneg = -1, nneg = -1, ldsmall = 1;
        float     alpha = 1.0f;
        assert(strsm_("X", "L", "N", "N", &mi, &ni, &alpha, A, &lda, B, &ldb) == 1);
        assert(strsm_("L", "X", "N", "N", &mi, &ni, &alpha, A, &lda, B, &ldb) == 2);
        assert(strsm_("L", "L", "X", "N", &mi, &ni, &alpha, A, &lda, B, &ldb) == 3);
        assert(strsm_("L", "L", "N", "X", &mi, &ni, &alpha, A, &lda, B, &ldb) == 4);
        assert(strsm_("L", "L", "N", "N", &mneg, &ni, &alpha, A, &lda, B, &ldb) == 5);
        assert(strsm_("L", "L", "N", "N", &mi, &nneg, &alpha, A, &lda, B, &ldb) == 6);
        assert(strsm_("L", "L", "N", "N", &mi, &ni, &alpha, A, &ldsmall, B, &ldb) == 9);
        assert(strsm_("L", "L", "N", "N", &mi, &ni, &alpha, A, &lda, B, &ldsmall) == 11);
        printf("[x] strsm_ info/error codes for invalid parameters\n");
    }
}

/* sgemm_ IS what matmul() calls, so it can't serve as an oracle here
 * (unlike strsm_ above) -- these use A = identity (2x2) so the expected
 * C is trivially alpha*op(B) + beta*C by inspection, letting the
 * assertions stand on their own without an external reference. */
static void check_sgemm(const char* ta, const char* tb, float alpha, float beta, int k,
                        const float Cexp[4], const char* msg)
{
    const float A2[4] = { 1, 0, 0, 1 }; /* identity */
    const float B2[4] = { 0, 2, 3, 0 }; /* has both zero and nonzero entries */
    float       C[4]  = { 5, 7, 6, 8 };
    int         mi = 2, ni = 2, ki = k, lda = 2, ldb = 2, ldc = 2;
    float       al = alpha, be = beta;
    const int   rc =
        sgemm_(ta, tb, &mi, &ni, &ki, &al, (float*)A2, &lda, (float*)B2, &ldb, &be, C, &ldc);
    assert(rc == 0 && "sgemm_ unexpected error code");
    for (int i = 0; i < 4; i++) { TEST_FLOAT_WITHIN(1.0e-05f, Cexp[i], C[i], msg); }
}

static void testsgemm(void)
{
    printf("Running sgemm_ branch coverage tests...\n");

    /* NN has a 3-way beta split (0 / other / ==1 untouched); TN/TT only
       have a 2-way split (0 / else) -- exercise every one. */
    check_sgemm("N", "N", 1.0f, 0.0f, 2, (const float[4]){ 0, 2, 3, 0 }, "sgemm_ NN beta=0");
    check_sgemm("N", "N", 1.0f, 1.0f, 2, (const float[4]){ 5, 9, 9, 8 }, "sgemm_ NN beta=1");
    check_sgemm("N", "N", 1.0f, 3.0f, 2, (const float[4]){ 15, 23, 21, 24 }, "sgemm_ NN beta=other");
    check_sgemm("N", "T", 1.0f, 0.0f, 2, (const float[4]){ 0, 3, 2, 0 }, "sgemm_ NT beta=0");
    check_sgemm("N", "T", 1.0f, 1.0f, 2, (const float[4]){ 5, 10, 8, 8 }, "sgemm_ NT beta=1");
    check_sgemm("N", "T", 1.0f, 3.0f, 2, (const float[4]){ 15, 24, 20, 24 }, "sgemm_ NT beta=other");
    check_sgemm("T", "N", 1.0f, 0.0f, 2, (const float[4]){ 0, 2, 3, 0 }, "sgemm_ TN beta=0");
    check_sgemm("T", "N", 1.0f, 3.0f, 2, (const float[4]){ 15, 23, 21, 24 }, "sgemm_ TN beta=other");
    check_sgemm("T", "T", 1.0f, 0.0f, 2, (const float[4]){ 0, 3, 2, 0 }, "sgemm_ TT beta=0");
    check_sgemm("T", "T", 1.0f, 3.0f, 2, (const float[4]){ 15, 24, 20, 24 }, "sgemm_ TT beta=other");
    /* "C" (conjugate transpose) is accepted as an alternative to "T" --
       equivalent for these real-valued matrices, but a distinct info-
       check branch (only ever reached via "T" above otherwise). */
    check_sgemm("C", "N", 1.0f, 0.0f, 2, (const float[4]){ 0, 2, 3, 0 }, "sgemm_ transa=C");
    check_sgemm("N", "C", 1.0f, 0.0f, 2, (const float[4]){ 0, 3, 2, 0 }, "sgemm_ transb=C");
    printf("[x] sgemm_ NN/NT/TN/TT beta branches\n");

    /* alpha == 0 special case (checked before the transpose branching). */
    check_sgemm("N", "N", 0.0f, 0.0f, 2, (const float[4]){ 0, 0, 0, 0 }, "sgemm_ alpha=0 beta=0");
    check_sgemm("N", "N", 0.0f, 3.0f, 2, (const float[4]){ 15, 21, 18, 24 }, "sgemm_ alpha=0 beta=other");
    printf("[x] sgemm_ alpha=0 branch\n");

    /* k == 0: the inner accumulation loop never runs, so C reduces to
       the same beta-only scaling for every transpose combination. */
    {
        const char* tas[2] = { "N", "T" };
        const char* tbs[2] = { "N", "T" };
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                check_sgemm(tas[i], tbs[j], 1.0f, 0.0f, 0, (const float[4]){ 0, 0, 0, 0 },
                           "sgemm_ k=0 beta=0");
                check_sgemm(tas[i], tbs[j], 1.0f, 1.0f, 0, (const float[4]){ 5, 7, 6, 8 },
                           "sgemm_ k=0 beta=1");
                check_sgemm(tas[i], tbs[j], 1.0f, 3.0f, 0, (const float[4]){ 15, 21, 18, 24 },
                           "sgemm_ k=0 beta=other");
            }
        }
    }
    printf("[x] sgemm_ k=0 quick path for all transpose combinations\n");

    /* Quick return via m==0 / n==0 (as opposed to the alpha==0/k==0
       sub-condition already exercised above): must be a no-op. */
    {
        float     A[1] = { 1.0f }, B[1] = { 1.0f }, C[1] = { 42.0f };
        int       mi = 0, ni = 1, ki = 1, lda = 1, ldb = 1, ldc = 1;
        float     alpha = 1.0f, beta = 1.0f;
        const int rc = sgemm_("N", "N", &mi, &ni, &ki, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        assert(rc == 0);
        TEST_FLOAT_WITHIN(0.0f, 42.0f, C[0], "sgemm_ m=0 quick return must not touch C");
        printf("[x] sgemm_ m=0/n=0 quick return\n");
    }
    /* Same quick return, reached via n==0 (with m!=0) instead of m==0. */
    {
        float     A[1] = { 1.0f }, B[1] = { 1.0f }, C[1] = { 42.0f };
        int       mi = 1, ni = 0, ki = 1, lda = 1, ldb = 1, ldc = 1;
        float     alpha = 1.0f, beta = 1.0f;
        const int rc = sgemm_("N", "N", &mi, &ni, &ki, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        assert(rc == 0);
        TEST_FLOAT_WITHIN(0.0f, 42.0f, C[0], "sgemm_ n=0 quick return must not touch C");
        printf("[x] sgemm_ n=0 quick return\n");
    }

    /* info/error codes: one invalid parameter at a time. */
    {
        float     A[4] = { 1, 0, 0, 1 }, B[4] = { 1, 0, 0, 1 }, C[4] = { 0, 0, 0, 0 };
        int       mi = 2, ni = 2, ki = 2, lda = 2, ldb = 2, ldc = 2;
        int       mneg = -1, nneg = -1, kneg = -1, ldsmall = 1;
        float     alpha = 1.0f, beta = 0.0f;
        assert(sgemm_("X", "N", &mi, &ni, &ki, &alpha, A, &lda, B, &ldb, &beta, C, &ldc) == 1);
        assert(sgemm_("N", "X", &mi, &ni, &ki, &alpha, A, &lda, B, &ldb, &beta, C, &ldc) == 2);
        assert(sgemm_("N", "N", &mneg, &ni, &ki, &alpha, A, &lda, B, &ldb, &beta, C, &ldc) == 3);
        assert(sgemm_("N", "N", &mi, &nneg, &ki, &alpha, A, &lda, B, &ldb, &beta, C, &ldc) == 4);
        assert(sgemm_("N", "N", &mi, &ni, &kneg, &alpha, A, &lda, B, &ldb, &beta, C, &ldc) == 5);
        assert(sgemm_("N", "N", &mi, &ni, &ki, &alpha, A, &ldsmall, B, &ldb, &beta, C, &ldc) == 8);
        assert(sgemm_("N", "N", &mi, &ni, &ki, &alpha, A, &lda, B, &ldsmall, &beta, C, &ldc) == 10);
        assert(sgemm_("N", "N", &mi, &ni, &ki, &alpha, A, &lda, B, &ldb, &beta, C, &ldsmall) == 13);
        printf("[x] sgemm_ info/error codes for invalid parameters\n");
    }
}

/* ssyrk_ only touches (reads/writes) the triangle named by uplo -- each
 * check verifies the referenced entries against the expected value AND
 * that the other (unreferenced) entry is left exactly as it started. */
static void check_ssyrk(const char* uplo, const char* trans, float alpha, float beta,
                        float exp_diag0, float exp_diag1, float exp_offref, int upper,
                        const char* msg)
{
    const float A2[4]        = { 1, 0, 0, 1 }; /* identity: A*A' == A'*A == I */
    float       C[4]         = { 5, 7, 6, 8 };
    const float untouched_lo = 7.0f; /* C[1] = (1,0), original */
    const float untouched_up = 6.0f; /* C[2] = (0,1), original */
    int         ni = 2, ki = 2, lda = 2, ldc = 2;
    float       al = alpha, be = beta;
    const int   rc = ssyrk_(uplo, trans, &ni, &ki, &al, (float*)A2, &lda, &be, C, &ldc);
    assert(rc == 0 && "ssyrk_ unexpected error code");
    TEST_FLOAT_WITHIN(1.0e-05f, exp_diag0, C[0], msg);
    TEST_FLOAT_WITHIN(1.0e-05f, exp_diag1, C[3], msg);
    if (upper)
    {
        TEST_FLOAT_WITHIN(1.0e-05f, exp_offref, C[2], msg);
        TEST_FLOAT_WITHIN(1.0e-05f, untouched_lo, C[1], "ssyrk_ touched the unreferenced triangle");
    }
    else
    {
        TEST_FLOAT_WITHIN(1.0e-05f, exp_offref, C[1], msg);
        TEST_FLOAT_WITHIN(1.0e-05f, untouched_up, C[2], "ssyrk_ touched the unreferenced triangle");
    }
}

static void testssyrk(void)
{
    printf("Running ssyrk_ branch coverage tests...\n");

    /* uplo x trans, beta = 0 / other -- trans=N and trans=T give the
       same numbers here (A*A' == A'*A == I for A = identity) but are
       different source lines, so both need to run. */
    check_ssyrk("U", "N", 1.0f, 0.0f, 1, 1, 0, 1, "ssyrk_ U/N beta=0");
    check_ssyrk("U", "N", 1.0f, 3.0f, 16, 25, 18, 1, "ssyrk_ U/N beta=other");
    check_ssyrk("U", "T", 1.0f, 0.0f, 1, 1, 0, 1, "ssyrk_ U/T beta=0");
    check_ssyrk("U", "T", 1.0f, 3.0f, 16, 25, 18, 1, "ssyrk_ U/T beta=other");
    check_ssyrk("L", "N", 1.0f, 0.0f, 1, 1, 0, 0, "ssyrk_ L/N beta=0");
    check_ssyrk("L", "N", 1.0f, 3.0f, 16, 25, 21, 0, "ssyrk_ L/N beta=other");
    check_ssyrk("L", "T", 1.0f, 0.0f, 1, 1, 0, 0, "ssyrk_ L/T beta=0");
    check_ssyrk("L", "T", 1.0f, 3.0f, 16, 25, 21, 0, "ssyrk_ L/T beta=other");
    /* beta == 1: the "untouched" skip of the beta-scaling loop. */
    check_ssyrk("U", "N", 1.0f, 1.0f, 6, 9, 6, 1, "ssyrk_ U/N beta=1");
    check_ssyrk("L", "N", 1.0f, 1.0f, 6, 9, 7, 0, "ssyrk_ L/N beta=1");
    /* "C" (conjugate transpose) is accepted as an alternative to "T". */
    check_ssyrk("U", "C", 1.0f, 0.0f, 1, 1, 0, 1, "ssyrk_ trans=C");
    printf("[x] ssyrk_ uplo/trans/beta branches (referenced triangle only)\n");

    /* alpha == 0 special case, uplo x beta. */
    check_ssyrk("U", "N", 0.0f, 0.0f, 0, 0, 0, 1, "ssyrk_ alpha=0 U beta=0");
    check_ssyrk("U", "N", 0.0f, 3.0f, 15, 24, 18, 1, "ssyrk_ alpha=0 U beta=other");
    check_ssyrk("L", "N", 0.0f, 0.0f, 0, 0, 0, 0, "ssyrk_ alpha=0 L beta=0");
    check_ssyrk("L", "N", 0.0f, 3.0f, 15, 24, 21, 0, "ssyrk_ alpha=0 L beta=other");
    printf("[x] ssyrk_ alpha=0 branch\n");

    /* Quick return: n==0, or (alpha==0 || k==0) with beta==1 -- a no-op. */
    {
        float     A[1] = { 1.0f };
        float     C[1] = { 42.0f };
        int       ni = 0, ki = 1, lda = 1, ldc = 1;
        float     alpha = 1.0f, beta = 1.0f;
        const int rc = ssyrk_("U", "N", &ni, &ki, &alpha, A, &lda, &beta, C, &ldc);
        assert(rc == 0);
        TEST_FLOAT_WITHIN(0.0f, 42.0f, C[0], "ssyrk_ n=0 quick return must not touch C");
        printf("[x] ssyrk_ n=0 quick return\n");
    }
    /* Same quick return, reached via k==0 && beta==1 instead of n==0. */
    {
        float     A[1] = { 1.0f };
        float     C[1] = { 42.0f };
        int       ni = 1, ki = 0, lda = 1, ldc = 1;
        float     alpha = 1.0f, beta = 1.0f;
        const int rc = ssyrk_("U", "N", &ni, &ki, &alpha, A, &lda, &beta, C, &ldc);
        assert(rc == 0);
        TEST_FLOAT_WITHIN(0.0f, 42.0f, C[0], "ssyrk_ k=0/beta=1 quick return must not touch C");
        printf("[x] ssyrk_ k=0/beta=1 quick return\n");
    }

    /* info/error codes: one invalid parameter at a time. */
    {
        float     A[4] = { 1, 0, 0, 1 }, C[4] = { 0, 0, 0, 0 };
        int       ni = 2, ki = 2, lda = 2, ldc = 2, nneg = -1, kneg = -1, ldsmall = 1;
        float     alpha = 1.0f, beta = 0.0f;
        assert(ssyrk_("X", "N", &ni, &ki, &alpha, A, &lda, &beta, C, &ldc) == 1);
        assert(ssyrk_("U", "X", &ni, &ki, &alpha, A, &lda, &beta, C, &ldc) == 2);
        assert(ssyrk_("U", "N", &nneg, &ki, &alpha, A, &lda, &beta, C, &ldc) == 3);
        assert(ssyrk_("U", "N", &ni, &kneg, &alpha, A, &lda, &beta, C, &ldc) == 4);
        assert(ssyrk_("U", "N", &ni, &ki, &alpha, A, &ldsmall, &beta, C, &ldc) == 7);
        assert(ssyrk_("U", "N", &ni, &ki, &alpha, A, &lda, &beta, C, &ldsmall) == 10);
        printf("[x] ssyrk_ info/error codes for invalid parameters\n");
    }
}

static void check_ssymm(const char* side, const char* uplo, float alpha, float beta,
                        const float Cexp[4], const char* msg)
{
    const float Asym[4] = { 2, 1, 1, 3 }; /* symmetric: same value seen from either triangle */
    const float B2[4]   = { 1, 0, 0, 2 };
    float       C[4]    = { 5, 7, 6, 8 };
    int         mi = 2, ni = 2, lda = 2, ldb = 2, ldc = 2;
    float       al = alpha, be = beta;
    const int   rc =
        ssymm_(side, uplo, &mi, &ni, &al, (float*)Asym, &lda, (float*)B2, &ldb, &be, C, &ldc);
    assert(rc == 0 && "ssymm_ unexpected error code");
    for (int i = 0; i < 4; i++) { TEST_FLOAT_WITHIN(1.0e-05f, Cexp[i], C[i], msg); }
}

static void testssymm(void)
{
    printf("Running ssymm_ branch coverage tests...\n");

    /* side x uplo, beta = 0 / other -- uplo U and L give the same
       numbers (A is fully symmetric) but are different source lines. */
    check_ssymm("L", "U", 1.0f, 0.0f, (const float[4]){ 2, 1, 2, 6 }, "ssymm_ L/U beta=0");
    check_ssymm("L", "U", 1.0f, 3.0f, (const float[4]){ 17, 22, 20, 30 }, "ssymm_ L/U beta=other");
    check_ssymm("L", "L", 1.0f, 0.0f, (const float[4]){ 2, 1, 2, 6 }, "ssymm_ L/L beta=0");
    check_ssymm("L", "L", 1.0f, 3.0f, (const float[4]){ 17, 22, 20, 30 }, "ssymm_ L/L beta=other");
    check_ssymm("R", "U", 1.0f, 0.0f, (const float[4]){ 2, 2, 1, 6 }, "ssymm_ R/U beta=0");
    check_ssymm("R", "U", 1.0f, 3.0f, (const float[4]){ 17, 23, 19, 30 }, "ssymm_ R/U beta=other");
    check_ssymm("R", "L", 1.0f, 0.0f, (const float[4]){ 2, 2, 1, 6 }, "ssymm_ R/L beta=0");
    check_ssymm("R", "L", 1.0f, 3.0f, (const float[4]){ 17, 23, 19, 30 }, "ssymm_ R/L beta=other");
    printf("[x] ssymm_ side/uplo/beta branches\n");

    /* side='R' has an "if (a != 0) skip" optimization not present for
       side='L'; the symmetric A above has no zero off-diagonals, so it
       never takes the skip path -- exercise it explicitly with a 3x3 A
       that does have zero entries. */
    {
        const float A3[9] = { 2, 0, 1, 0, 3, 0, 1, 0, 4 }; /* symmetric, zeros off-diag */
        const float B3[6] = { 1, 4, 2, 5, 3, 6 };          /* 2x3 */
        float       C3[6] = { 1, 1, 1, 1, 1, 1 };
        int         mi = 2, ni = 3, lda = 3, ldb = 2, ldc = 2;
        float       alpha = 2.0f, beta = 1.0f;
        const int   rc =
            ssymm_("R", "U", &mi, &ni, &alpha, (float*)A3, &lda, (float*)B3, &ldb, &beta, C3, &ldc);
        assert(rc == 0);
        const float Cexp[6] = { 11, 29, 13, 31, 27, 57 };
        for (int i = 0; i < 6; i++)
        {
            TEST_FLOAT_WITHIN(1.0e-04f, Cexp[i], C3[i], "ssymm_ side=R zero-skip branch");
        }
        printf("[x] ssymm_ side=R zero-entry skip branch\n");
    }

    /* alpha == 0 special case. */
    check_ssymm("L", "U", 0.0f, 0.0f, (const float[4]){ 0, 0, 0, 0 }, "ssymm_ alpha=0 beta=0");
    check_ssymm("L", "U", 0.0f, 3.0f, (const float[4]){ 15, 21, 18, 24 }, "ssymm_ alpha=0 beta=other");
    printf("[x] ssymm_ alpha=0 branch\n");

    /* Quick return: m==0 or n==0 or (alpha==0 && beta==1) -- a no-op. */
    {
        float     A[1] = { 1.0f };
        float     B[1] = { 1.0f };
        float     C[1] = { 42.0f };
        int       mi = 0, ni = 1, lda = 1, ldb = 1, ldc = 1;
        float     alpha = 1.0f, beta = 1.0f;
        const int rc = ssymm_("L", "U", &mi, &ni, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        assert(rc == 0);
        TEST_FLOAT_WITHIN(0.0f, 42.0f, C[0], "ssymm_ m=0 quick return must not touch C");
        printf("[x] ssymm_ m=0/n=0 quick return\n");
    }
    /* Same quick return, reached via n==0 (with m!=0) instead of m==0. */
    {
        float     A[1] = { 1.0f };
        float     B[1] = { 1.0f };
        float     C[1] = { 42.0f };
        int       mi = 1, ni = 0, lda = 1, ldb = 1, ldc = 1;
        float     alpha = 1.0f, beta = 1.0f;
        const int rc = ssymm_("L", "U", &mi, &ni, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        assert(rc == 0);
        TEST_FLOAT_WITHIN(0.0f, 42.0f, C[0], "ssymm_ n=0 quick return must not touch C");
        printf("[x] ssymm_ n=0 quick return\n");
    }
    /* Same quick return, reached via alpha==0 && beta==1 instead of m==0. */
    {
        float     A[4] = { 2, 1, 1, 3 }, B[4] = { 1, 0, 0, 2 }, C[4] = { 42, 42, 42, 42 };
        int       mi = 2, ni = 2, lda = 2, ldb = 2, ldc = 2;
        float     alpha = 0.0f, beta = 1.0f;
        const int rc = ssymm_("L", "U", &mi, &ni, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        assert(rc == 0);
        for (int i = 0; i < 4; i++)
        {
            TEST_FLOAT_WITHIN(0.0f, 42.0f, C[i], "ssymm_ alpha=0/beta=1 quick return must not touch C");
        }
        printf("[x] ssymm_ alpha=0/beta=1 quick return\n");
    }

    /* info/error codes: one invalid parameter at a time. */
    {
        float     A[4] = { 2, 1, 1, 3 }, B[4] = { 1, 0, 0, 1 }, C[4] = { 0, 0, 0, 0 };
        int       mi = 2, ni = 2, lda = 2, ldb = 2, ldc = 2, mneg = -1, nneg = -1, ldsmall = 1;
        float     alpha = 1.0f, beta = 0.0f;
        assert(ssymm_("X", "U", &mi, &ni, &alpha, A, &lda, B, &ldb, &beta, C, &ldc) == 1);
        assert(ssymm_("L", "X", &mi, &ni, &alpha, A, &lda, B, &ldb, &beta, C, &ldc) == 2);
        assert(ssymm_("L", "U", &mneg, &ni, &alpha, A, &lda, B, &ldb, &beta, C, &ldc) == 3);
        assert(ssymm_("L", "U", &mi, &nneg, &alpha, A, &lda, B, &ldb, &beta, C, &ldc) == 4);
        assert(ssymm_("L", "U", &mi, &ni, &alpha, A, &ldsmall, B, &ldb, &beta, C, &ldc) == 7);
        assert(ssymm_("L", "U", &mi, &ni, &alpha, A, &lda, B, &ldsmall, &beta, C, &ldc) == 9);
        assert(ssymm_("L", "U", &mi, &ni, &alpha, A, &lda, B, &ldb, &beta, C, &ldsmall) == 12);
        printf("[x] ssymm_ info/error codes for invalid parameters\n");
    }
}

/* strmm_'s defining property is a plain multiply (B := alpha*op(A)*B or
 * alpha*B*op(A)), so -- unlike sgemm_ -- it CAN be cross-checked against
 * matmul(): they reach the result through different code (strmm_'s own
 * hand-rolled triangular-aware loops vs. sgemm_'s general one), so this
 * isn't circular the way using matmul() to check sgemm_ itself would be. */
static void teststrmm(void)
{
    printf("Running strmm_ (triangular multiply) branch coverage tests...\n");

    const int          m = 3, n = 2;
    /* idx1=(1,0) and idx6=(0,2) of base_a3, idx1=(1,0) of base_a2, and
       base_b[0] are all 0 -- see teststrsm() for why. */
    const float         base_a3[9] = { 4, 0, 2, 1, 5, 1, 0, 1, 6 };
    const float         base_a2[4] = { 4, 0, 1, 3 };
    const float         base_b[6]  = { 0, -2, 3, -4, 5, -1 };
    const char* const   sides[2]   = { "L", "R" };
    const char* const   uplos[2]   = { "U", "L" };
    const char* const   transas[3] = { "N", "T", "C" }; /* T and C are equivalent for real matrices */
    const char* const   diags[2]   = { "N", "U" };
    const float         alphas[2]  = { 1.0f, 2.0f };

    for (int si = 0; si < 2; si++)
    {
        for (int ui = 0; ui < 2; ui++)
        {
            for (int ti = 0; ti < 3; ti++)
            {
                for (int di = 0; di < 2; di++)
                {
                    for (int ai = 0; ai < 2; ai++)
                    {
                        const char* side   = sides[si];
                        const char* uplo   = uplos[ui];
                        const char* transa = transas[ti];
                        const char* diag   = diags[di];
                        const int   nrowa  = (side[0] == 'L') ? m : n;

                        float A[9];
                        memcpy(A, (nrowa == 3) ? base_a3 : base_a2,
                              sizeof(float) * (size_t)(nrowa * nrowa));
                        for (int c = 0; c < nrowa; c++)
                        {
                            for (int r = 0; r < nrowa; r++)
                            {
                                if ((uplo[0] == 'U' && r > c) || (uplo[0] == 'L' && r < c))
                                {
                                    A[r + c * nrowa] = 0.0f;
                                }
                            }
                        }
                        if (diag[0] == 'U')
                        {
                            for (int k = 0; k < nrowa; k++) { A[k + k * nrowa] = 1.0f; }
                        }

                        float Borig[6];
                        memcpy(Borig, base_b, sizeof(Borig));
                        float B[6];
                        memcpy(B, base_b, sizeof(B));

                        int   mi = m, ni = n, lda = nrowa, ldb = m;
                        float alpha = alphas[ai];
                        strmm_(side, uplo, transa, diag, &mi, &ni, &alpha, A, &lda, B, &ldb);

                        const char* transa_for_matmul = (transa[0] == 'C') ? "T" : transa;
                        float       expected[6];
                        if (side[0] == 'L')
                        {
                            matmul(transa_for_matmul, "N", m, n, m, alpha, A, Borig, 0.0f, expected);
                        }
                        else
                        {
                            matmul("N", transa_for_matmul, m, n, n, alpha, Borig, A, 0.0f, expected);
                        }
                        for (int k = 0; k < m * n; k++)
                        {
                            TEST_FLOAT_WITHIN(1.0e-04f, expected[k], B[k],
                                              "strmm_ vs. matmul() cross-check failed");
                        }
                    }
                }
            }
        }
    }
    printf("[x] strmm_ all side/uplo/transa/diag/alpha combinations (matmul cross-check)\n");

    /* alpha == 0: B must be zeroed. */
    {
        float A[9] = { 1, 0.5f, 0.3f, 0, 1, 0.2f, 0, 0, 1 };
        float B[6] = { 1, 2, 3, 4, 5, 6 };
        int   mi = 3, ni = 2, lda = 3, ldb = 3;
        float alpha = 0.0f;
        strmm_("L", "L", "N", "U", &mi, &ni, &alpha, A, &lda, B, &ldb);
        for (int k = 0; k < 6; k++) { TEST_FLOAT_WITHIN(0.0f, 0.0f, B[k], "strmm_ alpha=0 must zero B"); }
        printf("[x] strmm_ alpha=0 zeroes B\n");
    }

    /* Quick return: m==0 or n==0 must be a no-op (B untouched). */
    {
        float A[1] = { 1.0f };
        float B[1] = { 42.0f };
        int   mi = 0, ni = 1, lda = 1, ldb = 1;
        float alpha = 1.0f;
        strmm_("L", "L", "N", "U", &mi, &ni, &alpha, A, &lda, B, &ldb);
        TEST_FLOAT_WITHIN(0.0f, 42.0f, B[0], "strmm_ m=0 quick return must not touch B");
        printf("[x] strmm_ m=0/n=0 quick return\n");
    }
    /* Same quick return, reached via n==0 instead of m==0. */
    {
        float A[1] = { 1.0f };
        float B[1] = { 42.0f };
        int   mi = 1, ni = 0, lda = 1, ldb = 1;
        float alpha = 1.0f;
        strmm_("L", "L", "N", "U", &mi, &ni, &alpha, A, &lda, B, &ldb);
        TEST_FLOAT_WITHIN(0.0f, 42.0f, B[0], "strmm_ n=0 quick return must not touch B");
        printf("[x] strmm_ n=0 quick return\n");
    }

    /* side=R's zero-skip checks (a[...] != 0.f) -- see the analogous
       comment in teststrsm() for why a 3x3 mixed zero/nonzero upper
       matrix is needed instead of the 2x2 one above -- for uplo=U AND
       uplo=L (each uplo reaches a *different* one of these checks). */
    {
        const float A3_upper[9] = { 2, 0, 0, 0, 4, 0, 3, 0, 5 }; /* (0,1)=0, (0,2)=3, (1,2)=0 */
        const float A3_lower[9] = { 2, 0, 3, 0, 4, 0, 0, 0, 5 }; /* (1,0)=0, (2,0)=3, (2,1)=0 */
        const float B3[6]       = { 1, 4, -2, -5, 3, 6 };        /* 2x3 */
        for (int ui = 0; ui < 2; ui++)
        {
            for (int ti = 0; ti < 2; ti++)
            {
                const char* uplo   = (ui == 0) ? "U" : "L";
                const char* transa = (ti == 0) ? "N" : "T";
                float       A[9];
                memcpy(A, (ui == 0) ? A3_upper : A3_lower, sizeof(A3_upper));
                float Borig[6];
                memcpy(Borig, B3, sizeof(B3));
                float B[6];
                memcpy(B, B3, sizeof(B3));

                int   mi = 2, ni = 3, lda = 3, ldb = 2;
                float alpha = 2.0f;
                strmm_("R", uplo, transa, "N", &mi, &ni, &alpha, A, &lda, B, &ldb);

                float expected[6];
                matmul("N", transa, 2, 3, 3, alpha, Borig, A, 0.0f, expected);
                for (int k = 0; k < 6; k++)
                {
                    TEST_FLOAT_WITHIN(1.0e-04f, expected[k], B[k],
                                      "strmm_ side=R zero/nonzero mix cross-check failed");
                }
            }
        }
        printf("[x] strmm_ side=R zero-entry skip branch (uplo=U/L, transa=N/T)\n");
    }

    /* info/error codes. NOTE: unlike strsm_/sgemm_/ssyrk_/ssymm_ (which
       return the specific 1-based "info" parameter index), strmm_
       returns a flat -1 for every invalid-parameter case -- unlike the
       other four reference-BLAS routines in this file. Documented here,
       not changed (no functional bug for callers that only check
       "!= 0", and the constraint on this pass is rename-only unless a
       real bug surfaces). */
    {
        float A[4] = { 1, 0, 0, 1 };
        float B[4] = { 1, 2, 3, 4 };
        int   mi = 2, ni = 2, lda = 2, ldb = 2, mneg = -1, nneg = -1, ldsmall = 1;
        float alpha = 1.0f;
        assert(strmm_("X", "L", "N", "N", &mi, &ni, &alpha, A, &lda, B, &ldb) == -1);
        assert(strmm_("L", "X", "N", "N", &mi, &ni, &alpha, A, &lda, B, &ldb) == -1);
        assert(strmm_("L", "L", "X", "N", &mi, &ni, &alpha, A, &lda, B, &ldb) == -1);
        assert(strmm_("L", "L", "N", "X", &mi, &ni, &alpha, A, &lda, B, &ldb) == -1);
        assert(strmm_("L", "L", "N", "N", &mneg, &ni, &alpha, A, &lda, B, &ldb) == -1);
        assert(strmm_("L", "L", "N", "N", &mi, &nneg, &alpha, A, &lda, B, &ldb) == -1);
        assert(strmm_("L", "L", "N", "N", &mi, &ni, &alpha, A, &ldsmall, B, &ldb) == -1);
        assert(strmm_("L", "L", "N", "N", &mi, &ni, &alpha, A, &lda, B, &ldsmall) == -1);
        printf("[x] strmm_ error codes for invalid parameters (always -1)\n");
    }
}

int main(int argc, char** argv)
{
    testlinalg();
    testnavtoolbox();
    teststrsm();
    testsgemm();
    testssyrk();
    testssymm();
    teststrmm();

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

static void nav_roll_pitch_from_accelerometer(const float f[3], float* roll_rad, float* pitch_rad)
{
    /* Source: Farrell, Jay. Aided navigation: GPS with high rate sensors.
     * McGraw-Hill, Inc., 2008.  */
    if (roll_rad)
    {
        *roll_rad = atan2f(-f[1], -f[2]); /* eq. 10.14 */
    }
    if (pitch_rad)
    {
        *pitch_rad = atan2f(f[0], SQRTF(f[1] * f[1] + f[2] * f[2])); /* eq. 10.15 */
    }
}

static void nav_matrix_body2nav(const float roll_rad, const float pitch_rad, const float yaw_rad,
                         float R_output[9])
{
    const float sinr = sinf(roll_rad);
    const float sinp = sinf(pitch_rad);
    const float siny = sinf(yaw_rad);
    const float cosr = cosf(roll_rad);
    const float cosp = cosf(pitch_rad);
    const float cosy = cosf(yaw_rad);
    /* Source: Farrell, Jay. Aided navigation: GPS with high rate sensors.
     * McGraw-Hill, Inc., 2008. eq. 2.43 */
    R_output[0] = cosp * cosy;
    R_output[1] = cosp * siny;
    R_output[2] = -sinp;
    R_output[3] = sinr * sinp * cosy - cosr * siny;
    R_output[4] = sinr * sinp * siny + cosr * cosy;
    R_output[5] = sinr * cosp;
    R_output[6] = cosr * sinp * cosy + sinr * siny;
    R_output[7] = cosr * sinp * siny - sinr * cosy;
    R_output[8] = cosr * cosp;
}

static float nav_mag_heading(const float mb[3], float roll_rad, float pitch_rad)
{
    const float sinr = sinf(roll_rad);
    const float sinp = sinf(pitch_rad);
    const float cosr = cosf(roll_rad);
    const float cosp = cosf(pitch_rad);

    /* Source: Farrell, Jay. Aided navigation: GPS with high rate sensors.
     * McGraw-Hill, Inc., 2008.  */
    /* Transform the magnetometer measurement in the body frame (mb) to the
     * w-frame.  The w-frame is an intermediate frame of reference defined by the
     * projection of the vehicle u-axis onto the Earth tangent plane */
    float mw_x = cosp*mb[0] + sinp*sinr*mb[1] + sinp*cosr*mb[2]; /* eq. 10.16 */
    float mw_y =                   cosr*mb[1] -      sinr*mb[2]; /* eq. 10.16 */

    return atan2f(-mw_y, mw_x);
}

/* @} */

