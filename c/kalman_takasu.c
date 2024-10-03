/** @file kalman_takasu.c
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Kalman Filter Implementation (Takasu Formulation)
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

#include <math.h>
#include <assert.h>
#include <string.h> /* memcpy */

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

#include "linalg.h"
#include "miniblas.h"

/******************************************************************************
 * DEFINES
 ******************************************************************************/

#ifndef KALMAN_MAX_STATE_SIZE
#define KALMAN_MAX_STATE_SIZE 32 /* kalman filter scratchpad buf size */
#endif
#ifndef KALMAN_MAX_MEASUREMENTS
#define KALMAN_MAX_MEASUREMENTS 3 /* kalman filter scratchpad buf size */
#endif

/******************************************************************************
 * TYPEDEFS
 ******************************************************************************/

/******************************************************************************
 * LOCAL DATA DEFINITIONS
 ******************************************************************************/

/******************************************************************************
 * LOCAL FUNCTION PROTOTYPES
 ******************************************************************************/

/******************************************************************************
 * FUNCTION BODIES
 ******************************************************************************/

int kalman_takasu(float* x, float* P, const float* dz, const float* R,
                  const float* Ht, int n, int m,
                  float chi2_threshold, float* chi2)
{
    float D[KALMAN_MAX_STATE_SIZE * KALMAN_MAX_MEASUREMENTS];
    float L[KALMAN_MAX_MEASUREMENTS * KALMAN_MAX_MEASUREMENTS];
    assert(n > 0 && n <= KALMAN_MAX_STATE_SIZE);
    assert(m > 0 && m <= KALMAN_MAX_MEASUREMENTS);

    /*  (1) D = P * H'              symm           |   Matrix dimensions:
     *  (2) S = H * D + R           gemm           |   D = n x m
     *  (3) L = chol(S) (L*L'=S)    potrf          |   S = m x m
     *  (4) E = D * L^-T            trsm           |   L = m x m
     *  (5) P = P - E*E'            syrk           |   E = n x m
     *  (6) K = E * L^-1            trsm           |   K = n x m
     *  (7) x = x + K*dz            gemm
     *
     *  n = state variables
     *  m = measurements */

    /* Inplace cholesky decomposition */
    /* Only update the required triangular parts (save instructions and memory access) */
    /* keep symmetry */
    /* numerically stable */

    matmulsym(P, Ht, n, m, D); // (1) D = P * H' (using upper triangular part of P)
    memcpy(L /*dst*/, R /*src*/, sizeof(float) * m * m); // Use L as temp. matrix, preload R
    matmul("T", "N", m, m, n, 1.0f, Ht, D, 1.0f, L);     // (2) L += H*D
    int result =
        cholesky(L, m, 1 /*don't fill upper triangular part of L*/); // (3) L = chol(H*D + R)
                                                                     // (inplace calculation of L)
    if (result != 0)
    {
        return -1; // Cholesky fails: bail out (*)
    }

    /* if chi2 stats are requested and/or outlier detection is activated,
     * calculated the chi2 test statistics: */
    if (chi2 || (chi2_threshold > 0.0f))
    {
        /*  Outlier test from:
            M. S. Grewal, L. R. Weill and A. P. Andrews
            Global Positioning Systems, Inertial Navigation and Integration
            John Wiley & Sons, 2000.

            chi2 = dz' * S^(-1) * dz / m
            dz' * (L*L')^(-1) * dz
            dz' * L^(-T) * L^(-1) * dz
            y = L^(-1) * dz
            L*y = dz, solve for y
            chi2 = y'*y / m
        */
        float y[KALMAN_MAX_MEASUREMENTS]; /* temp variable */
        memcpy(y, dz, sizeof(dz[0]) * m);
        trisolve(L, y, m, 1, "N"); /* L*y = dz, solve for y */
        float chi2sum = 0.0f;
        for (int i = 0; i < m; i++)
        {
            chi2sum += y[i] * y[i]; /* y'*y */
        }
        chi2sum /= (float)m;
        if (chi2)
        {
            *chi2 = chi2sum; /* supply chi2 stats requested by caller */
        }
        if ((chi2_threshold > 0.0f) && (chi2sum > chi2_threshold))
        {
            return -2; /* reject measurement */
        }
    }

    trisolveright(L, D, m, n, "T"); // (4) given L' and D, solve E*L' = D, for E, overwrite D with E
    symmetricrankupdate(P, D /*E*/, n, m); // (5) P = P - E*E'
    trisolveright(L, D /*E*/, m, n, "N");  // (6) solve K*L = E, for K, overwrite D with K
    matmul("N", "N", n, 1, m, 1.0f, D /*K*/, dz, 1.0f, x); // (7) x = x + K * dz (K is stored in D)

    /* FIXME check for P positive definite (symmetric is automatic)*/
    /* FIXME check for isfinite() in state vector */

    /* (*) If a Cholesky decomposition is found the trsm operations will succeed. */

    return 0;
}

void kalman_predict(float* x, float* P, const float* Phi,
                    const float* G, const float* Q, int n, int r)
{
    assert(r <= KALMAN_MAX_STATE_SIZE);
    float alpha, beta;

    if (x) //  if prediction of state vector is requested: x = Phi*x;
    {
        float tmp[KALMAN_MAX_STATE_SIZE];
        memcpy(tmp, x, sizeof(x[0])*n);
        matmul("N", "N", n, 1, n, 1.0f, Phi, tmp, 0.0f, x);
    }

    if (P && Phi)
    {
        // (1) Phi*P (n x n)
        float Phi_x_P[KALMAN_MAX_STATE_SIZE*KALMAN_MAX_STATE_SIZE];
        alpha = 1.0f;
        beta = 0.0f;
        ssymm_("R" /* calculate  C = B*A = Phi_x_P = Phi*P */,
               "U" /* reference upper triangular part of A */, &n, /* rows of B/C */
               &n,                                                 /* cols of B / C */
               &alpha, P, &n, (float*)Phi, &n, &beta, Phi_x_P, &n);

        if (G && Q) // P = Phi*P*Phi' + G*Q*G';
        {
            // (2) GQ = G*Q (n x r)
            float GQ[KALMAN_MAX_STATE_SIZE*KALMAN_MAX_STATE_SIZE];
            for (int j = 0; j < r; j++) // for each  column in G
            {
                for (int i = 0; i < n; i++) // scale the rows with Q(j)
                {
                    MAT_ELEM(GQ, i, j, n, r) = Q[j]*MAT_ELEM(G, i, j, n, r);
                }
            }

            // (3) save GQ*G' in P(n x n)
            matmul("N", "T", n, n, r, 1.0f, GQ, G, 0.0f, P);

            // (4) P += Phi*P*Phi'
            matmul("N", "T", n, n, n, 1.0f, Phi_x_P, Phi, 1.0f, P);
        }
        else // P = Phi*P*Phi'
        {
            matmul("N", "T", n, n, n, 1.0f, Phi_x_P, Phi, 0.0f, P);
        }
    }
}

/* @} */
