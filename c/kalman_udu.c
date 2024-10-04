/** @file kalman_udu.c
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief UDU Kalman Filter
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
#include "miniblas.h" /* strmm_ */
#include "kalman_udu.h"

/******************************************************************************
 * DEFINES
 ******************************************************************************/

#ifndef KALMAN_MAX_STATE_SIZE
#define KALMAN_MAX_STATE_SIZE 32 /* kalman filter scratchpad buf size */
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

int kalman_udu_scalar(float* x, float* U, float* d, const float dz, const float R,
                      const float* H_line, int n)
{
    assert(n <= KALMAN_MAX_STATE_SIZE);

    float a[KALMAN_MAX_STATE_SIZE];
    float b[KALMAN_MAX_STATE_SIZE];
    float alpha = R;
    float gamma = 1.0f / alpha;

    {
        // calculate: a = U'*H'
        int   tmpone   = 1;
        float tmpalpha = 1.0f;
        memcpy(a, H_line, sizeof(a[0]) * n); // preload with H_line
        strmm_("L", "U", "T", "U", &n, &tmpone, &tmpalpha, U, &n, a, &n);
    }

    for (int j = 0; j < n; j++)
    {
        b[j] = d[j] * a[j]; // b = D*a = diag(d)*a
    }

    for (int j = 0; j < n; j++)
    {
        float beta = alpha;
        alpha += a[j] * b[j];
        float lambda = -a[j] * gamma;

        gamma = 1.0f / alpha; // FIXME add test to check for UDU filter health

        d[j] *= beta * gamma;
        for (int i = 0; i < j; i++)
        {
            beta                    = MAT_ELEM(U, i, j, n, n);
            MAT_ELEM(U, i, j, n, n) = beta + b[i] * lambda;
            b[i] += b[j] * beta;
        }
    }

    for (int j = 0; j < n; j++)
    {
        x[j] += gamma * dz * b[j];
    }

    return 0;
}

int kalman_udu(float* x, float* U, float* d, const float* z, const float* R, const float* Ht,
               int n, int m, float chi2_threshold, int downweight_outlier)
{
    assert(n <= KALMAN_MAX_STATE_SIZE);

    int retcode = 0;

    for (int i = 0; i < m; i++, Ht += n) /* iterate over each measurement,
                                            goto next line of H after each iteration */
    {
        float Rv = MAT_ELEM(R, i, i, m, m); /// get scalar measurement variance
        float dz = z[i];                    // calculate residual for current scalar measurement
        matmul("N", "N", 1, 1, n, -1.0f, Ht, x, 1.0f, &dz); // dz = z - H(i,:)*x

        // <robust>
        if (chi2_threshold > 0.0f)
        {
            float tmp[KALMAN_MAX_STATE_SIZE];
            float s; // for chi2 test: s = H*U*diag(d)*U'*H' + R
                     // Chang, G. (2014). Robust Kalman filtering based on
                     // Mahalanobis distance as outlier judging criterion.
                     // Journal of Geodesy, 88(4), 391-401.

            float HPHT = 0.0f; // calc. scalar result of H_line*U*diag(d)*U'*H_line'
            matmul("N", "N", 1, n, n, 1.0f, Ht, U, 0.0f, tmp); // tmp = H(i,:) * U
            for (int j = 0; j < n; j++)
            {
                HPHT += tmp[j] * tmp[j] * d[j];
            }
            s = HPHT + Rv;

            const float mahalanobis_dist_sq = dz * dz / s;
            if (mahalanobis_dist_sq > chi2_threshold) // potential outlier?
            {
                if (!downweight_outlier)
                {
                    continue; // just skip this measurement
                }
                // process this measurement, but reduce the measurement precision
                const float f = mahalanobis_dist_sq / chi2_threshold;
                Rv            = (f - 1.0f) * HPHT + f * Rv;
            }
        }
        // </robust>

        int status = kalman_udu_scalar(x, U, d, dz, Rv, Ht, n);
        if (status != 0)
        {
            retcode = -1; // still process rest of the measurement vector
        }
    }
    return retcode;
}

int decorrelate(float* z, float* Ht, float* R, int n, int m)
{
    /* Basic decorrelation in MATLAB
    [G] = chol(R); % G'*G = R
    zdecorr = (G')\z;
    Hdecorr = (G')\H;
    Rdecorr = eye(length(z)); */

    // in-place cholesky so that L*L' = R:
    int result = cholesky(R, m, 0 /* 0 means: fill upper part with zeros */);
    if (result != 0)
    {
        return -1;
    }
    // L*H_decorr = H
    // (L*H_decorr)' = H'
    // H_decorr'*L' = H' solve for H_decorr
    trisolveright(R /*L*/, Ht, m, n, "T");
    trisolve(R /*L*/, z, m, 1, "N");

    return 0;
}

void kalman_udu_predict(float* x, float* U, float* d, const float* Phi,
                        const float* G, const float* Q, int n, int r)
{
    assert(n <= KALMAN_MAX_STATE_SIZE);
    assert(r <= KALMAN_MAX_STATE_SIZE);

    if (x) //  if prediction of state vector is requested: x = Phi*x;
    {
        float tmp[KALMAN_MAX_STATE_SIZE];
        memcpy(tmp, x, sizeof(x[0])*n);
        matmul("N", "N", n, 1, n, 1.0f, Phi, tmp, 0.0f, x);
    }

    // G_tmp = G; // move to internal array for destructive updates
    float G_tmp[KALMAN_MAX_STATE_SIZE*KALMAN_MAX_STATE_SIZE];
    memcpy(G_tmp, G, sizeof(G_tmp[0])*n*r);

    // PhiU  = Phi*U; // rows of [PhiU,G] are to be orthogonalized
    float PhiU[KALMAN_MAX_STATE_SIZE*KALMAN_MAX_STATE_SIZE];
    float tmpalpha = 1.0f;
    memcpy(PhiU, Phi, sizeof(Phi[0])*n*n);
    strmm_("R", "U", "N", "U", &n, &n, &tmpalpha, U, &n, PhiU, &n);

    mateye(U, n); // U = eye(n)

    // save origin input d vector
    float din[KALMAN_MAX_STATE_SIZE];
    memcpy(din, d, sizeof(d[0])*n); // din = d

    for (int i = n-1; i >= 0; i--)
    {
        float sigma = 0.0f;
        for (int j=0;j<n;j++)
        {
            sigma += MAT_ELEM(PhiU, i, j, n, n) *
                     MAT_ELEM(PhiU, i, j, n, n) * din[j];
            if (j < r)
            {
                sigma += MAT_ELEM(G_tmp, i, j, n, r) *
                         MAT_ELEM(G_tmp, i, j, n, r) * Q[j];
            }
        }
        d[i] = sigma;
        for (int j=0;j<i;j++)
        {
            sigma = 0.0f;
            for (int k=0;k<n;k++)
            {
                sigma += MAT_ELEM(PhiU, i, k, n, n) * din[k] *
                         MAT_ELEM(PhiU, j, k, n, n);
            }
            for (int k=0;k<r;k++)
            {
                sigma += MAT_ELEM(G_tmp, i, k, n, r) *
                         Q[k] *
                         MAT_ELEM(G_tmp, j, k, n, r);
            }
            MAT_ELEM(U, j, i, n, n) = sigma / d[i];
            for (int k=0;k<n;k++)
            {
                MAT_ELEM(PhiU, j, k, n, n) -= MAT_ELEM(U, j, i, n, n)*MAT_ELEM(PhiU, i, k, n, n);
            }
            for (int k=0;k<r;k++)
            {
                MAT_ELEM(G_tmp, j, k, n, r) -= MAT_ELEM(U, j, i, n, n)*MAT_ELEM(G_tmp, i, k, n, r);
            }
        }
    }
}

/* @} */
