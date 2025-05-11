/** @file kalman_udu_eigen.h
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief UDU Kalman Filter (Bierman/Thornton "Square Root" Implementation)
 * @{ */

/****************************************************************************
 * SYSTEM INCLUDE FILES
 ***************************************************************************/

#include <Eigen/Dense>

/****************************************************************************
 * FUNCTION PROTOTYPES
 ***************************************************************************/

/** @brief Square Root Kalman Filter (Bierman) Routine for a single scalar measurement.
 *
 * @param[in,out] x System state (n x 1)
 * @param[in,out] U Unit upper triangular factor of covariance matrix of a priori state uncertainty (n x n)
 * @param[in,out] d Diagonal factor (d) vector (n x 1) of the U-D factors of the covariance
 * @param[in] dz    Measurement residual dz = z - H*x (1 x 1)
 * @param[in] R     Scalar covariance of measurement uncertainty (1 x 1)
 * @param[in] H_line Row of measurement sensitivity matrix (1 x n)
 * @return 0 on success
 */
template <typename Scalar, int StateDim>
int kalman_udu_scalar_eigen(
    Eigen::Matrix<Scalar, StateDim, 1>& x,
    Eigen::Matrix<Scalar, StateDim, StateDim>& U,
    Eigen::Matrix<Scalar, StateDim, 1>& d,
    const Scalar dz,
    const Scalar R,
    const Eigen::Matrix<Scalar, 1, StateDim>& H_line)
{
    Eigen::Matrix<Scalar, StateDim, 1> a = H_line.transpose();
    a = U.transpose().template triangularView<Eigen::Upper>() * a;
    Eigen::Matrix<Scalar, StateDim, 1> b = d.array() * a.array();
    Scalar alpha = R;
    Scalar gamma = Scalar(1) / alpha;
    for (int j = 0; j < StateDim; j++)
    {
        Scalar beta = alpha;
        alpha += a[j] * b[j];
        Scalar lambda = -a[j] * gamma;
        gamma = Scalar(1) / alpha;
        d[j] *= beta * gamma;
        for (int i = 0; i < j; i++)
        {
            Scalar uij = U(i, j);
            U(i, j) = uij + b[i] * lambda;
            b[i] += b[j] * uij;
        }
    }
    x += gamma * dz * b;
    return 0;
}

/** @brief (Robust) Square Root Kalman Filter (Bierman) update routine for linear systems.
 *
 * @param[in,out] x  System state (n x 1)
 * @param[in,out] U  Unit upper triangular factor of covariance matrix of a priori state uncertainty (n x n)
 * @param[in,out] d  Diagonal factor (d) vector (n x 1) of the U-D factors
 * @param[in] z      Measurement vector z = H*x (m x 1)
 * @param[in] R      Full covariance matrix of measurement uncertainty (m x m)
 * @param[in] H      Measurement sensitivity matrix (m x n)
 * @param[in] chi2_threshold Scalar threshold for outlier classification. Set to 0.0 to disable
 * @param[in] downweight_outlier If false, measurements classified as outliers are skipped
 * @return 0 on success, -1 on failure
 */
template <typename Scalar, int StateDim, int MeasDim>
int kalman_udu_eigen(
    Eigen::Matrix<Scalar, StateDim, 1>& x,
    Eigen::Matrix<Scalar, StateDim, StateDim>& U,
    Eigen::Matrix<Scalar, StateDim, 1>& d,
    const Eigen::Matrix<Scalar, MeasDim, 1>& z,
    const Eigen::Matrix<Scalar, MeasDim, MeasDim>& R,
    const Eigen::Matrix<Scalar, MeasDim, StateDim>& H,
    const Scalar chi2_threshold = Scalar(0),
    const bool downweight_outlier = false)
{
    int retcode = 0;
    for (int i = 0; i < MeasDim; ++i)
    {
        const Scalar Rv = R(i, i);
        Scalar dz = z[i] - H.row(i) * x;
        if (chi2_threshold > Scalar(0))
        {
            Eigen::Matrix<Scalar, 1, StateDim> tmp = H.row(i) * U;
            Scalar HPHT = (tmp.array().square() * d.transpose().array()).sum();
            Scalar s = HPHT + Rv;
            Scalar mahalanobis_dist_sq = dz * dz / s;
            if (mahalanobis_dist_sq > chi2_threshold)
            {
                if (!downweight_outlier) { continue; }
                const Scalar f = mahalanobis_dist_sq / chi2_threshold;
                const Scalar new_Rv = (f - Scalar(1)) * HPHT + f * Rv;
                if (kalman_udu_scalar_eigen<Scalar, StateDim>(x, U, d, dz, new_Rv, H.row(i)) != 0)
                {
                    retcode = -1;
                }
                continue;
            }
        }
        if (kalman_udu_scalar_eigen<Scalar, StateDim>(x, U, d, dz, Rv, H.row(i)) != 0)
        {
            retcode = -1;
        }
    }
    return retcode;
}

/** @brief UDU' (Thornton) Filter Temporal / Prediction Step
 *
 * Catherine Thornton's modified weighted Gram-Schmidt orthogonalization method for the predictor update
 * of the U-D factors of the covariance matrix of estimation uncertainty in Kalman filtering.
 *
 * @param[in,out] x   (optional) State vector (n x 1), predicted with x = Phi * x
 * @param[in,out] U   Unit upper triangular factor (n x n) of the U-D factorization of P
 * @param[in,out] d   Diagonal vector (n x 1) of the U-D factorization of P
 * @param[in] Phi     State transition matrix (n x n)
 * @param[in] G       Process noise distribution matrix (n x r)
 * @param[in] Q_diag  Diagonal elements of process noise covariance (r x 1)
 */
template <typename Scalar, int StateDim, int QnoiseDim>
void kalman_udu_predict_eigen(
    Eigen::Matrix<Scalar, StateDim, 1>& x,
    Eigen::Matrix<Scalar, StateDim, StateDim>& U,
    Eigen::Matrix<Scalar, StateDim, 1>& d,
    const Eigen::Matrix<Scalar, StateDim, StateDim>& Phi,
    const Eigen::Matrix<Scalar, StateDim, QnoiseDim>& G,
    const Eigen::Matrix<Scalar, QnoiseDim, 1>& Q_diag)
{
    x = Phi * x;
    Eigen::Matrix<Scalar, StateDim, StateDim> PhiU = Phi * U.template triangularView<Eigen::Upper>();
    Eigen::Matrix<Scalar, StateDim, QnoiseDim> G_tmp = G;
    U.setIdentity();
    Eigen::Matrix<Scalar, StateDim, 1> din = d;
    for (int i = StateDim - 1; i >= 0; --i)
    {
        Scalar sigma = (PhiU.row(i).array().square() * din.transpose().array()).sum();
        for (int j = 0; j < QnoiseDim; ++j)
        {
            sigma += G_tmp(i, j) * G_tmp(i, j) * Q_diag[j];
        }
        d[i] = sigma;
        for (int j = 0; j < i; ++j)
        {
            Scalar sum = Scalar(0);
            for (int k = 0; k < StateDim; ++k)
            {
                sum += PhiU(i, k) * din[k] * PhiU(j, k);
            }
            for (int k = 0; k < QnoiseDim; ++k)
            {
                sum += G_tmp(i, k) * Q_diag[k] * G_tmp(j, k);
            }
            Scalar u = sum / d[i];
            U(j, i) = u;
            PhiU.row(j) -= u * PhiU.row(i);
            G_tmp.row(j) -= u * G_tmp.row(i);
        }
    }
}

/** @brief Decorrelate measurements via Cholesky factorization
 *
 * Converts correlated measurement vector z and matrix H such that R becomes identity,
 * using Cholesky factorization of the original R.
 *
 * @param[in,out] z Correlated measurement vector (m x 1), replaced in-place
 * @param[in,out] H Measurement sensitivity matrix (m x n), replaced in-place
 * @param[in,out] R Measurement covariance matrix (m x m), replaced in-place by identity
 * @return 0 on success, -1 if Cholesky fails
 */
template <typename Scalar, int MeasDim, int StateDim>
int decorrelate_eigen(
    Eigen::Matrix<Scalar, MeasDim, 1>& z,
    Eigen::Matrix<Scalar, MeasDim, StateDim>& H,
    Eigen::Matrix<Scalar, MeasDim, MeasDim>& R)
{
    Eigen::LLT<Eigen::Matrix<Scalar, MeasDim, MeasDim>> lltOfR(R);
    if (lltOfR.info() != Eigen::Success) { return -1; } // Cholesky decomposition failed
    auto L = lltOfR.matrixL();
    auto Lt = L.transpose();
    z = Lt.template triangularView<Eigen::Upper>().solve(z);
    H = Lt.template triangularView<Eigen::Upper>().solve(H);
    R.setIdentity();
    return 0;
}

/* @} */
