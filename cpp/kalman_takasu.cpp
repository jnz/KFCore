/** @file kalman_takasu.cpp
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Takasu Kalman Filter (C++ w. Eigen) Implementation for
 * dynamic matrices dimensions at runtime.
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

#include "kalman_takasu_eigen.h"

/******************************************************************************
 * DEFINES
 ******************************************************************************/

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

int kalman_takasu_dynamic(Eigen::Matrix<float, Eigen::Dynamic, 1>&                    x,
                          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&       P,
                          const Eigen::Matrix<float, Eigen::Dynamic, 1>&              dz,
                          const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& R,
                          const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& H)
{
    // Vanilla implementation:
    // -----------------------
    // MatrixXf S = H * P * H.transpose() + R;
    // MatrixXf K = P * H.transpose() * S.inverse();
    //          x = x + K * dz;
    // MatrixXf I = MatrixXf::Identity(x.size(), x.size());
    //          P = (I - K * H) * P;

    // Takasu formulation:
    Eigen::MatrixXf D;
    Eigen::MatrixXf L = R; // L is used as a temp matrix and preloaded with R

    // (1) D = P * H'
    D.noalias() = P.selfadjointView<Eigen::Upper>() * H.transpose();
    // (2) L = H * D + R
    L.noalias() += H * D;
    // (3) L = chol(L)
    Eigen::LLT<Eigen::MatrixXf> lltOfL(L);
    if (lltOfL.info() != Eigen::Success)
    {
        return -1; // Cholesky decomposition failed
    }
    L = lltOfL.matrixL();

    // (4) E = D * (L')^-1
    Eigen::MatrixXf E = L.triangularView<Eigen::Lower>().solve(D.transpose()).transpose();

    // (5) P = P - E * E'
    P.selfadjointView<Eigen::Upper>().rankUpdate(E, -1);

    // (6) K = E * L^-1
    Eigen::MatrixXf K =
        L.transpose().triangularView<Eigen::Upper>().solve(E.transpose()).transpose();

    x.noalias() += K * dz;

    return 0;
}

