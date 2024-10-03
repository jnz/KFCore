# KFCore

![LOGO](kfcore.png)

**KFCore** is a lightweight and efficient Kalman Filter library implemented in
C, C++, and MATLAB. Designed for both embedded systems and research
applications, KFCore offers numerically stable algorithms with minimal
dependencies and low memory usage. By leveraging advanced formulations and
optimized computations, KFCore provides a robust solution for state estimation
in various projects.


## Features

- **High Numerical Stability**
  - Implements the **UDU** (Bierman/Thornton) algorithms for superior numerical stability compared to the standard Kalman Filter formulations [(2)](https://ntrs.nasa.gov/api/citations/20180003657/downloads/20180003657.pdf).
  - Includes the **Takasu formulation**, a fast and efficient implementation offering speed improvements with enhanced stability.

- **Focus on Embedded Targets**
  - Uses only **static memory allocation**, ensuring guaranteed runtime and memory usage suitable for resource-constrained environments.
  - No dynamic memory allocation, making it ideal for embedded systems.

- **No External Dependencies**
  - Written in plain **C code** with no external code dependencies.
  - Easy integration into any project without the need for additional libraries.

- **Mathematical Optimizations**
  - Leverages the symmetry and positive semi-definiteness of covariance matrices.
  - Taking advantage of triangular shaped matrices.

- **Robust**
  - Added functionality to detect measurement errors and reject them with
  a Χ² statistical test.
  - Option to reduce the influence of potential outliers based on the Mahalanobis
    distance (Chang, 2014).

- **Optional: Optimized Computations with BLAS Interface**
  - Utilizes a **BLAS interface** to take advantage of optimized BLAS libraries on the target platform.
  - By default provides a small built-in **miniblas** library for platforms without an available BLAS library.

- **Optional: C++ Support with Eigen**
  - Offers a C++ version based on the **Eigen** math library.
  - Features template functions for convenience in projects already using Eigen.


## Implementations

| Feature                               | `UDU`      | `Takasu`       |
|---------------------------------------|:----------:|--------------:|
| Numerical Stability                   | Excellent  | Good          |
| Speed                                 | Fast       | Very fast     |
| C implementation available            |   ✅        |       ✅       |
| MATLAB implementation available       |   ✅        |       ✅       |
| C++ implementation available          |   ❌        |       ✅       |
| Outlier detection                     |   ✅        |       ✅       |
| No measurement preprocessing req.     |   ❌        |       ✅       |

The `UDU` formulation offers superior numerical stability but requires an UDU
decomposition of the covariance matrix (hence the name) and additionally a
decorrelation of the measurements before they processing. This is not
the case for the `Takasu` formulation which is basically an efficient
implementation of the vanilla Kalman filter equations.

If speed is the top priority and an optimized LAPACK/BLAS implementation is
available, consider using the Takasu formulation. However, it may struggle with
stability in scenarios involving:

 - A large number of state variables
 - A large number of measurements
 - Very high precision measurements
 - Poorly conditioned measurement sensitivity matrices $\mathbf{H}$

## Benchmarks

| Average Run Time Test                                                                   | `UDU` C   | `Takasu` C  | `Takasu` C++  |
|-----------------------------------------------------------------------------------------|-----------|-------------|---------------|
| Intel i5-13600KF Desktop CPU - Kalman Update Routine                                    | 2.95 µS   | 1.97 µS     | 12.06 µS      |
| STM32F429 180 MHz Embedded CPU (ARM Cortex M4) - Kalman Update Routine                  | 103 µS    | 135 µS      | *N/A*         |
| STM32F429 180 MHz Embedded CPU (ARM Cortex M4) - Kalman Prediction Routine              | 593 µS    | 393 µS      | *N/A*         |

(based on commit `2b35963`)

*All benchmarks with -O2 optimization*.
*15 elements state vector in benchmarks, 15x15 covariance matrix, 3x1 measurement vectors*.
*C++ implementation not tested on the embedded platform*.


## Getting Started

### Installation

#### Cloning the Repository

Clone the repository:

    git clone https://github.com/jnz/KFCore.git

### C Version

- Copy `linalg.c` and `linalg.h` from the `c/` directory to your project.
- Copy `miniblas.c` and `miniblas.h` from the `c/` directory to your project.
- If your platform has an optimized BLAS library that you want to use, you
  can exclude miniblas.

**How to add the KFCore Takasu formulation to your project**
   - Add the files `kalman_takasu.c` and `kalman_takasu.h` from the `c/`
     directory to your project.
   - Include the header file in your code:

    #include "kalman_takasu.h"

**How to add the KFCore UDU formulation to your project**
   - Add the files `kalman_udu.c` and `kalman_udu.h` from the `c/`
     directory to your project.
   - Include the header file in your code:

    #include "kalman_udu.h"

### Eigen C++ Version

- Just copy the header file `kalman_takasu_eigen.h` to your project.
  The file `kalman_takasu_eigen.cpp` is only required if you want to use
  the non-templated function version.

    #include "kalman_takasu_eigen.h"    // For C++ projects

### MATLAB Version

1. **Add to MATLAB Path**
   - Add the `matlab` directory to your project or just copy the `.m` files:

     ```matlab
     addpath('kfcore/matlab');
     ```

2. **Using the Functions**
   - Utilize the provided functions for your state estimation tasks.

## Usage examples

### Quick Start Takasu Filter in C

For a 4x1 state vector and a 3x1 measurement vector an example setup is shown below.
Two important things to highlight:

- The column-major format is used (LAPACK/BLAS default)
- The matrix $\mathbf{H}$ is supplied in a transposed way (this makes the implementation more efficient)


     ```c
        float x[4]    = { 1, 1, 1, 1 }; // State vector
        float P[4*4]  = { 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f }; // Covariance matrix of state vector
        float R[3*3]  = { 0.25f, 0, 0, 0, 0.25f, 0, 0, 0, 0.25f }; // Covariance matrix of measurement
        float dz[3]   = { 0.2688f, 0.9169f, -1.1294f }; // Measurement residuals
        float Ht[4*3] = { 8, 1, 6, 1, 3, 5, 7, 2, 4, 9, 2, 3 }; // Transposed design matrix / measurement sensitivity matrix, such that
        int result    = kalman_takasu(x, P, dz, R, Ht, 4, 3, 0.0f, NULL); // Call to update routine

     ```

### Quick Start UDU Filter in C

First a state vector `x` and an initial covariance matrix `P` of the
state vector is needed, as an example:


     ```c
        float x[4]     = { 1.0f, 0.0f, 0.0f, 0.0f };
        float P[4 * 4] = { 0.5f, 0, 0, 0, 0, 0.5f, 0, 0, 0, 0, 0.5f, 0, 0, 0, 0, 0.5f };
     ```

As the UDU filter can be called a square root filter, the matrix `P` needs to
be decomposed first:


     ```c
        float U[4 * 4];
        float d[4];
     ```

The `udu` function will perform the decomposition of `P` into `U` and `d`:


     ```c
        udu(P, U, d, 4);
     ```

`d` is a vector that describes the diagonal matrix of the UDU decomposition.
After this, the `P` matrix is no longer required.

Then let's assume a measurement vector `z` with a covariance matrix `R` that
describes the uncertainty of the measurements. Note that `R` is not purely a
diagonal matrix so it describes correlations between the measurements in `z`


     ```c
        const float z[3]      = { 16.2688f, 17.9169f, 16.8706f };
        const float R[3 * 3]  = { 0.25f, 0.25f, 0.0f, 0.25f, 0.5f, 0.1f, 0.0f, 0.1f, 0.5f };
     ```

Then we need a measurement sensitivity matrix `H` but we store it in a
transposed form, that's why it is named `Ht` for transposed. Note that the
column-major form is used in this library.


     ```c
        const float Ht[4 * 3] = { 8, 1, 6, 1, 3, 5, 7, 2, 4, 9, 2, 3 };
     ```

The UDU filter can only process one scalar measurement at a time, that's why
we first need to decorrelate the measurements:


    ```c
        decorrelate(z, Ht, R, 4, 3);
    ```

After the decorrelation, the measurements have a unit variance:


    ```c
        float eye[3 * 3]  = { 1, 0, 0,
                              0, 1, 0,
                              0, 0, 1 };
    ```

Now we can finally process the measurements in the Kalman filter update step:


    ```c
        kalman_udu(x, U, d, z, eye, Ht, 4, 3, 0.0f, 0);
    ```

### Quick Start Takasu Filter in C++

For C++ a template function is used in this example. Note that float can be replaced by double.


     ```c
        const int StateDim = 15;
        const int MeasDim  = 3;

        Matrix<float, StateDim, 1>        x;
        Matrix<float, StateDim, StateDim> P;
        Matrix<float, MeasDim, MeasDim>   R;
        Matrix<float, MeasDim, StateDim>  H;
        Matrix<float, MeasDim, 1>         z;
        Matrix<float, MeasDim, 1>         dz;

        dz.noalias() = z - H * x; // Calculate residuals
        kalman_takasu_eigen<float, StateDim, MeasDim>(x, P, dz, R, H);

     ```

For dynamic matrices there is also the function `kalman_takasu_dynamic` but the template function potentially helps the compiler to generate more efficient code.


## Takasu Formulation

| Equation                                                                             | BLAS Function   | Description                      |
|--------------------------------------------------------------------------------------|-----------------|----------------------------------|
| $\mathbf{K} = \mathbf{D} \cdot \mathbf{S}^{-1}$                                      |                 | Kalman Gain                      |
| $\mathbf{D} = \mathbf{P}^{-} \cdot \mathbf{H}^T$                                     | `symm()`        | Symmetric Matrix Product         |
| $\mathbf{S} = \mathbf{H} \cdot \mathbf{D} + \mathbf{R}$                              | `gemm()`        | General Matrix Product           |
| $\mathbf{S} = \mathbf{U}\cdot \mathbf{U}^T$                                          | `potrf()`       | Cholesky Factorization           |
| $\mathbf{E} = \mathbf{D} \cdot \mathbf{U}^{-1}$                                      | `trsm()`        | Solving Triangular Matrix        |
| $\mathbf{K} = \mathbf{E} \cdot \left( \mathbf{U}^{-1} \right)^{T}$                   | `trsm()`        | Solving Triangular Matrix        |
| $\mathbf{x}^{+} = \mathbf{x}^{-} + \mathbf{K} \cdot \mathbf{dz}$                     | `gemv()`        | Matrix Vector Product            |
| $\mathbf{P}^{+} = \mathbf{P}^{-} - \mathbf{E} \cdot \mathbf{E}^T$                    | `syrk()`        | Symmetric Rank Update            |

A priori state vector: $\mathbf{x}^{-}$, a posteriori state vector: $\mathbf{x}^{+}$,
state covariance matrix $\mathbf{P}$,
measurement residual $\mathbf{dz}$,
Measurement sensitivity / design matrix $\mathbf{H}$ such that $\mathbf{dz} = \mathbf{z} - \mathbf{H}\cdot \mathbf{x}$ for a
measurement vector $\mathbf{z}$,
covariance matrix of measurement uncertainty $\mathbf{R}$ of $\mathbf{z}$.

## License

This project is licensed under the modified BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.


## References

1. Chang, G. (2014). *Robust Kalman filtering based on Mahalanobis distance as outlier judging criterion*. **Journal of Geodesy**, **88**(4), 391-401.
2. Carpenter, J. Russell, and Christopher N. D’souza (2018). [*Navigation Filter Best Practices*](https://ntrs.nasa.gov/api/citations/20180003657/downloads/20180003657.pdf). No. NF1676L-29886.

## Acknowledgments

- **Bierman/Thornton Algorithms**: For the work on numerically stable Kalman Filter implementations.
- **Tomoji Takasu**: Appreciation to T. Takasu (the author of RTKLIB) for the efficient formulation.
- **Eigen Library**: Utilized for the C++ version to offer convenient and fast matrix operations.

## Contact

For questions, suggestions, or support:

- **Email**: [jan@zwiener.org](mailto:jan@zwiener.org)
- **GitHub Issues**: [GitHub Issues Page](https://github.com/jnz/KFCore/issues)

------------------------------------------

