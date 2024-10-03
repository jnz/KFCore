/** @file navtoolbox.h
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Navigation Toolbox Helper Functions
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

/******************************************************************************
 * DEFINES
 ******************************************************************************/

#define PI_FLOAT (3.141592653589793f)
#define RAD2DEG(x) ((x) * (180.0f / PI_FLOAT))
#define DEG2RAD(x) ((x) * (PI_FLOAT / 180.0f))
#define CLIGHT (299792458.0)    /* speed of light (m/s) */
#define OMGE (7.2921151467E-5f) /* Earth rotation rate 15deg/h */
#define GRAVITY (9.81f)         /* Gravity */

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
 * FUNCTION PROTOTYPES
 ******************************************************************************/

#ifdef __cplusplus
extern "C"
{
#endif

    /** @brief Calculate an approximate orientation from accelerometer data,
     * assuming that the accelerometer measurement is mainly gravity.
     * Beware that the equations become nearly singular near 90 degrees pitch.
     *
     * @param[in] f Specific force measurement x,y,z component (m/s^2)
     * @param[out] roll_rad Output roll angle (rad)
     * @param[out] pitch_rad Output pitch angle (rad) */
    void nav_roll_pitch_from_accelerometer(const float f[3], float* roll_rad, float* pitch_rad);

    /** @brief Calculate a matrix R that transforms from
     * the body-frame (b) to the navigation-frame (n): R^n_b.
     * @param[in] roll_rad Roll angle in (rad)
     * @param[in] pitch_rad Pitch angle in (rad)
     * @param[in] yaw_rad Yaw angle in (rad)
     * @param[out] R_output Output 3x3 matrix in column-major format */
    void nav_matrix_body2nav(const float roll_rad, const float pitch_rad, const float yaw_rad,
                             float R_output[9]);

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
    float nav_mag_heading(const float mb[3], float roll_rad, float pitch_rad);

#ifdef __cplusplus
}
#endif

/* @} */
