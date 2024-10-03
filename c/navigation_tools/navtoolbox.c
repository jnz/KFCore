/** @file navtoolbox.c
 * KFCore
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Navigation Toolbox Helper Functions
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

#include <math.h>
#include <assert.h>

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

#include "navtoolbox.h"
#include "linalg.h"

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

void nav_roll_pitch_from_accelerometer(const float f[3], float* roll_rad, float* pitch_rad)
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

void nav_matrix_body2nav(const float roll_rad, const float pitch_rad, const float yaw_rad,
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

float nav_mag_heading(const float mb[3], float roll_rad, float pitch_rad)
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
