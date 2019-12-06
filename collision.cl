inline bool ccw(float2 A, float2 B, float2 C){
    return (C.hi-A.hi)*(B.lo-A.lo) > (B.hi-A.hi)*(C.lo-A.lo)
}
       
inline bool intersection(float2 A, float2 B, float2 C, float2 D){
    return ccw(A,C,D) != ccw(B,C,D) && ccw(A,B,C) != ccw(A,B,D)
}

__kernel void collision_lines(__global uint *result, 
    __constant float2 *p0, __constant float2 *dp, 
    __global float2 *l1, __global float2 *l2, const uint L)
{
    size_t i = get_global_id(0); // action index
    size_t ni = get_global_size(0); // number of actions
    size_t j = get_global_id(1); // point index
    
    bool c = 0; // default conflict value
    for(uint k = 0; k < L; ++k){
        c ||= intersection(p0[j], p0[j] + dp[i], l1[k], l2[k]));
    }

    result[i + ni * j] = convert_int(c)
}

/*
 p0 - initial position (x,y)
 dp - action; delta position (r, th)
 pc - point cloud point (x,y)
 bnds - region definition (r_max, dth_max)
 */
inline bool region(float2 p0, float2 dp, float2 pc, float2 bnds){
    const float2 v = pc - p0;
    // radial distance check and angular difference check
    return fast_length(v) < bnds.lo && fabs(atan2(v.y, v.x) - radians(dp.hi)) < bnds.hi;
}

/*
result - number of points in the region
p0 - state; initial position (x,y)
dp - action; delta position (r, th)
pc - point cloud point (x,y)
Np - number of points in the point cloud
bnds - region definition (r_max, dth_max) (deg)
*/
__kernel void collision_points(__global uint *result, 
    __constant float2 *p0, __constant float2 *dp, 
    __global float2 *pc, const uint Np, __constant float2 bnds)
{
    size_t i = get_global_id(0); // action index
    size_t ni = get_global_size(0); // number of actions
    size_t j = get_global_id(1); // point index
    
    uint c = 0; // default number of conflicts value
    for(uint k = 0; k < Np; ++k){
        c += region(p0[j], dp[i], pc[k]);
    }

    result[i + ni * j] = convert_int(c)
}

