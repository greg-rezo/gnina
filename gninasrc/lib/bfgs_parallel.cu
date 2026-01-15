/*
 * bfgs_parallel.cu
 *
 * Massively parallel BFGS implementation.
 * Each CUDA thread runs an independent optimizer with no sync points.
 */

#include "bfgs_parallel.h"
#include "model.h"
#include "conf.h"
#include "quaternion.h"
#include "curl.h"
#include "gpu_util.h"
#include <curand_kernel.h>
#include <cstring>
#include <vector>
#include <algorithm>

// ============================================================================
// Device Helper Functions
// ============================================================================

// Quaternion from axis-angle (single thread)
__device__ inline qt angle_to_quaternion_device(float ax, float ay, float az) {
    float angle = sqrtf(ax*ax + ay*ay + az*az);
    if (angle > epsilon_fl) {
        float s = sinf(angle * 0.5f) / angle;
        return qt(cosf(angle * 0.5f), ax * s, ay * s, az * s);
    }
    return qt(1, 0, 0, 0);
}

// Quaternion multiplication
__device__ inline qt quat_mult(const qt& a, const qt& b) {
    return qt(
        a.R_component_1()*b.R_component_1() - a.R_component_2()*b.R_component_2()
            - a.R_component_3()*b.R_component_3() - a.R_component_4()*b.R_component_4(),
        a.R_component_1()*b.R_component_2() + a.R_component_2()*b.R_component_1()
            + a.R_component_3()*b.R_component_4() - a.R_component_4()*b.R_component_3(),
        a.R_component_1()*b.R_component_3() - a.R_component_2()*b.R_component_4()
            + a.R_component_3()*b.R_component_1() + a.R_component_4()*b.R_component_2(),
        a.R_component_1()*b.R_component_4() + a.R_component_2()*b.R_component_3()
            - a.R_component_3()*b.R_component_2() + a.R_component_4()*b.R_component_1()
    );
}

// Quaternion to rotation matrix (stored as 9 floats)
__device__ inline void quat_to_matrix(const qt& q, float* m) {
    float a = q.R_component_1();
    float b = q.R_component_2();
    float c = q.R_component_3();
    float d = q.R_component_4();

    float aa = a*a, bb = b*b, cc = c*c, dd = d*d;
    float ab = a*b, ac = a*c, ad = a*d;
    float bc = b*c, bd = b*d, cd = c*d;

    // Column-major order (matching gpu_mat)
    m[0] = aa + bb - cc - dd;
    m[1] = 2*(bc + ad);
    m[2] = 2*(bd - ac);
    m[3] = 2*(bc - ad);
    m[4] = aa - bb + cc - dd;
    m[5] = 2*(cd + ab);
    m[6] = 2*(bd + ac);
    m[7] = 2*(cd - ab);
    m[8] = aa - bb - cc + dd;
}

// Rotate vector by matrix
__device__ inline void mat_vec_mult(const float* m, const float* v, float* out) {
    out[0] = m[0]*v[0] + m[3]*v[1] + m[6]*v[2];
    out[1] = m[1]*v[0] + m[4]*v[1] + m[7]*v[2];
    out[2] = m[2]*v[0] + m[5]*v[1] + m[8]*v[2];
}

// Cross product
__device__ inline void cross_product_device(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

// Dot product
__device__ inline float dot3(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// Vector norm
__device__ inline float norm3(const float* v) {
    return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// Normalize angle to [-pi, pi]
__device__ inline float normalize_angle_device(float a) {
    const float pi = 3.14159265358979323846f;
    while (a > pi) a -= 2*pi;
    while (a < -pi) a += 2*pi;
    return a;
}

// ============================================================================
// Single-Thread Tree Transformation
// ============================================================================

__device__ void set_conf_single_thread(
    const ScoringContext& ctx,
    const float* conf,
    float* coords,
    float* node_origins,      // [num_nodes × 3] scratch
    float* node_orientations, // [num_nodes × 9] scratch (rotation matrices)
    float* node_axes          // [num_nodes × 3] output: transformed axes in lab frame
) {
    // Early exit if no atoms or missing data
    if (ctx.num_atoms == 0 || ctx.atom_local_data == nullptr) {
        return;
    }

    // Process ligand roots first (layer 0)
    for (unsigned i = 0; i < ctx.nlig_roots; i++) {
        // Extract position from conf
        const float* rigid_conf = &conf[i * 7];
        float* origin = &node_origins[i * 3];
        origin[0] = rigid_conf[0];
        origin[1] = rigid_conf[1];
        origin[2] = rigid_conf[2];

        // Extract quaternion and convert to matrix
        qt q(rigid_conf[3], rigid_conf[4], rigid_conf[5], rigid_conf[6]);
        // Normalize quaternion
        float qnorm = sqrtf(q.R_component_1()*q.R_component_1() +
                          q.R_component_2()*q.R_component_2() +
                          q.R_component_3()*q.R_component_3() +
                          q.R_component_4()*q.R_component_4());
        if (qnorm > epsilon_fl) {
            q = qt(q.R_component_1()/qnorm, q.R_component_2()/qnorm,
                   q.R_component_3()/qnorm, q.R_component_4()/qnorm);
        }
        quat_to_matrix(q, &node_orientations[i * 9]);
    }

    // Process remaining nodes layer by layer
    for (unsigned layer = 1; layer < ctx.num_layers; layer++) {
        for (unsigned nid = ctx.nlig_roots; nid < ctx.num_nodes; nid++) {
            const segment_node& node = ctx.tree_nodes[nid];
            if ((unsigned)node.layer != layer) continue;

            int parent = node.parent;
            if (parent < 0) continue;

            // Get torsion angle from conf
            float torsion = conf[nid + 6 * ctx.nlig_roots];

            // Get parent's orientation matrix
            const float* parent_mat = &node_orientations[parent * 9];
            const float* parent_origin = &node_origins[parent * 3];

            // Transform relative axis to lab frame
            float rel_axis[3] = {node.relative_axis.x, node.relative_axis.y, node.relative_axis.z};
            float axis[3];
            mat_vec_mult(parent_mat, rel_axis, axis);

            // Store transformed axis for gradient computation
            node_axes[nid * 3 + 0] = axis[0];
            node_axes[nid * 3 + 1] = axis[1];
            node_axes[nid * 3 + 2] = axis[2];

            // Transform relative origin to lab frame
            float rel_origin[3] = {node.relative_origin.x, node.relative_origin.y, node.relative_origin.z};
            float local_origin[3];
            mat_vec_mult(parent_mat, rel_origin, local_origin);

            float* origin = &node_origins[nid * 3];
            origin[0] = parent_origin[0] + local_origin[0];
            origin[1] = parent_origin[1] + local_origin[1];
            origin[2] = parent_origin[2] + local_origin[2];

            // Compute orientation: rotation about axis by torsion, then parent orientation
            // Using Rodrigues' formula to directly compute the rotation matrix
            float c = cosf(torsion);
            float s = sinf(torsion);
            float t = 1.0f - c;
            float ax = axis[0], ay = axis[1], az = axis[2];
            float anorm = sqrtf(ax*ax + ay*ay + az*az);
            if (anorm > epsilon_fl) {
                ax /= anorm; ay /= anorm; az /= anorm;
            }

            // Rodrigues' rotation formula for rotation matrix
            float rot[9];
            rot[0] = c + ax*ax*t;      rot[3] = ax*ay*t - az*s;  rot[6] = ax*az*t + ay*s;
            rot[1] = ay*ax*t + az*s;   rot[4] = c + ay*ay*t;     rot[7] = ay*az*t - ax*s;
            rot[2] = az*ax*t - ay*s;   rot[5] = az*ay*t + ax*s;  rot[8] = c + az*az*t;

            // Multiply with parent rotation matrix
            float* node_mat = &node_orientations[nid * 9];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    node_mat[i + j*3] = 0;
                    for (int k = 0; k < 3; k++) {
                        node_mat[i + j*3] += rot[i + k*3] * parent_mat[k + j*3];
                    }
                }
            }
        }
    }

    // Transform atom coordinates
    if (ctx.atom_owners == nullptr) return;

    for (unsigned atom = 0; atom < ctx.num_atoms; atom++) {
        unsigned owner = ctx.atom_owners[atom];
        const float* origin = &node_origins[owner * 3];
        const float* mat = &node_orientations[owner * 9];

        // Get local coords from marked_coord structure
        const gfloat3& local_coords = ctx.atom_local_data[atom].coords;
        float local[3] = {local_coords.x, local_coords.y, local_coords.z};

        float rotated[3];
        mat_vec_mult(mat, local, rotated);

        coords[atom * 3 + 0] = origin[0] + rotated[0];
        coords[atom * 3 + 1] = origin[1] + rotated[1];
        coords[atom * 3 + 2] = origin[2] + rotated[2];
    }
}

// ============================================================================
// Single-Thread Grid-Based Energy Evaluation
// ============================================================================

// Core trilinear interpolation on a specific data array
__device__ inline void trilinear_interp_core(
    const grid_gpu& grid,
    const array3d_gpu<fl, fl>& data,
    float x, float y, float z,
    float slope,
    float v,  // curl parameter
    float* energy,
    float* fx, float* fy, float* fz
) {
    // Convert to grid coordinates
    float sx = (x - grid.m_init.x) * grid.m_factor.x;
    float sy = (y - grid.m_init.y) * grid.m_factor.y;
    float sz = (z - grid.m_init.z) * grid.m_factor.z;

    // Handle out-of-bounds
    float miss_x = 0, miss_y = 0, miss_z = 0;
    int region_x = 0, region_y = 0, region_z = 0;

    if (sx < 0) {
        miss_x = -sx;
        region_x = -1;
        sx = 0;
    } else if (sx >= grid.m_dim_fl_minus_1.x) {
        miss_x = sx - grid.m_dim_fl_minus_1.x;
        region_x = 1;
        sx = grid.m_dim_fl_minus_1.x - epsilon_fl;
    }

    if (sy < 0) {
        miss_y = -sy;
        region_y = -1;
        sy = 0;
    } else if (sy >= grid.m_dim_fl_minus_1.y) {
        miss_y = sy - grid.m_dim_fl_minus_1.y;
        region_y = 1;
        sy = grid.m_dim_fl_minus_1.y - epsilon_fl;
    }

    if (sz < 0) {
        miss_z = -sz;
        region_z = -1;
        sz = 0;
    } else if (sz >= grid.m_dim_fl_minus_1.z) {
        miss_z = sz - grid.m_dim_fl_minus_1.z;
        region_z = 1;
        sz = grid.m_dim_fl_minus_1.z - epsilon_fl;
    }

    // Penalty for out-of-bounds
    float penalty = slope * (miss_x * grid.m_factor_inv.x +
                             miss_y * grid.m_factor_inv.y +
                             miss_z * grid.m_factor_inv.z);

    // Grid indices
    int ax = (int)sx;
    int ay = (int)sy;
    int az = (int)sz;

    // Fractional parts
    float fx_frac = sx - ax;
    float fy_frac = sy - ay;
    float fz_frac = sz - az;

    float mx = 1 - fx_frac;
    float my = 1 - fy_frac;
    float mz = 1 - fz_frac;

    // Get 8 corner values using array3d_gpu accessor
    float f000 = data(ax, ay, az);
    float f100 = data(ax+1, ay, az);
    float f010 = data(ax, ay+1, az);
    float f110 = data(ax+1, ay+1, az);
    float f001 = data(ax, ay, az+1);
    float f101 = data(ax+1, ay, az+1);
    float f011 = data(ax, ay+1, az+1);
    float f111 = data(ax+1, ay+1, az+1);

    // Trilinear interpolation
    float f = f000 * mx * my * mz +
              f100 * fx_frac * my * mz +
              f010 * mx * fy_frac * mz +
              f110 * fx_frac * fy_frac * mz +
              f001 * mx * my * fz_frac +
              f101 * fx_frac * my * fz_frac +
              f011 * mx * fy_frac * fz_frac +
              f111 * fx_frac * fy_frac * fz_frac;

    // Gradient
    float gx = -f000 * my * mz + f100 * my * mz +
               -f010 * fy_frac * mz + f110 * fy_frac * mz +
               -f001 * my * fz_frac + f101 * my * fz_frac +
               -f011 * fy_frac * fz_frac + f111 * fy_frac * fz_frac;

    float gy = f000 * mx * (-1) * mz + f100 * fx_frac * (-1) * mz +
               f010 * mx * mz + f110 * fx_frac * mz +
               f001 * mx * (-1) * fz_frac + f101 * fx_frac * (-1) * fz_frac +
               f011 * mx * fz_frac + f111 * fx_frac * fz_frac;

    float gz = f000 * mx * my * (-1) + f100 * fx_frac * my * (-1) +
               f010 * mx * fy_frac * (-1) + f110 * fx_frac * fy_frac * (-1) +
               f001 * mx * my + f101 * fx_frac * my +
               f011 * mx * fy_frac + f111 * fx_frac * fy_frac;

    // Apply soft curl (matches curl.h in standard gnina)
    // Smoothly scales energy and gradient for positive energies
    if (f > 0) {
        float tmp = v / (v + f);
        f *= tmp;
        float tmp_sq = tmp * tmp;
        gx *= tmp_sq;
        gy *= tmp_sq;
        gz *= tmp_sq;
    }

    // Scale gradient by factor and add penalty gradient
    *energy = f + penalty;
    *fx = (region_x == 0 ? grid.m_factor.x * gx : 0) + slope * region_x;
    *fy = (region_y == 0 ? grid.m_factor.y * gy : 0) + slope * region_y;
    *fz = (region_z == 0 ? grid.m_factor.z * gz : 0) + slope * region_z;
}

// Wrapper for main grid data
__device__ inline void trilinear_interp_device(
    const grid_gpu& grid,
    float x, float y, float z,
    float slope,
    float v,
    float* energy,
    float* fx, float* fy, float* fz
) {
    trilinear_interp_core(grid, grid.data, x, y, z, slope, v, energy, fx, fy, fz);
}

// Wrapper for charge data
__device__ inline void trilinear_interp_charge(
    const grid_gpu& grid,
    float x, float y, float z,
    float slope,
    float v,
    float* energy,
    float* fx, float* fy, float* fz
) {
    trilinear_interp_core(grid, grid.chargedata, x, y, z, slope, v, energy, fx, fy, fz);
}

__device__ float eval_energy_grid_single_thread(
    const ScoringContext& ctx,
    const float* coords,
    float* forces,
    bool debug_print = false
) {
    float total_energy = 0;

    // Zero forces
    for (unsigned i = 0; i < ctx.num_atoms * 3; i++) {
        forces[i] = 0;
    }

    // Early exit if no grids or types
    if (ctx.grids == nullptr || ctx.atom_types == nullptr || ctx.num_atoms == 0) {
        return total_energy;
    }

    // Evaluate each atom against grid
    for (unsigned i = 0; i < ctx.num_atoms; i++) {
        unsigned atom_type = ctx.atom_types[i];

        // Skip hydrogens (type <= 1) and out-of-bounds types
        if (atom_type <= 1 || atom_type >= ctx.ngrids) continue;

        float x = coords[i * 3 + 0];
        float y = coords[i * 3 + 1];
        float z = coords[i * 3 + 2];

        const grid_gpu& grid = ctx.grids[atom_type];

        float e, fx, fy, fz;
        trilinear_interp_device(grid, x, y, z, ctx.slope, ctx.forcecap, &e, &fx, &fy, &fz);

        if (debug_print) {
            printf("GPU atom %u: type=%u pos=(%.3f,%.3f,%.3f) e=%.6f\n", i, atom_type, x, y, z, e);
        }

        total_energy += e;
        forces[i * 3 + 0] = fx;
        forces[i * 3 + 1] = fy;
        forces[i * 3 + 2] = fz;

        // Handle charge-dependent grid if present
        if (ctx.atom_params_data != nullptr && grid.chargedata.dim0() > 0) {
            float charge = ctx.atom_params_data[i].charge;
            if (charge != 0) {
                float ce, cfx, cfy, cfz;
                trilinear_interp_charge(grid, x, y, z, ctx.slope, ctx.forcecap, &ce, &cfx, &cfy, &cfz);
                total_energy += charge * ce;
                forces[i * 3 + 0] += charge * cfx;
                forces[i * 3 + 1] += charge * cfy;
                forces[i * 3 + 2] += charge * cfz;
            }
        }
    }

    return total_energy;
}

// ============================================================================
// Single-Thread Intramolecular Energy
// ============================================================================

// Helper: evaluate a single spline at distance r
// Returns energy value and sets deriv to derivative
__device__ inline float evaluate_spline_device(
    float* spline,
    float r,
    float fraction,
    float cutoff,
    float& deriv
) {
    if (r >= cutoff || r < 0) {
        deriv = 0;
        return 0;
    }

    unsigned index = (unsigned)(r / fraction);
    unsigned base = 5 * index;

    float x = spline[base];
    float a = spline[base + 1];
    float b = spline[base + 2];
    float c = spline[base + 3];
    float d = spline[base + 4];

    float lx = r - x;
    float val = ((a * lx + b) * lx + c) * lx + d;
    deriv = (3 * a * lx + 2 * b) * lx + c;

    return val;
}

// Evaluate intramolecular pair interaction with charge-dependent components
// Matches gpucode.cu eval_deriv_gpu()
__device__ inline float eval_pair_deriv_gpu(
    const GPUSplineInfo* splineInfo,
    unsigned t,       // atom A type
    float charge,     // atom A charge
    unsigned rt,      // atom B type
    float rcharge,    // atom B charge
    float r2,         // distance squared
    float& dor        // output: derivative/r for force computation
) {
    float r = sqrtf(r2);

    // Sort types so t1 <= t2, and sort charges accordingly
    unsigned t1, t2;
    float charge1, charge2;
    if (t < rt) {
        t1 = t;
        t2 = rt;
        charge1 = fabsf(charge);
        charge2 = fabsf(rcharge);
    } else {
        t1 = rt;
        t2 = t;
        charge1 = fabsf(rcharge);
        charge2 = fabsf(charge);
    }

    // Symmetric indexing for spline lookup
    unsigned tindex = t1 + t2 * (t2 + 1) / 2;
    const GPUSplineInfo& spInfo = splineInfo[tindex];
    unsigned n = spInfo.n;  // number of charge-dependent components

    float ret = 0, d = 0;

    // Evaluate up to 4 charge-dependent spline components
    // (matches gpucode.cu eval_deriv_gpu)
    if (n > 0) {
        float fraction = spInfo.fraction;
        float cutoff = spInfo.cutoff;
        float val, deriv;

        // Component 0: TypeDependentOnly (no charge adjustment)
        val = evaluate_spline_device(spInfo.splines[0], r, fraction, cutoff, deriv);
        ret += val;
        d += deriv;

        // Component 1: AbsAChargeDependent (multiply by abs(chargeA))
        if (n > 1) {
            val = evaluate_spline_device(spInfo.splines[1], r, fraction, cutoff, deriv);
            ret += val * charge1;
            d += deriv * charge1;

            // Component 2: AbsBChargeDependent (multiply by abs(chargeB))
            if (n > 2) {
                val = evaluate_spline_device(spInfo.splines[2], r, fraction, cutoff, deriv);
                ret += val * charge2;
                d += deriv * charge2;

                // Component 3: ABChargeDependent (multiply by chargeA * chargeB)
                if (n > 3) {
                    val = evaluate_spline_device(spInfo.splines[3], r, fraction, cutoff, deriv);
                    ret += val * charge2 * charge1;
                    d += deriv * charge2 * charge1;
                }
            }
        }
    }

    dor = d / r;  // Divide by distance to normalize for force computation
    return ret;
}

__device__ float eval_intramolecular_single_thread(
    const ScoringContext& ctx,
    const float* coords,
    float* forces
) {
    float total_energy = 0;

    // Early exit if no pairs or spline info
    if (ctx.pairs == nullptr || ctx.splineInfo == nullptr || ctx.num_pairs == 0) {
        return total_energy;
    }

    // Curl parameter from forcecap (matches standard gnina)
    const float v = ctx.forcecap;

    for (unsigned i = 0; i < ctx.num_pairs; i++) {
        const interacting_pair& ip = ctx.pairs[i];

        // Get atom positions
        float r_x = coords[ip.b * 3 + 0] - coords[ip.a * 3 + 0];
        float r_y = coords[ip.b * 3 + 1] - coords[ip.a * 3 + 1];
        float r_z = coords[ip.b * 3 + 2] - coords[ip.a * 3 + 2];

        float r2 = r_x*r_x + r_y*r_y + r_z*r_z;

        if (r2 >= ctx.cutoff_sq) continue;

        // Get atom types and charges
        unsigned t1 = ip.t1;
        unsigned t2 = ip.t2;
        float charge_a = (ctx.atom_params_data != nullptr) ? ctx.atom_params_data[ip.a].charge : 0;
        float charge_b = (ctx.atom_params_data != nullptr) ? ctx.atom_params_data[ip.b].charge : 0;

        // Evaluate pair interaction with all charge-dependent components
        float dor;
        float energy = eval_pair_deriv_gpu(ctx.splineInfo, t1, charge_a, t2, charge_b, r2, dor);

        // Compute derivative vector
        float deriv_x = r_x * dor;
        float deriv_y = r_y * dor;
        float deriv_z = r_z * dor;

        // Apply soft curl (matches curl.h in standard gnina)
        if (energy > 0) {
            float tmp = v / (v + energy);
            energy *= tmp;
            float tmp_sq = tmp * tmp;
            deriv_x *= tmp_sq;
            deriv_y *= tmp_sq;
            deriv_z *= tmp_sq;
        }

        total_energy += energy;

        // Add forces (deriv points from a to b)
        forces[ip.b * 3 + 0] += deriv_x;
        forces[ip.b * 3 + 1] += deriv_y;
        forces[ip.b * 3 + 2] += deriv_z;

        forces[ip.a * 3 + 0] -= deriv_x;
        forces[ip.a * 3 + 1] -= deriv_y;
        forces[ip.a * 3 + 2] -= deriv_z;
    }

    return total_energy;
}

// ============================================================================
// Single-Thread Gradient Computation
// ============================================================================

// Matches tree_gpu.cu _derivative()
// Computes gradient from Cartesian forces using the tree structure
__device__ void compute_gradient_single_thread(
    const ScoringContext& ctx,
    const float* coords,        // Cartesian atom coordinates [num_atoms × 3]
    const float* forces,        // Cartesian atom forces [num_atoms × 3]
    const float* node_origins,  // Node origins from set_conf [num_nodes × 3]
    const float* node_axes,     // Transformed axes from set_conf [num_nodes × 3]
    float* node_forces,         // Scratch: [num_nodes × 3]
    float* node_torques,        // Scratch: [num_nodes × 3]
    float* gradient             // Output: [n_change]
) {
    // Initialize node forces and torques to zero
    for (unsigned i = 0; i < ctx.num_nodes * 3; i++) {
        node_forces[i] = 0;
        node_torques[i] = 0;
    }

    // Step 1: Accumulate atom forces AND torques to their owner nodes
    // Matches tree_gpu.cu lines 283-285:
    //   pseudoAtomicAdd(&force_torques[nid].first, forces[tid]);
    //   pseudoAtomicAdd(&force_torques[nid].second,
    //       cross_product(coords[tid] - owner.origin, forces[tid]));
    for (unsigned atom = 0; atom < ctx.num_atoms; atom++) {
        unsigned owner = ctx.atom_owners[atom];

        float fx = forces[atom * 3 + 0];
        float fy = forces[atom * 3 + 1];
        float fz = forces[atom * 3 + 2];

        // Add force to owner node
        node_forces[owner * 3 + 0] += fx;
        node_forces[owner * 3 + 1] += fy;
        node_forces[owner * 3 + 2] += fz;

        // Compute r = atom_pos - node_origin
        float rx = coords[atom * 3 + 0] - node_origins[owner * 3 + 0];
        float ry = coords[atom * 3 + 1] - node_origins[owner * 3 + 1];
        float rz = coords[atom * 3 + 2] - node_origins[owner * 3 + 2];

        // Add torque = r × F to owner node
        node_torques[owner * 3 + 0] += ry * fz - rz * fy;
        node_torques[owner * 3 + 1] += rz * fx - rx * fz;
        node_torques[owner * 3 + 2] += rx * fy - ry * fx;
    }

    // Step 2: Propagate from leaves to root (reverse BFS)
    // Matches tree_gpu.cu lines 304-312:
    //   pseudoAtomicAdd(&force_torques[parent_id].first, ft.first);
    //   pseudoAtomicAdd(&force_torques[parent_id].second,
    //       cross_product(r, ft.first) + ft.second);
    for (int layer = ctx.num_layers - 1; layer > 0; layer--) {
        for (unsigned nid = ctx.nlig_roots; nid < ctx.num_nodes; nid++) {
            const segment_node& node = ctx.tree_nodes[nid];
            if (node.layer != layer) continue;

            int parent = node.parent;
            if (parent < 0) continue;

            float child_fx = node_forces[nid * 3 + 0];
            float child_fy = node_forces[nid * 3 + 1];
            float child_fz = node_forces[nid * 3 + 2];

            float child_tx = node_torques[nid * 3 + 0];
            float child_ty = node_torques[nid * 3 + 1];
            float child_tz = node_torques[nid * 3 + 2];

            // r = child_origin - parent_origin
            float rx = node_origins[nid * 3 + 0] - node_origins[parent * 3 + 0];
            float ry = node_origins[nid * 3 + 1] - node_origins[parent * 3 + 1];
            float rz = node_origins[nid * 3 + 2] - node_origins[parent * 3 + 2];

            // parent_force += child_force
            node_forces[parent * 3 + 0] += child_fx;
            node_forces[parent * 3 + 1] += child_fy;
            node_forces[parent * 3 + 2] += child_fz;

            // parent_torque += cross(r, child_force) + child_torque
            node_torques[parent * 3 + 0] += (ry * child_fz - rz * child_fy) + child_tx;
            node_torques[parent * 3 + 1] += (rz * child_fx - rx * child_fz) + child_ty;
            node_torques[parent * 3 + 2] += (rx * child_fy - ry * child_fx) + child_tz;
        }
    }

    // Step 3: Extract gradient components
    // Rigid body roots: 3 position + 3 orientation
    for (unsigned i = 0; i < ctx.nlig_roots; i++) {
        gradient[i * 6 + 0] = node_forces[i * 3 + 0];
        gradient[i * 6 + 1] = node_forces[i * 3 + 1];
        gradient[i * 6 + 2] = node_forces[i * 3 + 2];
        gradient[i * 6 + 3] = node_torques[i * 3 + 0];
        gradient[i * 6 + 4] = node_torques[i * 3 + 1];
        gradient[i * 6 + 5] = node_torques[i * 3 + 2];
    }

    // Torsion nodes: dot(torque, axis)
    // Uses transformed axis from node_axes (computed by set_conf_single_thread)
    for (unsigned nid = ctx.nlig_roots; nid < ctx.num_nodes; nid++) {
        const float* axis = &node_axes[nid * 3];
        const float* torque = &node_torques[nid * 3];

        gradient[nid + 5 * ctx.nlig_roots] = dot3(torque, axis);
    }
}

// ============================================================================
// Single-Thread BFGS Components
// ============================================================================

// Helper: increment configuration by alpha * p
__device__ void increment_conf_single_thread(
    const ScoringContext& ctx,
    const float* x,
    float* x_new,
    const float* p,
    float alpha
) {
    // Copy and increment position
    for (int i = 0; i < ctx.n_conf; i++) {
        x_new[i] = x[i];
    }

    // Increment rigid body positions and orientations
    for (unsigned i = 0; i < ctx.nlig_roots; i++) {
        for (int j = 0; j < 3; j++) {
            x_new[i * 7 + j] += alpha * p[i * 6 + j];
        }
        // Increment orientation via quaternion
        float rot[3] = {alpha * p[i * 6 + 3],
                       alpha * p[i * 6 + 4],
                       alpha * p[i * 6 + 5]};
        qt dq = angle_to_quaternion_device(rot[0], rot[1], rot[2]);
        qt q(x_new[i * 7 + 3], x_new[i * 7 + 4],
             x_new[i * 7 + 5], x_new[i * 7 + 6]);
        qt qnew = quat_mult(dq, q);
        // Normalize
        float qn = sqrtf(qnew.R_component_1()*qnew.R_component_1() +
                        qnew.R_component_2()*qnew.R_component_2() +
                        qnew.R_component_3()*qnew.R_component_3() +
                        qnew.R_component_4()*qnew.R_component_4());
        if (qn > epsilon_fl) {
            x_new[i * 7 + 3] = qnew.R_component_1() / qn;
            x_new[i * 7 + 4] = qnew.R_component_2() / qn;
            x_new[i * 7 + 5] = qnew.R_component_3() / qn;
            x_new[i * 7 + 6] = qnew.R_component_4() / qn;
        }
    }

    // Increment torsions
    for (unsigned i = ctx.nlig_roots; i < ctx.num_nodes; i++) {
        float dt = alpha * p[i + 5 * ctx.nlig_roots];
        x_new[i + 6 * ctx.nlig_roots] += normalize_angle_device(dt);
        x_new[i + 6 * ctx.nlig_roots] = normalize_angle_device(x_new[i + 6 * ctx.nlig_roots]);
    }
}

// Helper: evaluate energy at configuration
__device__ float eval_conf_single_thread(
    const ScoringContext& ctx,
    const float* conf,
    float* coords,
    float* forces,
    float* node_origins,
    float* node_orientations,
    float* node_axes
) {
    set_conf_single_thread(ctx, conf, coords, node_origins, node_orientations, node_axes);
    float e = eval_energy_grid_single_thread(ctx, coords, forces);
    e += eval_intramolecular_single_thread(ctx, coords, forces);
    return e;
}

// Accurate line search (matches bfgs.h accurate_line_search)
// Based on Numerical Recipes lnsrch with quadratic/cubic interpolation
// Returns alpha=0 on failure (not a descent direction or step too small)
__device__ float accurate_line_search_single_thread(
    const ScoringContext& ctx,
    float* x,
    float* x_new,
    const float* g,
    const float* p,
    float f0,
    float* coords,
    float* forces,
    float* node_forces,
    float* node_torques,
    float* g_new,
    float& f_new
) {
    const float ALF = 1.0e-4f;  // Armijo coefficient

    // Scratch for node state
    float node_origins[PARALLEL_MAX_TORSIONS * 3];
    float node_orientations[PARALLEL_MAX_TORSIONS * 9];
    float node_axes[PARALLEL_MAX_TORSIONS * 3];

    // Compute slope = g · p
    float slope = 0;
    for (int i = 0; i < ctx.n_change; i++) {
        slope += g[i] * p[i];
    }

    // Check if p is a descent direction
    if (slope >= 0) {
        // Not a descent direction - copy x to x_new and zero gradient
        for (int i = 0; i < ctx.n_conf; i++) {
            x_new[i] = x[i];
        }
        for (int i = 0; i < ctx.n_change; i++) {
            g_new[i] = 0;
        }
        f_new = f0;
        return 0;  // Signal failure
    }

    // Compute minimum step size (lambdamin)
    // test = max_i(|p[i]| / max(|x[i]|, 1))
    float test = 0;
    for (int i = 0; i < ctx.n_change; i++) {
        // For change vector, we need corresponding x values
        // This is approximate - use 1.0 as reference scale
        float temp = fabsf(p[i]);  // Simplified: assume x scale ~ 1
        if (temp > test) test = temp;
    }
    float alamin = (test > 0) ? (epsilon_fl / test) : epsilon_fl;

    float alpha = 1.0f;  // Start with full Newton step
    float alpha2 = 0, f2 = 0;
    bool first_backtrack = true;

    for (;;) {
        // Check for too small step
        if (alpha < alamin || !isfinite(alpha)) {
            // Step too small - copy x to x_new and zero gradient
            for (int i = 0; i < ctx.n_conf; i++) {
                x_new[i] = x[i];
            }
            for (int i = 0; i < ctx.n_change; i++) {
                g_new[i] = 0;
            }
            f_new = f0;
            return 0;  // Signal failure
        }

        // Try step: x_new = x + alpha * p
        increment_conf_single_thread(ctx, x, x_new, p, alpha);

        // Evaluate at new point
        f_new = eval_conf_single_thread(ctx, x_new, coords, forces,
                                        node_origins, node_orientations, node_axes);

        // Check Armijo sufficient decrease condition
        if (f_new <= f0 + ALF * alpha * slope) {
            // Success - compute gradient and return
            compute_gradient_single_thread(ctx, coords, forces, node_origins,
                                          node_axes, node_forces, node_torques, g_new);
            return alpha;
        }

        // Need to backtrack - use interpolation
        float tmplam;
        if (first_backtrack) {
            // First backtrack: quadratic interpolation
            // tmplam = -slope / (2 * (f_new - f0 - slope))
            float denom = 2.0f * (f_new - f0 - slope);
            if (fabsf(denom) > epsilon_fl) {
                tmplam = -slope / denom;
            } else {
                tmplam = 0.5f * alpha;
            }
            first_backtrack = false;
        } else {
            // Subsequent backtracks: cubic interpolation
            float rhs1 = f_new - f0 - alpha * slope;
            float rhs2 = f2 - f0 - alpha2 * slope;
            float alpha_diff = alpha - alpha2;

            if (fabsf(alpha_diff) < epsilon_fl) {
                tmplam = 0.5f * alpha;
            } else {
                float a = (rhs1 / (alpha * alpha) - rhs2 / (alpha2 * alpha2)) / alpha_diff;
                float b = (-alpha2 * rhs1 / (alpha * alpha) +
                           alpha * rhs2 / (alpha2 * alpha2)) / alpha_diff;

                if (fabsf(a) < epsilon_fl) {
                    // Quadratic case
                    tmplam = -slope / (2.0f * b);
                } else {
                    float disc = b * b - 3.0f * a * slope;
                    if (disc < 0) {
                        tmplam = 0.5f * alpha;
                    } else if (b <= 0) {
                        tmplam = (-b + sqrtf(disc)) / (3.0f * a);
                    } else {
                        tmplam = -slope / (b + sqrtf(disc));
                    }
                }

                // Always at least cut in half
                if (tmplam > 0.5f * alpha) {
                    tmplam = 0.5f * alpha;
                }
            }
        }

        // Save current values for next cubic interpolation
        alpha2 = alpha;
        f2 = f_new;

        // Update alpha, but never smaller than a tenth of previous
        alpha = fmaxf(tmplam, 0.1f * alpha);
    }
}

// BFGS Hessian update
__device__ void bfgs_hessian_update_single_thread(
    float* h,
    const float* p,
    const float* y,
    float alpha,
    int n
) {
    // Compute y · p
    float yp = 0;
    for (int i = 0; i < n; i++) {
        yp += y[i] * p[i];
    }

    if (alpha * yp < epsilon_fl) return;  // Skip update

    // Compute -H * y (minus_hy), same as CPU version
    float minus_hy[PARALLEL_MAX_CHANGE_SIZE];
    for (int i = 0; i < n; i++) {
        minus_hy[i] = 0;
        for (int j = 0; j < n; j++) {
            // Triangular indexing
            int idx = (i <= j) ? (i + j * (j + 1) / 2) : (j + i * (i + 1) / 2);
            minus_hy[i] -= h[idx] * y[j];  // Note: negation to match CPU
        }
    }

    // Compute y · H · y = -y · minus_hy
    float yhy = 0;
    for (int i = 0; i < n; i++) {
        yhy -= y[i] * minus_hy[i];  // Note: negation since minus_hy = -H*y
    }

    float r = 1.0f / (alpha * yp);
    float coef = alpha * alpha * (r * r * yhy + r);

    // Update H (matching CPU formula exactly)
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int idx = i + j * (j + 1) / 2;
            h[idx] += alpha * r * (minus_hy[i] * p[j] + minus_hy[j] * p[i]) + coef * p[i] * p[j];
        }
    }
}

// ============================================================================
// Main Parallel BFGS Kernel
// ============================================================================

__global__ void bfgs_parallel_kernel(
    BFGSState* states,
    const float* initial_confs,
    const ScoringContext* contexts,  // One per ligand type
    const int* optimizer_to_ligand,
    BFGSBatchMemory mem,
    int max_iterations,
    int n_optimizers
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_optimizers) return;

    // Get ligand for this optimizer
    int lig_id = optimizer_to_ligand[tid];
    const ScoringContext& ctx = contexts[lig_id];

    // Set up pointers into batch memory
    BFGSState state;
    state.x = &mem.all_x[tid * mem.max_conf_size];
    state.x_new = &mem.all_x_new[tid * mem.max_conf_size];
    state.g = &mem.all_g[tid * mem.max_change_size];
    state.g_new = &mem.all_g_new[tid * mem.max_change_size];
    state.p = &mem.all_p[tid * mem.max_change_size];
    state.y = &mem.all_y[tid * mem.max_change_size];
    state.h = &mem.all_h[tid * mem.max_hessian_size];
    state.coords = &mem.all_coords[tid * mem.max_atoms * 3];
    state.forces = &mem.all_forces[tid * mem.max_atoms * 3];
    state.node_forces = &mem.all_node_forces[tid * mem.max_nodes * 3];
    state.node_torques = &mem.all_node_torques[tid * mem.max_nodes * 3];
    state.ligand_id = lig_id;

    // Initialize from input conf
    for (int i = 0; i < ctx.n_conf; i++) {
        state.x[i] = initial_confs[tid * mem.max_conf_size + i];
    }

    // Scratch for node state during set_conf
    float node_origins[PARALLEL_MAX_TORSIONS * 3];
    float node_orientations[PARALLEL_MAX_TORSIONS * 9];
    float node_axes[PARALLEL_MAX_TORSIONS * 3];

    // Initial energy and gradient
    set_conf_single_thread(ctx, state.x, state.coords, node_origins, node_orientations, node_axes);
    state.energy = eval_energy_grid_single_thread(ctx, state.coords, state.forces);
    state.energy += eval_intramolecular_single_thread(ctx, state.coords, state.forces);
    compute_gradient_single_thread(ctx, state.coords, state.forces, node_origins,
                                   node_axes, state.node_forces, state.node_torques, state.g);
    state.best_energy = state.energy;

    // Save best conf
    for (int i = 0; i < ctx.n_conf; i++) {
        mem.all_best_confs[tid * mem.max_conf_size + i] = state.x[i];
    }

    // Initialize Hessian to identity
    int hess_size = ctx.n_change * (ctx.n_change + 1) / 2;
    for (int i = 0; i < hess_size; i++) {
        state.h[i] = 0;
    }
    for (int i = 0; i < ctx.n_change; i++) {
        int idx = i + i * (i + 1) / 2;
        state.h[idx] = 1.0f;
    }

    // BFGS iterations (with early stopping on line search failure, matching bfgs.h)
    for (int iter = 0; iter < max_iterations; iter++) {
        // Compute search direction: p = -H * g
        for (int i = 0; i < ctx.n_change; i++) {
            state.p[i] = 0;
            for (int j = 0; j < ctx.n_change; j++) {
                int idx = (i <= j) ? (i + j * (j + 1) / 2) : (j + i * (i + 1) / 2);
                state.p[i] += state.h[idx] * state.g[j];
            }
            state.p[i] = -state.p[i];
        }

        // Line search (accurate, matching bfgs.h default)
        float f_new;
        float alpha = accurate_line_search_single_thread(
            ctx, state.x, state.x_new, state.g, state.p,
            state.energy, state.coords, state.forces,
            state.node_forces, state.node_torques, state.g_new, f_new
        );

        // Check for line search failure (matches bfgs.h behavior)
        if (alpha == 0) {
            break;  // Line direction was wrong or step too small, give up
        }

        // y = g_new - g
        for (int i = 0; i < ctx.n_change; i++) {
            state.y[i] = state.g_new[i] - state.g[i];
        }

        // Shanno-Phua scaling on first iteration (matches bfgs.h behavior)
        if (iter == 0) {
            float yy = 0;
            float yp = 0;
            for (int i = 0; i < ctx.n_change; i++) {
                yy += state.y[i] * state.y[i];
                yp += state.y[i] * state.p[i];
            }
            if (yy > epsilon_fl) {
                float scale = alpha * yp / yy;
                // Reset Hessian diagonal to scaled identity
                int hess_size = ctx.n_change * (ctx.n_change + 1) / 2;
                for (int i = 0; i < hess_size; i++) {
                    state.h[i] = 0;
                }
                for (int i = 0; i < ctx.n_change; i++) {
                    int idx = i + i * (i + 1) / 2;
                    state.h[idx] = scale;
                }
            }
        }

        // BFGS Hessian update
        bfgs_hessian_update_single_thread(state.h, state.p, state.y, alpha, ctx.n_change);

        // Accept step
        for (int i = 0; i < ctx.n_conf; i++) {
            state.x[i] = state.x_new[i];
        }
        for (int i = 0; i < ctx.n_change; i++) {
            state.g[i] = state.g_new[i];
        }
        state.energy = f_new;

        // Track best
        if (state.energy < state.best_energy) {
            state.best_energy = state.energy;
            for (int i = 0; i < ctx.n_conf; i++) {
                mem.all_best_confs[tid * mem.max_conf_size + i] = state.x[i];
            }
        }
    }

    // Store final results
    mem.all_energies[tid] = state.energy;
    mem.all_best_energies[tid] = state.best_energy;
}

// ============================================================================
// Random Conformation Generation Kernel
// ============================================================================

__global__ void generate_random_confs_kernel(
    float* confs,
    const ScoringContext* contexts,
    const int* optimizer_to_ligand,
    int max_conf_size,
    int n_optimizers,
    float3 box_min,
    float3 box_max,
    unsigned int seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_optimizers) return;

    // Initialize RNG
    curandState rng;
    curand_init(seed, tid, 0, &rng);

    int lig_id = optimizer_to_ligand[tid];
    const ScoringContext& ctx = contexts[lig_id];

    float* conf = &confs[tid * max_conf_size];

    // Random position in box
    for (unsigned i = 0; i < ctx.nlig_roots; i++) {
        conf[i * 7 + 0] = box_min.x + curand_uniform(&rng) * (box_max.x - box_min.x);
        conf[i * 7 + 1] = box_min.y + curand_uniform(&rng) * (box_max.y - box_min.y);
        conf[i * 7 + 2] = box_min.z + curand_uniform(&rng) * (box_max.z - box_min.z);

        // Random quaternion (normalized)
        float u1 = curand_uniform(&rng);
        float u2 = curand_uniform(&rng) * 2.0f * 3.14159265f;
        float u3 = curand_uniform(&rng) * 2.0f * 3.14159265f;

        float sq1 = sqrtf(1 - u1);
        float sq2 = sqrtf(u1);
        conf[i * 7 + 3] = sq1 * sinf(u2);
        conf[i * 7 + 4] = sq1 * cosf(u2);
        conf[i * 7 + 5] = sq2 * sinf(u3);
        conf[i * 7 + 6] = sq2 * cosf(u3);
    }

    // Random torsions [-π, π]
    for (unsigned i = ctx.nlig_roots; i < ctx.num_nodes; i++) {
        float torsion = (curand_uniform(&rng) * 2.0f - 1.0f) * 3.14159265f;
        conf[i + 6 * ctx.nlig_roots] = torsion;
    }
}

// ============================================================================
// Host Functions
// ============================================================================

void allocate_batch_memory(
    BFGSBatchMemory& mem,
    int n_optimizers,
    int max_conf_size,
    int max_change_size,
    int max_atoms,
    int max_nodes
) {
    mem.n_optimizers = n_optimizers;
    mem.max_conf_size = max_conf_size;
    mem.max_change_size = max_change_size;
    mem.max_hessian_size = max_change_size * (max_change_size + 1) / 2;
    mem.max_atoms = max_atoms;
    mem.max_nodes = max_nodes;

    size_t total = 0;

    // Allocate all arrays
    size_t conf_bytes = n_optimizers * max_conf_size * sizeof(float);
    size_t change_bytes = n_optimizers * max_change_size * sizeof(float);
    size_t hess_bytes = n_optimizers * mem.max_hessian_size * sizeof(float);
    size_t coords_bytes = n_optimizers * max_atoms * 3 * sizeof(float);
    size_t node_bytes = n_optimizers * max_nodes * 3 * sizeof(float);

    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_x, conf_bytes));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_x_new, conf_bytes));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_g, change_bytes));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_g_new, change_bytes));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_p, change_bytes));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_y, change_bytes));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_h, hess_bytes));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_coords, coords_bytes));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_forces, coords_bytes));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_node_forces, node_bytes));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_node_torques, node_bytes));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_energies, n_optimizers * sizeof(float)));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_best_energies, n_optimizers * sizeof(float)));
    CUDA_CHECK_GNINA(cudaMalloc(&mem.all_best_confs, conf_bytes));

    total = 2 * conf_bytes + 4 * change_bytes + hess_bytes +
            2 * coords_bytes + 2 * node_bytes +
            2 * n_optimizers * sizeof(float) + conf_bytes;

    mem.total_bytes = total;
}

void free_batch_memory(BFGSBatchMemory& mem) {
    if (mem.all_x) cudaFree(mem.all_x);
    if (mem.all_x_new) cudaFree(mem.all_x_new);
    if (mem.all_g) cudaFree(mem.all_g);
    if (mem.all_g_new) cudaFree(mem.all_g_new);
    if (mem.all_p) cudaFree(mem.all_p);
    if (mem.all_y) cudaFree(mem.all_y);
    if (mem.all_h) cudaFree(mem.all_h);
    if (mem.all_coords) cudaFree(mem.all_coords);
    if (mem.all_forces) cudaFree(mem.all_forces);
    if (mem.all_node_forces) cudaFree(mem.all_node_forces);
    if (mem.all_node_torques) cudaFree(mem.all_node_torques);
    if (mem.all_energies) cudaFree(mem.all_energies);
    if (mem.all_best_energies) cudaFree(mem.all_best_energies);
    if (mem.all_best_confs) cudaFree(mem.all_best_confs);

    memset(&mem, 0, sizeof(mem));
}

void launch_parallel_bfgs(
    const LigandBatch& batch,
    BFGSBatchMemory& mem,
    int max_iterations,
    const gfloat3& box_min,
    const gfloat3& box_max,
    unsigned int random_seed
) {
    int threads_per_block = 128;
    int num_blocks = (batch.total_optimizers + threads_per_block - 1) / threads_per_block;

    // Allocate device memory for initial confs
    float* d_initial_confs;
    CUDA_CHECK_GNINA(cudaMalloc(&d_initial_confs,
        batch.total_optimizers * mem.max_conf_size * sizeof(float)));

    // Generate random initial conformations
    generate_random_confs_kernel<<<num_blocks, threads_per_block>>>(
        d_initial_confs,
        batch.ligand_contexts,
        batch.optimizer_to_ligand,
        mem.max_conf_size,
        batch.total_optimizers,
        make_float3(box_min.x, box_min.y, box_min.z),
        make_float3(box_max.x, box_max.y, box_max.z),
        random_seed
    );

    // Run BFGS
    bfgs_parallel_kernel<<<num_blocks, threads_per_block>>>(
        nullptr,  // states are embedded in mem
        d_initial_confs,
        batch.ligand_contexts,
        batch.optimizer_to_ligand,
        mem,
        max_iterations,
        batch.total_optimizers
    );

    cudaFree(d_initial_confs);
}

void collect_bfgs_results(
    const BFGSBatchMemory& mem,
    int n_optimizers,
    std::vector<float>& energies,
    std::vector<std::vector<float>>& conformations
) {
    energies.resize(n_optimizers);
    conformations.resize(n_optimizers);

    // Copy energies
    CUDA_CHECK_GNINA(cudaMemcpy(energies.data(), mem.all_best_energies,
        n_optimizers * sizeof(float), cudaMemcpyDeviceToHost));

    // Copy conformations
    std::vector<float> all_confs(n_optimizers * mem.max_conf_size);
    CUDA_CHECK_GNINA(cudaMemcpy(all_confs.data(), mem.all_best_confs,
        n_optimizers * mem.max_conf_size * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_optimizers; i++) {
        conformations[i].assign(
            all_confs.begin() + i * mem.max_conf_size,
            all_confs.begin() + (i + 1) * mem.max_conf_size
        );
    }
}

// ============================================================================
// Scoring Context Creation
// ============================================================================

void create_scoring_context(
    ScoringContext& ctx,
    const gpu_data& gdata,
    const GPUCacheInfo& cacheInfo
) {
    // Copy from cache info
    ctx.grids = cacheInfo.grids;
    ctx.ngrids = cacheInfo.ngrids;
    ctx.gridbegins = cacheInfo.gridbegins;
    ctx.gridends = cacheInfo.gridends;
    ctx.slope = cacheInfo.slope;
    ctx.cutoff_sq = cacheInfo.cutoff_sq;
    ctx.splineInfo = cacheInfo.splineInfo;
    ctx.num_atoms = cacheInfo.num_movable_atoms;
    ctx.atom_types = cacheInfo.types;
    ctx.forcecap = cacheInfo.forcecap;

    // From tree_gpu - need to copy from device to host first
    if (gdata.treegpu) {
        // Copy tree_gpu struct from device to host
        tree_gpu host_tree;
        CUDA_CHECK_GNINA(cudaMemcpy(&host_tree, gdata.treegpu, sizeof(tree_gpu), cudaMemcpyDeviceToHost));

        ctx.tree_nodes = host_tree.device_nodes;
        ctx.num_nodes = host_tree.num_nodes;
        ctx.num_layers = host_tree.num_layers;
        ctx.nlig_roots = host_tree.nlig_roots;
        ctx.atom_owners = host_tree.owners;
    } else {
        ctx.tree_nodes = nullptr;
        ctx.num_nodes = 0;
        ctx.num_layers = 0;
        ctx.nlig_roots = 1;
        ctx.atom_owners = nullptr;
    }

    // From gpu_data
    ctx.pairs = gdata.interacting_pairs;
    ctx.num_pairs = gdata.pairs_size;

    // Atom reference coordinates (local frame) - stored as marked_coord
    // gdata.atom_coords is vec* but actually contains marked_coord data
    ctx.atom_local_data = reinterpret_cast<const marked_coord*>(gdata.atom_coords);

    // Atom params (for charge access)
    ctx.atom_params_data = gdata.coords;

    // Compute dimensions
    ctx.n_torsions = ctx.num_nodes > ctx.nlig_roots ? ctx.num_nodes - ctx.nlig_roots : 0;
    ctx.n_conf = 7 * ctx.nlig_roots + ctx.n_torsions;
    ctx.n_change = 6 * ctx.nlig_roots + ctx.n_torsions;
}

// ============================================================================
// High-Level Interface for Main Integration
// ============================================================================

void run_parallel_bfgs_docking(
    const gpu_data& gdata,
    const GPUCacheInfo& cacheInfo,
    int n_poses,
    int max_iterations,
    const gfloat3& box_min,
    const gfloat3& box_max,
    unsigned int seed,
    std::vector<float>& out_energies,
    std::vector<std::vector<float>>& out_conformations
) {
    // Timing events
    cudaEvent_t start_total, end_setup, end_bfgs, end_collect;
    CUDA_CHECK_GNINA(cudaEventCreate(&start_total));
    CUDA_CHECK_GNINA(cudaEventCreate(&end_setup));
    CUDA_CHECK_GNINA(cudaEventCreate(&end_bfgs));
    CUDA_CHECK_GNINA(cudaEventCreate(&end_collect));

    CUDA_CHECK_GNINA(cudaEventRecord(start_total));

    // Create scoring context
    ScoringContext ctx;
    create_scoring_context(ctx, gdata, cacheInfo);

    // Allocate context on device
    ScoringContext* d_contexts;
    CUDA_CHECK_GNINA(cudaMalloc(&d_contexts, sizeof(ScoringContext)));
    CUDA_CHECK_GNINA(cudaMemcpy(d_contexts, &ctx, sizeof(ScoringContext), cudaMemcpyHostToDevice));

    // Set up single-ligand batch
    int* d_optimizer_to_ligand;
    CUDA_CHECK_GNINA(cudaMalloc(&d_optimizer_to_ligand, n_poses * sizeof(int)));
    CUDA_CHECK_GNINA(cudaMemset(d_optimizer_to_ligand, 0, n_poses * sizeof(int)));

    LigandBatch batch;
    batch.ligand_contexts = d_contexts;
    batch.optimizer_to_ligand = d_optimizer_to_ligand;
    batch.num_ligands = 1;
    batch.total_optimizers = n_poses;

    // Allocate batch memory
    BFGSBatchMemory mem;
    int max_conf = ctx.n_conf > 0 ? ctx.n_conf : PARALLEL_MAX_CONF_SIZE;
    int max_change = ctx.n_change > 0 ? ctx.n_change : PARALLEL_MAX_CHANGE_SIZE;
    int max_atoms = ctx.num_atoms > 0 ? ctx.num_atoms : PARALLEL_MAX_ATOMS;
    int max_nodes = ctx.num_nodes > 0 ? ctx.num_nodes : PARALLEL_MAX_TORSIONS;

    allocate_batch_memory(mem, n_poses, max_conf, max_change, max_atoms, max_nodes);

    CUDA_CHECK_GNINA(cudaEventRecord(end_setup));

    // Launch parallel BFGS
    launch_parallel_bfgs(batch, mem, max_iterations, box_min, box_max, seed);

    // Synchronize
    CUDA_CHECK_GNINA(cudaDeviceSynchronize());
    CUDA_CHECK_GNINA(cudaEventRecord(end_bfgs));

    // Collect results
    collect_bfgs_results(mem, n_poses, out_energies, out_conformations);

    CUDA_CHECK_GNINA(cudaEventRecord(end_collect));
    CUDA_CHECK_GNINA(cudaEventSynchronize(end_collect));

    // Calculate and print timing
    float setup_ms, bfgs_ms, collect_ms;
    CUDA_CHECK_GNINA(cudaEventElapsedTime(&setup_ms, start_total, end_setup));
    CUDA_CHECK_GNINA(cudaEventElapsedTime(&bfgs_ms, end_setup, end_bfgs));
    CUDA_CHECK_GNINA(cudaEventElapsedTime(&collect_ms, end_bfgs, end_collect));

    float total_ms = setup_ms + bfgs_ms + collect_ms;
    fprintf(stderr, "Parallel BFGS timing:\n");
    fprintf(stderr, "  Setup (context + alloc): %.2f ms (%.1f%%)\n", setup_ms, 100.0f * setup_ms / total_ms);
    fprintf(stderr, "  BFGS kernel:             %.2f ms (%.1f%%)\n", bfgs_ms, 100.0f * bfgs_ms / total_ms);
    fprintf(stderr, "  Result collection:       %.2f ms (%.1f%%)\n", collect_ms, 100.0f * collect_ms / total_ms);
    fprintf(stderr, "  Total:                   %.2f ms\n", total_ms);
    fprintf(stderr, "  Throughput:              %.1f poses/sec\n", 1000.0f * n_poses / total_ms);

    // Cleanup timing events
    cudaEventDestroy(start_total);
    cudaEventDestroy(end_setup);
    cudaEventDestroy(end_bfgs);
    cudaEventDestroy(end_collect);

    // Cleanup
    free_batch_memory(mem);
    cudaFree(d_contexts);
    cudaFree(d_optimizer_to_ligand);
}

// ============================================================================
// Conf <-> Flat Array Conversion Helpers
// ============================================================================

void conf_to_flat(
    const conf& c,
    unsigned nlig_roots,
    unsigned n_torsions,
    float* flat_conf
) {
    // Convert ligand rigid bodies
    for (unsigned i = 0; i < nlig_roots && i < c.ligands.size(); i++) {
        const rigid_conf& rc = c.ligands[i].rigid;
        // Position
        flat_conf[i * 7 + 0] = rc.position[0];
        flat_conf[i * 7 + 1] = rc.position[1];
        flat_conf[i * 7 + 2] = rc.position[2];
        // Quaternion (w, x, y, z)
        flat_conf[i * 7 + 3] = rc.orientation.R_component_1();
        flat_conf[i * 7 + 4] = rc.orientation.R_component_2();
        flat_conf[i * 7 + 5] = rc.orientation.R_component_3();
        flat_conf[i * 7 + 6] = rc.orientation.R_component_4();
    }

    // Convert torsions
    unsigned torsion_idx = 0;
    for (unsigned i = 0; i < c.ligands.size() && torsion_idx < n_torsions; i++) {
        for (unsigned j = 0; j < c.ligands[i].torsions.size() && torsion_idx < n_torsions; j++) {
            flat_conf[nlig_roots + 6 * nlig_roots + torsion_idx] = c.ligands[i].torsions[j];
            torsion_idx++;
        }
    }
}

void flat_to_conf(
    const float* flat_conf,
    unsigned nlig_roots,
    unsigned n_torsions,
    conf& c
) {
    // Convert ligand rigid bodies
    for (unsigned i = 0; i < nlig_roots && i < c.ligands.size(); i++) {
        rigid_conf& rc = c.ligands[i].rigid;
        // Position
        rc.position[0] = flat_conf[i * 7 + 0];
        rc.position[1] = flat_conf[i * 7 + 1];
        rc.position[2] = flat_conf[i * 7 + 2];
        // Quaternion
        rc.orientation = qt(
            flat_conf[i * 7 + 3],
            flat_conf[i * 7 + 4],
            flat_conf[i * 7 + 5],
            flat_conf[i * 7 + 6]
        );
    }

    // Convert torsions
    unsigned torsion_idx = 0;
    for (unsigned i = 0; i < c.ligands.size() && torsion_idx < n_torsions; i++) {
        for (unsigned j = 0; j < c.ligands[i].torsions.size() && torsion_idx < n_torsions; j++) {
            c.ligands[i].torsions[j] = flat_conf[nlig_roots + 6 * nlig_roots + torsion_idx];
            torsion_idx++;
        }
    }
}

// ============================================================================
// Single-Pose Minimize Kernel
// ============================================================================

// Kernel for minimizing a single pose (or N copies of the same pose)
__global__ void bfgs_minimize_kernel(
    const float* initial_conf,
    const ScoringContext* contexts,
    BFGSBatchMemory mem,
    int max_iterations,
    int n_optimizers
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_optimizers) return;

    // All optimizers use the same ligand (ligand 0)
    const ScoringContext& ctx = contexts[0];

    // Set up pointers into batch memory
    BFGSState state;
    state.x = &mem.all_x[tid * mem.max_conf_size];
    state.x_new = &mem.all_x_new[tid * mem.max_conf_size];
    state.g = &mem.all_g[tid * mem.max_change_size];
    state.g_new = &mem.all_g_new[tid * mem.max_change_size];
    state.p = &mem.all_p[tid * mem.max_change_size];
    state.y = &mem.all_y[tid * mem.max_change_size];
    state.h = &mem.all_h[tid * mem.max_hessian_size];
    state.coords = &mem.all_coords[tid * mem.max_atoms * 3];
    state.forces = &mem.all_forces[tid * mem.max_atoms * 3];
    state.node_forces = &mem.all_node_forces[tid * mem.max_nodes * 3];
    state.node_torques = &mem.all_node_torques[tid * mem.max_nodes * 3];
    state.ligand_id = 0;

    // Initialize from input conf (same for all threads in minimize mode)
    for (int i = 0; i < ctx.n_conf; i++) {
        state.x[i] = initial_conf[i];
    }

    // Scratch for node state during set_conf
    float node_origins[PARALLEL_MAX_TORSIONS * 3];
    float node_orientations[PARALLEL_MAX_TORSIONS * 9];
    float node_axes[PARALLEL_MAX_TORSIONS * 3];

    // Initial energy and gradient
    set_conf_single_thread(ctx, state.x, state.coords, node_origins, node_orientations, node_axes);
    float inter_energy = eval_energy_grid_single_thread(ctx, state.coords, state.forces, tid == 0);  // Debug on first thread
    float intra_energy = eval_intramolecular_single_thread(ctx, state.coords, state.forces);
    state.energy = inter_energy + intra_energy;
    compute_gradient_single_thread(ctx, state.coords, state.forces, node_origins,
                                   node_axes, state.node_forces, state.node_torques, state.g);
    state.best_energy = state.energy;

    // DEBUG: Print initial state
    if (tid == 0) {
        printf("GPU DEBUG: forcecap=%f, slope=%f\n", ctx.forcecap, ctx.slope);
        printf("GPU DEBUG: Initial energy=%f (inter=%f, intra=%f)\n", state.energy, inter_energy, intra_energy);
        printf("GPU DEBUG: Initial gradient (first 6 rigid): ");
        for (int i = 0; i < 6 && i < ctx.n_change; i++) {
            printf("%f ", state.g[i]);
        }
        printf("\n");
        if (ctx.n_change > 6) {
            printf("GPU DEBUG: Initial gradient (torsions): ");
            for (int i = 6; i < ctx.n_change; i++) {
                printf("%f ", state.g[i]);
            }
            printf("\n");
        }
        printf("GPU DEBUG: n_change=%d, n_conf=%d, num_atoms=%d, num_nodes=%d, ngrids=%d\n",
               ctx.n_change, ctx.n_conf, ctx.num_atoms, ctx.num_nodes, ctx.ngrids);

        // Print first 5 atom positions, types and per-atom energies
        printf("GPU DEBUG: First atoms (pos, type):\n");
        for (unsigned i = 0; i < 5 && i < ctx.num_atoms; i++) {
            unsigned atom_type = ctx.atom_types[i];
            printf("  Atom %d: pos=(%f,%f,%f), type=%d\n",
                   i, state.coords[i*3], state.coords[i*3+1], state.coords[i*3+2], atom_type);
        }
    }

    // Save best conf
    for (int i = 0; i < ctx.n_conf; i++) {
        mem.all_best_confs[tid * mem.max_conf_size + i] = state.x[i];
    }

    // Initialize Hessian to identity
    int hess_size = ctx.n_change * (ctx.n_change + 1) / 2;
    for (int i = 0; i < hess_size; i++) {
        state.h[i] = 0;
    }
    for (int i = 0; i < ctx.n_change; i++) {
        int idx = i + i * (i + 1) / 2;
        state.h[idx] = 1.0f;
    }

    // BFGS iterations
    for (int iter = 0; iter < max_iterations; iter++) {
        // Compute search direction: p = -H * g
        for (int i = 0; i < ctx.n_change; i++) {
            state.p[i] = 0;
            for (int j = 0; j < ctx.n_change; j++) {
                int idx = (i <= j) ? (i + j * (j + 1) / 2) : (j + i * (i + 1) / 2);
                state.p[i] += state.h[idx] * state.g[j];
            }
            state.p[i] = -state.p[i];
        }

        // DEBUG: Print first iteration details
        if (tid == 0 && iter == 0) {
            printf("GPU DEBUG iter 0: Search direction p (first 6): ");
            for (int i = 0; i < 6 && i < ctx.n_change; i++) {
                printf("%f ", state.p[i]);
            }
            printf("\n");
        }

        // Line search
        float f_new;
        float alpha = accurate_line_search_single_thread(
            ctx, state.x, state.x_new, state.g, state.p,
            state.energy, state.coords, state.forces,
            state.node_forces, state.node_torques, state.g_new, f_new
        );

        // DEBUG: Print line search result
        if (tid == 0 && iter == 0) {
            printf("GPU DEBUG iter 0: alpha=%f, f0=%f, f_new=%f\n", alpha, state.energy, f_new);
        }

        // Check for line search failure
        if (alpha == 0) {
            if (tid == 0) printf("GPU DEBUG: Line search failed at iter %d\n", iter);
            break;
        }

        // y = g_new - g
        for (int i = 0; i < ctx.n_change; i++) {
            state.y[i] = state.g_new[i] - state.g[i];
        }

        // Shanno-Phua scaling on first iteration
        if (iter == 0) {
            float yy = 0;
            float yp = 0;
            for (int i = 0; i < ctx.n_change; i++) {
                yy += state.y[i] * state.y[i];
                yp += state.y[i] * state.p[i];
            }
            if (yy > epsilon_fl) {
                float scale = alpha * yp / yy;
                int hess_size = ctx.n_change * (ctx.n_change + 1) / 2;
                for (int i = 0; i < hess_size; i++) {
                    state.h[i] = 0;
                }
                for (int i = 0; i < ctx.n_change; i++) {
                    int idx = i + i * (i + 1) / 2;
                    state.h[idx] = scale;
                }
            }
        }

        // BFGS Hessian update
        bfgs_hessian_update_single_thread(state.h, state.p, state.y, alpha, ctx.n_change);

        // Accept step
        for (int i = 0; i < ctx.n_conf; i++) {
            state.x[i] = state.x_new[i];
        }
        for (int i = 0; i < ctx.n_change; i++) {
            state.g[i] = state.g_new[i];
        }
        state.energy = f_new;

        // Track best
        if (state.energy < state.best_energy) {
            state.best_energy = state.energy;
            for (int i = 0; i < ctx.n_conf; i++) {
                mem.all_best_confs[tid * mem.max_conf_size + i] = state.x[i];
            }
        }
    }

    // Store final results
    mem.all_energies[tid] = state.energy;
    mem.all_best_energies[tid] = state.best_energy;
}

// ============================================================================
// Local Minimization Interface
// ============================================================================

void run_parallel_bfgs_minimize(
    const gpu_data& gdata,
    const GPUCacheInfo& cacheInfo,
    const std::vector<float>& input_conf,
    int max_iterations,
    float& out_energy,
    float& out_intramolecular,
    std::vector<float>& out_conf
) {
    // Create scoring context
    ScoringContext ctx;
    create_scoring_context(ctx, gdata, cacheInfo);

    // Allocate context on device
    ScoringContext* d_contexts;
    CUDA_CHECK_GNINA(cudaMalloc(&d_contexts, sizeof(ScoringContext)));
    CUDA_CHECK_GNINA(cudaMemcpy(d_contexts, &ctx, sizeof(ScoringContext), cudaMemcpyHostToDevice));

    // Copy input conf to device
    float* d_input_conf;
    CUDA_CHECK_GNINA(cudaMalloc(&d_input_conf, input_conf.size() * sizeof(float)));
    CUDA_CHECK_GNINA(cudaMemcpy(d_input_conf, input_conf.data(),
                                input_conf.size() * sizeof(float), cudaMemcpyHostToDevice));

    // For minimize, we just need 1 optimizer
    int n_optimizers = 1;

    // Allocate batch memory
    BFGSBatchMemory mem;
    int max_conf = ctx.n_conf > 0 ? ctx.n_conf : PARALLEL_MAX_CONF_SIZE;
    int max_change = ctx.n_change > 0 ? ctx.n_change : PARALLEL_MAX_CHANGE_SIZE;
    int max_atoms = ctx.num_atoms > 0 ? ctx.num_atoms : PARALLEL_MAX_ATOMS;
    int max_nodes = ctx.num_nodes > 0 ? ctx.num_nodes : PARALLEL_MAX_TORSIONS;

    allocate_batch_memory(mem, n_optimizers, max_conf, max_change, max_atoms, max_nodes);

    // Launch minimize kernel (single thread is fine)
    bfgs_minimize_kernel<<<1, 1>>>(
        d_input_conf,
        d_contexts,
        mem,
        max_iterations,
        n_optimizers
    );

    // Synchronize
    CUDA_CHECK_GNINA(cudaDeviceSynchronize());

    // Collect results
    float best_energy;
    CUDA_CHECK_GNINA(cudaMemcpy(&best_energy, mem.all_best_energies,
                                sizeof(float), cudaMemcpyDeviceToHost));

    out_conf.resize(max_conf);
    CUDA_CHECK_GNINA(cudaMemcpy(out_conf.data(), mem.all_best_confs,
                                max_conf * sizeof(float), cudaMemcpyDeviceToHost));

    // The energy from BFGS is total (inter + intra)
    // We need to compute intramolecular separately for reporting
    // For now, return the total energy as inter-adjusted
    // (TODO: compute proper intramolecular on final pose)
    out_energy = best_energy;
    out_intramolecular = 0;  // Will be computed by caller using model

    // Cleanup
    free_batch_memory(mem);
    cudaFree(d_contexts);
    cudaFree(d_input_conf);
}

// ============================================================================
// Score-Only Kernel and Interface
// ============================================================================

// Kernel for evaluating energy without optimization (score_only mode)
__global__ void score_only_kernel(
    const float* input_conf,
    const ScoringContext* contexts,
    float* out_inter_energy,
    float* out_intra_energy,
    float* scratch_coords,      // [num_atoms * 3]
    float* scratch_forces       // [num_atoms * 3]
) {
    // Single thread kernel
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const ScoringContext& ctx = contexts[0];

    // Scratch for node state during set_conf
    float node_origins[PARALLEL_MAX_TORSIONS * 3];
    float node_orientations[PARALLEL_MAX_TORSIONS * 9];
    float node_axes[PARALLEL_MAX_TORSIONS * 3];  // Not used for score_only, but required by set_conf

    // Transform internal coordinates to Cartesian
    set_conf_single_thread(ctx, input_conf, scratch_coords, node_origins, node_orientations, node_axes);

    // Evaluate inter-molecular energy (grid-based)
    float inter_e = eval_energy_grid_single_thread(ctx, scratch_coords, scratch_forces);

    // Evaluate intramolecular energy
    float intra_e = eval_intramolecular_single_thread(ctx, scratch_coords, scratch_forces);

    *out_inter_energy = inter_e;
    *out_intra_energy = intra_e;
}

void run_gpu_score_only(
    const gpu_data& gdata,
    const GPUCacheInfo& cacheInfo,
    const std::vector<float>& input_conf,
    float& out_inter_energy,
    float& out_intramolecular_energy
) {
    // Create scoring context
    ScoringContext ctx;
    create_scoring_context(ctx, gdata, cacheInfo);

    // Allocate context on device
    ScoringContext* d_contexts;
    CUDA_CHECK_GNINA(cudaMalloc(&d_contexts, sizeof(ScoringContext)));
    CUDA_CHECK_GNINA(cudaMemcpy(d_contexts, &ctx, sizeof(ScoringContext), cudaMemcpyHostToDevice));

    // Copy input conf to device
    float* d_input_conf;
    CUDA_CHECK_GNINA(cudaMalloc(&d_input_conf, input_conf.size() * sizeof(float)));
    CUDA_CHECK_GNINA(cudaMemcpy(d_input_conf, input_conf.data(),
                                input_conf.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate output
    float* d_inter_energy;
    float* d_intra_energy;
    CUDA_CHECK_GNINA(cudaMalloc(&d_inter_energy, sizeof(float)));
    CUDA_CHECK_GNINA(cudaMalloc(&d_intra_energy, sizeof(float)));

    // Allocate scratch space for coords and forces
    int max_atoms = ctx.num_atoms > 0 ? ctx.num_atoms : PARALLEL_MAX_ATOMS;
    float* d_coords;
    float* d_forces;
    CUDA_CHECK_GNINA(cudaMalloc(&d_coords, max_atoms * 3 * sizeof(float)));
    CUDA_CHECK_GNINA(cudaMalloc(&d_forces, max_atoms * 3 * sizeof(float)));

    // Launch score-only kernel (single thread)
    score_only_kernel<<<1, 1>>>(
        d_input_conf,
        d_contexts,
        d_inter_energy,
        d_intra_energy,
        d_coords,
        d_forces
    );

    // Synchronize
    CUDA_CHECK_GNINA(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK_GNINA(cudaMemcpy(&out_inter_energy, d_inter_energy,
                                sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_GNINA(cudaMemcpy(&out_intramolecular_energy, d_intra_energy,
                                sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_contexts);
    cudaFree(d_input_conf);
    cudaFree(d_inter_energy);
    cudaFree(d_intra_energy);
    cudaFree(d_coords);
    cudaFree(d_forces);
}
