kernel void integrate_positions(global float *x, global float *y,
                                global float *z, global const float *vx,
                                global const float *vy, global const float *vz,
                                global const float *fx, global const float *fy,
                                global const float *fz,
                                global const float *invMass, const float dt,
                                const int n) {
  const int i = get_global_id(0);
  if (i >= n)
    return;

  const float damping = 0.6f;
  const float half_dt = 0.5f * dt;

  x[i] += (vx[i] * damping + fx[i] * invMass[i] * half_dt) * dt;
  y[i] += (vy[i] * damping + fy[i] * invMass[i] * half_dt) * dt;
  z[i] += (vz[i] * damping + fz[i] * invMass[i] * half_dt) * dt;
}