kernel void
compute_forces(global const float *x, global const float *y,
               global const float *z, global float *fx, global float *fy,
               global float *fz, global float *energy,
               global const uint *neighbors, global const uint *offsets,
               global const float4 *ljTable, global const uint *atomTypes,
               const float wallMinX, const float wallMinY, const float wallMinZ,
               const float wallMaxX, const float wallMaxY, const float wallMaxZ,
               const float gravX, const float gravY, const float gravZ,
               const float epsilon, const uint typeCount, const int n) {
  const int i = get_global_id(0);
  if (i >= n)
    return;

  const float xi = x[i], yi = y[i], zi = z[i];
  float fix = 0.0f, fiy = 0.0f, fiz = 0.0f, ei = 0.0f;

  const uint typeI = atomTypes[i];
  const uint begin = offsets[i];
  const uint end = offsets[i + 1];

  for (uint p = begin; p < end; ++p) {
    const uint j = neighbors[p];

    const float dx = x[j] - xi;
    const float dy = y[j] - yi;
    const float dz = z[j] - zi;
    const float d2 = dx * dx + dy * dy + dz * dz;
    if (d2 <= epsilon)
      continue;

    const float4 lj = ljTable[typeI * typeCount + atomTypes[j]];

    const float invD2 = 1.0f / d2;
    const float invD6 = invD2 * invD2 * invD2;
    const float invD12 = invD6 * invD6;

    const float scale = (lj.y * invD12 - lj.x * invD6) * invD2;
    fix -= dx * scale;
    fiy -= dy * scale;
    fiz -= dz * scale;

    ei += 0.5f * (lj.w * invD12 - lj.z * invD6);
  }

  // мягкие стены
  {
    const float border = 2.0f;
    const float k = 500.0f;

    // X
    if (xi > wallMinX && xi < wallMaxX) {
      if (xi < wallMinX + border) {
        const float p = (wallMinX + border) - xi;
        fix += k * p * p * p * p * p * p;
      } else if (xi > wallMaxX - border) {
        const float p = xi - (wallMaxX - border);
        fix -= k * p * p * p * p * p * p;
      }
    }
    // Y
    if (yi > wallMinY && yi < wallMaxY) {
      if (yi < wallMinY + border) {
        const float p = (wallMinY + border) - yi;
        fiy += k * p * p * p * p * p * p;
      } else if (yi > wallMaxY - border) {
        const float p = yi - (wallMaxY - border);
        fiy -= k * p * p * p * p * p * p;
      }
    }
    // Z
    if (zi > wallMinZ && zi < wallMaxZ) {
      if (zi < wallMinZ + border) {
        const float p = (wallMinZ + border) - zi;
        fiz += k * p * p * p * p * p * p;
      } else if (zi > wallMaxZ - border) {
        const float p = zi - (wallMaxZ - border);
        fiz -= k * p * p * p * p * p * p;
      }
    }
  }

  fix += gravX;
  fiy += gravY;
  fiz += gravZ;

  fx[i] = fix;
  fy[i] = fiy;
  fz[i] = fiz;
  energy[i] = ei;
}

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

kernel void correct_velocities(global float *vx, global float *vy,
                               global float *vz, global const float *fx,
                               global const float *fy, global const float *fz,
                               global const float *pfx, global const float *pfy,
                               global const float *pfz,
                               global const float *invMass, const float dt,
                               const int n) {
  const int i = get_global_id(0);
  if (i >= n)
    return;

  const float halfDtInvMass = 0.5f * dt * invMass[i];

  vx[i] += (pfx[i] + fx[i]) * halfDtInvMass;
  vy[i] += (pfy[i] + fy[i]) * halfDtInvMass;
  vz[i] += (pfz[i] + fz[i]) * halfDtInvMass;
}

kernel void confine_to_box(global float *x, global float *y, global float *z,
                           global float *vx, global float *vy, global float *vz,
                           const float maxX, const float maxY, const float maxZ,
                           const int n) {
  const int i = get_global_id(0);
  if (i >= n)
    return;

  const float restitution = 0.8f;

  if (x[i] < 0.0f) {
    x[i] = 0.0f;
    if (vx[i] < 0.0f)
      vx[i] = -vx[i] * restitution;
  } else if (x[i] > maxX) {
    x[i] = maxX;
    if (vx[i] > 0.0f)
      vx[i] = -vx[i] * restitution;
  }

  if (y[i] < 0.0f) {
    y[i] = 0.0f;
    if (vy[i] < 0.0f)
      vy[i] = -vy[i] * restitution;
  } else if (y[i] > maxY) {
    y[i] = maxY;
    if (vy[i] > 0.0f)
      vy[i] = -vy[i] * restitution;
  }

  if (z[i] < 0.0f) {
    z[i] = 0.0f;
    if (vz[i] < 0.0f)
      vz[i] = -vz[i] * restitution;
  } else if (z[i] > maxZ) {
    z[i] = maxZ;
    if (vz[i] > 0.0f)
      vz[i] = -vz[i] * restitution;
  }
}
