#include "VerletCL.h"

#include "Engine/metrics/Profiler.h"
#include "StepOps.h"

OpenCLManager VerletCL::openclManager{};
bool VerletCL::buffersReady = false;

VerletCL::VerletCL()
{
    openclManager.loadProgram("Engine/VerletCL.cl");
}

void VerletCL::pipeline(AtomStorage& atomStorage, SimBox& box, ForceField& forceField, NeighborList* neighborList, float dt) const {
    PROFILE_SCOPE("VerletScheme::pipeline");
    
    const size_t n = atomStorage.mobileCount();
    
    if (!buffersReady)
    {
        VerletCL::openclManager.setupResources(
            {atomStorage.xData(), n},
            {atomStorage.yData(), n},
            {atomStorage.zData(), n},
            {atomStorage.vxData(), n},
            {atomStorage.vyData(), n},
            {atomStorage.vzData(), n},
            {atomStorage.fxData(), n},
            {atomStorage.fyData(), n},
            {atomStorage.fzData(), n},
            {atomStorage.invMassData(), n}
        );
        VerletCL::openclManager.setupArgs(dt);
        VerletCL::buffersReady = true;
    }

    // Расчет новых позиций
    StepOps::predictAndSync(atomStorage, box, dt, &predict);
    // Расчет сил
    StepOps::computeForces(atomStorage, box, forceField, neighborList, dt);
    // Корректировка скоростей
    correct(atomStorage, dt);
}

void VerletCL::predict(AtomStorage& atomStorage, float dt) {
    PROFILE_SCOPE("VerletScheme::predict");
    const size_t n = atomStorage.mobileCount();

    openclManager.uploadVelocities(
        {atomStorage.vxData(), n},
        {atomStorage.vyData(), n},
        {atomStorage.vzData(), n}
    );
    openclManager.uploadForces(
        {atomStorage.fxData(), n},
        {atomStorage.fyData(), n},
        {atomStorage.fzData(), n}
    );

    openclManager.runIntegrate();
    openclManager.downloadPositions(
        {atomStorage.xData(), n},
        {atomStorage.yData(), n},
        {atomStorage.zData(), n}
    );
    openclManager.finish();
}

void VerletCL::correct(AtomStorage& atomStorage, float dt) {
    PROFILE_SCOPE("VerletScheme::correct");
    const size_t n = atomStorage.mobileCount();

    const float* RESTRICT fx  = atomStorage.fxData();
    const float* RESTRICT fy  = atomStorage.fyData();
    const float* RESTRICT fz  = atomStorage.fzData();

    const float* RESTRICT pfx = atomStorage.pfxData();    
    const float* RESTRICT pfy = atomStorage.pfyData();
    const float* RESTRICT pfz = atomStorage.pfzData();

    float* RESTRICT vx = atomStorage.vxData();
    float* RESTRICT vy = atomStorage.vyData();
    float* RESTRICT vz = atomStorage.vzData();

    const float* RESTRICT invMass = atomStorage.invMassData();

    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        const float halfDtInvMass = 0.5f * dt * invMass[i];

        vx[i] += (pfx[i] + fx[i]) * halfDtInvMass;
        vy[i] += (pfy[i] + fy[i]) * halfDtInvMass;
        vz[i] += (pfz[i] + fz[i]) * halfDtInvMass;
    }
}
