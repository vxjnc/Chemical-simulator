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
        constexpr size_t N = 119;
        std::vector<float> flatLJ(N * N * 4);
        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = 0; j < N; ++j) {
                const auto& p = forceField.ljPairTable[i][j];
                flatLJ[(i * N + j) * 4 + 0] = p.forceC6;
                flatLJ[(i * N + j) * 4 + 1] = p.forceC12;
                flatLJ[(i * N + j) * 4 + 2] = p.potentialC6;
                flatLJ[(i * N + j) * 4 + 3] = p.potentialC12;
            }
        }

        std::vector<uint32_t> types(n);
        for (size_t i = 0; i < n; ++i)
            types[i] = static_cast<uint32_t>(atomStorage.type(i));


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
            {atomStorage.pfxData(), n},
            {atomStorage.pfyData(), n},
            {atomStorage.pfzData(), n},
            {atomStorage.invMassData(), n}
        );
        openclManager.setupLJTable(flatLJ);
        const Vec3f max = box.size - Vec3f(1.0, 1.0, 1.0);
        openclManager.setupConfineToBoxArgs(max.x, max.y, max.z);
        openclManager.setupAtomTypes(types);
        openclManager.setupIntegrateArgs(dt);
        buffersReady = true;
    }

    openclManager.runIntegrate();
    openclManager.downloadPositions(atomStorage.xDataSpan(), atomStorage.yDataSpan(), atomStorage.zDataSpan());
    openclManager.finish();
    openclManager.runConfineToBox();
    openclManager.swapAndClearForces();
    openclManager.runComputeForces();
    openclManager.runCorrect();
    openclManager.finish();
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
