#pragma once

#include "Engine/OpenCLManager.h"

class AtomStorage;
class ForceField;
class NeighborList;
class SimBox;

class VerletCL {
public:
    VerletCL();
    void pipeline(AtomStorage& atomStorage, SimBox& box, ForceField& forceField, NeighborList* neighborList, float dt) const;

    static void predict(AtomStorage& atomStorage, float dt);
    static void correct(AtomStorage& atomStorage, float dt);

    static OpenCLManager openclManager;
    static bool buffersReady;
};