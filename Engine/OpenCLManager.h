#pragma once

#include <CL/opencl.hpp>
#include <vector>
#include <string_view>
#include <fstream>
#include <span>

class OpenCLManager
{
private:
    cl::Context      context;
    cl::CommandQueue queue;
    cl::Program      program;

    struct
    {
        cl::Kernel integratePositions;
    } kernels;

    struct
    {
        cl::Buffer x, y, z;
        cl::Buffer vx, vy, vz;
        cl::Buffer fx, fy, fz;
        cl::Buffer invMass;
    } buffers;

    size_t maxWorkGroupSize;
    size_t atomCount;

public:
    OpenCLManager()
    {
        initDevice();
    }

    void initDevice()
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
            throw std::runtime_error("No OpenCL platforms found");

        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (devices.empty())
            throw std::runtime_error("No OpenCL devices found");

        context = cl::Context(devices[0]);
        queue   = cl::CommandQueue(context, devices[0]);

        devices[0].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
        if (maxWorkGroupSize > 1024)
            maxWorkGroupSize = 1024;
    }

    void setupResources(std::span<const float> x,
                        std::span<const float> y,
                        std::span<const float> z,
                        std::span<const float> vx,
                        std::span<const float> vy,
                        std::span<const float> vz,
                        std::span<const float> fx,
                        std::span<const float> fy,
                        std::span<const float> fz,
                        std::span<const float> invMass)
    {
        atomCount = x.size();

        auto createBuffer = [&](cl_mem_flags flags, const float* ptr = nullptr)
        {
            cl_int err;
            cl::Buffer buf(context,
                           flags | (ptr ? CL_MEM_COPY_HOST_PTR : 0),
                           atomCount * sizeof(float),
                           const_cast<float*>(ptr),
                           &err);
            if (err != CL_SUCCESS)
                throw std::runtime_error("Failed to create OpenCL buffer: " + std::to_string(err));
            return buf;
        };

        buffers.x       = createBuffer(CL_MEM_READ_WRITE, x.data());
        buffers.y       = createBuffer(CL_MEM_READ_WRITE, y.data());
        buffers.z       = createBuffer(CL_MEM_READ_WRITE, z.data());
        buffers.vx      = createBuffer(CL_MEM_READ_WRITE, vx.data());
        buffers.vy      = createBuffer(CL_MEM_READ_WRITE, vy.data());
        buffers.vz      = createBuffer(CL_MEM_READ_WRITE, vz.data());
        buffers.fx      = createBuffer(CL_MEM_READ_WRITE, fx.data());
        buffers.fy      = createBuffer(CL_MEM_READ_WRITE, fy.data());
        buffers.fz      = createBuffer(CL_MEM_READ_WRITE, fz.data());
        buffers.invMass = createBuffer(CL_MEM_READ_ONLY,  invMass.data());
    }

    void loadProgram(std::string_view filename)
    {
        std::ifstream file(filename.data());
        if (!file)
            throw std::runtime_error("Kernel file not found: " + std::string(filename));

        std::string source((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        program = cl::Program(context, source);

        auto device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        if (program.build({device}) != CL_SUCCESS)
            throw std::runtime_error("OpenCL build error:\n" +
                                     program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));

        kernels.integratePositions = cl::Kernel(program, "integrate_positions");
    }

    void uploadForces(std::span<const float> fx,
                      std::span<const float> fy,
                      std::span<const float> fz)
    {
        queue.enqueueWriteBuffer(buffers.fx, CL_FALSE, 0, atomCount * sizeof(float), fx.data());
        queue.enqueueWriteBuffer(buffers.fy, CL_FALSE, 0, atomCount * sizeof(float), fy.data());
        queue.enqueueWriteBuffer(buffers.fz, CL_FALSE, 0, atomCount * sizeof(float), fz.data());
    }

    void uploadVelocities(std::span<const float> vx,
                          std::span<const float> vy,
                          std::span<const float> vz)
    {
        queue.enqueueWriteBuffer(buffers.vx, CL_FALSE, 0, atomCount * sizeof(float), vx.data());
        queue.enqueueWriteBuffer(buffers.vy, CL_FALSE, 0, atomCount * sizeof(float), vy.data());
        queue.enqueueWriteBuffer(buffers.vz, CL_FALSE, 0, atomCount * sizeof(float), vz.data());
    }

    void setupArgs(float dt)
    {
        int i = 0;
        auto& k = kernels.integratePositions;
        k.setArg(i++, buffers.x);
        k.setArg(i++, buffers.y);
        k.setArg(i++, buffers.z);
        k.setArg(i++, buffers.vx);
        k.setArg(i++, buffers.vy);
        k.setArg(i++, buffers.vz);
        k.setArg(i++, buffers.fx);
        k.setArg(i++, buffers.fy);
        k.setArg(i++, buffers.fz);
        k.setArg(i++, buffers.invMass);
        k.setArg(i++, dt);
        k.setArg(i++, static_cast<cl_int>(atomCount));
    }

    void runIntegrate()
    {
        size_t globalSize = ((atomCount + maxWorkGroupSize - 1) / maxWorkGroupSize) * maxWorkGroupSize;
        queue.enqueueNDRangeKernel(kernels.integratePositions,
                                   cl::NullRange,
                                   cl::NDRange(globalSize),
                                   cl::NDRange(maxWorkGroupSize));
    }

    // Считать результаты обратно на CPU
    void downloadPositions(std::span<float> x,
                           std::span<float> y,
                           std::span<float> z)
    {
        queue.enqueueReadBuffer(buffers.x, CL_FALSE, 0, atomCount * sizeof(float), x.data());
        queue.enqueueReadBuffer(buffers.y, CL_FALSE, 0, atomCount * sizeof(float), y.data());
        queue.enqueueReadBuffer(buffers.z, CL_FALSE, 0, atomCount * sizeof(float), z.data());
    }

    void finish() { queue.finish(); }
};