// =====================================================================
// FE_GPU.cpp
// FEM 2D/3D - GPU Computing via OpenCL (TANPA CUDA Toolkit)
//
// - GPU NVIDIA memakai runtime OpenCL bawaan driver (OpenCL.dll) yang
//   di-load DINAMIS via LoadLibrary -> tidak butuh CUDA Toolkit dan
//   tidak butuh OpenCL SDK/header apapun.
// - Kernel OpenCL (double precision via cl_khr_fp64) menghitung &
//   merakit matriks kekakuan elemen (CPS3 2D / C3D4 3D) -> triplet COO.
// - CPU (Eigen SparseLU) menyelesaikan sistem linear + menulis VTK.
// - Fallback CPU otomatis bila OpenCL/GPU/fp64 tidak tersedia.
//
// Penggunaan:
//   ./FE_GPU.exe <file_input.inp> <file_output.vtk>
// =====================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#define NOMINMAX
#include <windows.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

// ============ Definisi OpenCL Minimal (tanpa SDK header) ============
typedef uint32_t cl_uint;
typedef uint32_t cl_bool;
typedef int32_t  cl_int;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef uint64_t cl_mem_flags;
typedef uint64_t cl_device_type;
typedef uint64_t cl_command_queue_properties;
typedef intptr_t cl_context_properties;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;

#define CL_API_CALL
#define CL_CALLBACK
#define CL_SUCCESS                0
#define CL_TRUE                   1
#define CL_DEVICE_TYPE_GPU        ((cl_device_type)1 << 1)
#define CL_DEVICE_TYPE_ALL        ((cl_device_type)0xFFFFFFFF)
#define CL_MEM_READ_WRITE         ((cl_mem_flags)1 << 0)
#define CL_MEM_WRITE_ONLY         ((cl_mem_flags)1 << 1)
#define CL_MEM_READ_ONLY          ((cl_mem_flags)1 << 2)
#define CL_MEM_COPY_HOST_PTR      ((cl_mem_flags)1 << 5)
#define CL_QUEUE_PROFILING_ENABLE ((cl_command_queue_properties)1 << 1)
#define CL_QUEUE_PROPERTIES       0x1093
#define CL_PLATFORM_NAME          0x0902
#define CL_PLATFORM_VERSION       0x0903
#define CL_DEVICE_NAME                0x102B
#define CL_DEVICE_MAX_COMPUTE_UNITS   0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_GLOBAL_MEM_SIZE     0x101F
#define CL_DEVICE_EXTENSIONS          0x102D
#define CL_PROGRAM_BUILD_LOG      0x1183

typedef cl_int (CL_API_CALL *pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (CL_API_CALL *pfn_clGetPlatformInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*);
typedef cl_int (CL_API_CALL *pfn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (CL_API_CALL *pfn_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (CL_API_CALL *pfn_clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void (CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_mem (CL_API_CALL *pfn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_program (CL_API_CALL *pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (CL_API_CALL *pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void (CL_CALLBACK*)(cl_program, void*), void*);
typedef cl_int (CL_API_CALL *pfn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_kernel (CL_API_CALL *pfn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (CL_API_CALL *pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_command_queue (CL_API_CALL *pfn_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_command_queue (CL_API_CALL *pfn_clCreateCommandQueueWithProperties)(cl_context, cl_device_id, const cl_context_properties*, cl_int*);
typedef cl_int (CL_API_CALL *pfn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, void**, void**);
typedef cl_int (CL_API_CALL *pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, void**, void**);
typedef cl_int (CL_API_CALL *pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, void**, void**);
typedef cl_int (CL_API_CALL *pfn_clFinish)(cl_command_queue);
typedef cl_int (CL_API_CALL *pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (CL_API_CALL *pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (CL_API_CALL *pfn_clReleaseProgram)(cl_program);
typedef cl_int (CL_API_CALL *pfn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (CL_API_CALL *pfn_clReleaseContext)(cl_context);

// ============ Loader Dinamis OpenCL.dll ============
struct OpenCL {
    HMODULE dll = nullptr;
    pfn_clGetPlatformIDs clGetPlatformIDs = nullptr;
    pfn_clGetPlatformInfo clGetPlatformInfo = nullptr;
    pfn_clGetDeviceIDs clGetDeviceIDs = nullptr;
    pfn_clGetDeviceInfo clGetDeviceInfo = nullptr;
    pfn_clCreateContext clCreateContext = nullptr;
    pfn_clCreateBuffer clCreateBuffer = nullptr;
    pfn_clCreateProgramWithSource clCreateProgramWithSource = nullptr;
    pfn_clBuildProgram clBuildProgram = nullptr;
    pfn_clGetProgramBuildInfo clGetProgramBuildInfo = nullptr;
    pfn_clCreateKernel clCreateKernel = nullptr;
    pfn_clSetKernelArg clSetKernelArg = nullptr;
    pfn_clCreateCommandQueue clCreateCommandQueue = nullptr;
    pfn_clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties = nullptr;
    pfn_clEnqueueWriteBuffer clEnqueueWriteBuffer = nullptr;
    pfn_clEnqueueReadBuffer clEnqueueReadBuffer = nullptr;
    pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel = nullptr;
    pfn_clFinish clFinish = nullptr;
    pfn_clReleaseMemObject clReleaseMemObject = nullptr;
    pfn_clReleaseKernel clReleaseKernel = nullptr;
    pfn_clReleaseProgram clReleaseProgram = nullptr;
    pfn_clReleaseCommandQueue clReleaseCommandQueue = nullptr;
    pfn_clReleaseContext clReleaseContext = nullptr;

    bool ok = false;
    std::string err;

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_uint computeUnits = 0;
    size_t maxWorkGroup = 0;
    cl_ulong globalMem = 0;
    std::string devName, platName, platVersion;
};

static const char* clErrStr(cl_int e) {
    switch (e) {
        case  0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -30: return "CL_INVALID_VALUE";
        case -32: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -44: return "CL_INVALID_KERNEL";
        case -46: return "CL_INVALID_KERNEL_ARGS";
        case -48: return "CL_INVALID_WORK_GROUP_SIZE";
        case -52: return "CL_INVALID_ARG_VALUE";
        case -54: return "CL_INVALID_ARG_SIZE";
        default: return "CL_UNKNOWN_ERROR";
    }
}

static bool loadOpenCL(OpenCL& ocl) {
    ocl.dll = LoadLibraryA("OpenCL.dll");
    if (!ocl.dll) ocl.dll = LoadLibraryA("C:\\Windows\\System32\\OpenCL.dll");
    if (!ocl.dll) {
        ocl.err = "OpenCL.dll tidak ditemukan (driver GPU belum terpasang)";
        return false;
    }
    #define CLGET(f) do { \
        ocl.f = (pfn_##f)GetProcAddress(ocl.dll, #f); \
        if (!ocl.f) { ocl.err = "Simbol " #f " tidak ditemukan di OpenCL.dll"; return false; } \
    } while (0)
    CLGET(clGetPlatformIDs);
    CLGET(clGetPlatformInfo);
    CLGET(clGetDeviceIDs);
    CLGET(clGetDeviceInfo);
    CLGET(clCreateContext);
    CLGET(clCreateBuffer);
    CLGET(clCreateProgramWithSource);
    CLGET(clBuildProgram);
    CLGET(clGetProgramBuildInfo);
    CLGET(clCreateKernel);
    CLGET(clSetKernelArg);
    CLGET(clEnqueueWriteBuffer);
    CLGET(clEnqueueReadBuffer);
    CLGET(clEnqueueNDRangeKernel);
    CLGET(clFinish);
    CLGET(clReleaseMemObject);
    CLGET(clReleaseKernel);
    CLGET(clReleaseProgram);
    CLGET(clReleaseCommandQueue);
    CLGET(clReleaseContext);
    #undef CLGET
    // Salah satu API pembuatan queue cukup (CL 1.2 legacy / CL 2.0+ properties)
    ocl.clCreateCommandQueue = (pfn_clCreateCommandQueue)GetProcAddress(ocl.dll, "clCreateCommandQueue");
    ocl.clCreateCommandQueueWithProperties = (pfn_clCreateCommandQueueWithProperties)GetProcAddress(ocl.dll, "clCreateCommandQueueWithProperties");
    if (!ocl.clCreateCommandQueue && !ocl.clCreateCommandQueueWithProperties) {
        ocl.err = "API pembuatan command queue tidak tersedia";
        return false;
    }
    return true;
}

static bool pickDevice(OpenCL& ocl) {
    cl_uint npl = 0;
    if (ocl.clGetPlatformIDs(0, nullptr, &npl) != CL_SUCCESS || npl == 0) {
        ocl.err = "Tidak ada platform OpenCL";
        return false;
    }
    std::vector<cl_platform_id> pls(npl);
    ocl.clGetPlatformIDs(npl, pls.data(), nullptr);

    for (auto p : pls) {
        char nm[256] = {0}, ver[256] = {0};
        ocl.clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(nm), nm, nullptr);
        ocl.clGetPlatformInfo(p, CL_PLATFORM_VERSION, sizeof(ver), ver, nullptr);
        cl_uint nd = 0;
        if (ocl.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0)
            continue;
        std::vector<cl_device_id> devs(nd);
        ocl.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
        for (auto d : devs) {
            char ext[8192] = {0};
            ocl.clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, sizeof(ext), ext, nullptr);
            if (!strstr(ext, "cl_khr_fp64")) continue; // butuh double precision
            char dn[256] = {0};
            ocl.clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(dn), dn, nullptr);
            cl_uint cu = 0; cl_ulong gm = 0; size_t mwg = 0;
            ocl.clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
            ocl.clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gm), &gm, nullptr);
            ocl.clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(mwg), &mwg, nullptr);
            ocl.platform = p; ocl.device = d;
            ocl.devName = dn; ocl.platName = nm; ocl.platVersion = ver;
            ocl.computeUnits = cu; ocl.globalMem = gm; ocl.maxWorkGroup = mwg;
            return true;
        }
    }
    ocl.err = "Tidak ada GPU OpenCL dengan dukungan double (cl_khr_fp64)";
    return false;
}

static bool setupContextAndKernel(OpenCL& ocl, const char* src) {
    cl_int e = 0;
    ocl.context = ocl.clCreateContext(nullptr, 1, &ocl.device, nullptr, nullptr, &e);
    if (!ocl.context || e != CL_SUCCESS) {
        ocl.err = std::string("clCreateContext gagal: ") + clErrStr(e);
        return false;
    }
    if (ocl.clCreateCommandQueue)
        ocl.queue = ocl.clCreateCommandQueue(ocl.context, ocl.device, CL_QUEUE_PROFILING_ENABLE, &e);
    else {
        cl_context_properties props[3] = {CL_QUEUE_PROPERTIES, (cl_context_properties)CL_QUEUE_PROFILING_ENABLE, 0};
        ocl.queue = ocl.clCreateCommandQueueWithProperties(ocl.context, ocl.device, props, &e);
    }
    if (!ocl.queue || e != CL_SUCCESS) {
        ocl.err = std::string("clCreateCommandQueue gagal: ") + clErrStr(e);
        return false;
    }
    size_t len = strlen(src);
    ocl.program = ocl.clCreateProgramWithSource(ocl.context, 1, &src, &len, &e);
    if (!ocl.program || e != CL_SUCCESS) {
        ocl.err = std::string("clCreateProgramWithSource gagal: ") + clErrStr(e);
        return false;
    }
    e = ocl.clBuildProgram(ocl.program, 1, &ocl.device, nullptr, nullptr, nullptr);
    if (e != CL_SUCCESS) {
        char log[8192] = {0};
        ocl.clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, nullptr);
        ocl.err = std::string("Build kernel gagal (") + clErrStr(e) + "):\n" + log;
        return false;
    }
    ocl.kernel = ocl.clCreateKernel(ocl.program, "assemble_fe", &e);
    if (!ocl.kernel || e != CL_SUCCESS) {
        ocl.err = std::string("clCreateKernel gagal: ") + clErrStr(e);
        return false;
    }
    return true;
}

static void releaseOpenCL(OpenCL& ocl) {
    if (ocl.ok) {
        if (ocl.kernel)  ocl.clReleaseKernel(ocl.kernel);
        if (ocl.program) ocl.clReleaseProgram(ocl.program);
        if (ocl.queue)   ocl.clReleaseCommandQueue(ocl.queue);
        if (ocl.context) ocl.clReleaseContext(ocl.context);
    }
}

// =====================================================================
// Kernel OpenCL: perakitan matriks kekakuan elemen -> triplets COO
// Satu work-item per elemen; tata urut triplet identik dgn versi CPU.
// =====================================================================
static const char* kSrc = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void assemble_fe(
    __global const double* coords,   // dn koordinat per node (x,y[,z])
    __global const int*    conn,     // npe indeks node per elemen (0-based)
    __global int*          rows,
    __global int*          cols,
    __global double*       vals,
    const int   nElem,
    const int   is3D,
    const double E,
    const double nu,
    const double thick)
{
    const int e = get_global_id(0);
    if (e >= nElem) return;

    const int npe   = is3D ? 4 : 3;
    const int dn    = is3D ? 3 : 2;
    const int width = npe * dn;      // 12 (3D) / 6 (2D)
    const long base = (long)e * (long)(npe*npe*dn*dn);

    double B[72];  // width x 6 (row-major: B[l*width + k], l=dof, k=regangan)
    double D[36];  // 6 x 6
    for (int k = 0; k < 72; ++k) B[k] = 0.0;
    for (int k = 0; k < 36; ++k) D[k] = 0.0;

    double vol = 0.0;

    if (is3D) {
        const int n0 = conn[e*4+0], n1 = conn[e*4+1];
        const int n2 = conn[e*4+2], n3 = conn[e*4+3];
        const double x0=coords[n0*3], y0=coords[n0*3+1], z0=coords[n0*3+2];
        const double x1=coords[n1*3], y1=coords[n1*3+1], z1=coords[n1*3+2];
        const double x2=coords[n2*3], y2=coords[n2*3+1], z2=coords[n2*3+2];
        const double x3=coords[n3*3], y3=coords[n3*3+1], z3=coords[n3*3+2];

        double T[16];
        T[0]=1.0; T[1]=x0; T[2]=y0; T[3]=z0;
        T[4]=1.0; T[5]=x1; T[6]=y1; T[7]=z1;
        T[8]=1.0; T[9]=x2; T[10]=y2; T[11]=z2;
        T[12]=1.0; T[13]=x3; T[14]=y3; T[15]=z3;

        // determinan 4x4 (ekspansi kofaktor baris pertama)
        double det = 0.0, sgn = 1.0;
        for (int c0 = 0; c0 < 4; ++c0) {
            double m[9]; int mi = 0;
            for (int r = 1; r < 4; ++r)
                for (int cc = 0; cc < 4; ++cc)
                    if (cc != c0) m[mi++] = T[r*4+cc];
            const double dm = m[0]*(m[4]*m[8]-m[5]*m[7])
                            - m[1]*(m[3]*m[8]-m[5]*m[6])
                            + m[2]*(m[3]*m[7]-m[4]*m[6]);
            det += sgn * T[c0] * dm;
            sgn = -sgn;
        }
        vol = fabs(det) / 6.0;

        // invers 4x4 (Gauss-Jordan + partial pivoting)
        double A[16], I[16];
        for (int k = 0; k < 16; ++k) { A[k] = T[k]; I[k] = 0.0; }
        I[0] = I[5] = I[10] = I[15] = 1.0;
        for (int col = 0; col < 4; ++col) {
            int piv = col;
            for (int r = col+1; r < 4; ++r)
                if (fabs(A[r*4+col]) > fabs(A[piv*4+col])) piv = r;
            if (piv != col)
                for (int j = 0; j < 4; ++j) {
                    const double ta = A[col*4+j]; A[col*4+j] = A[piv*4+j]; A[piv*4+j] = ta;
                    const double tb = I[col*4+j]; I[col*4+j] = I[piv*4+j]; I[piv*4+j] = tb;
                }
            const double d = A[col*4+col];
            for (int j = 0; j < 4; ++j) { A[col*4+j] /= d; I[col*4+j] /= d; }
            for (int r = 0; r < 4; ++r)
                if (r != col) {
                    const double f = A[r*4+col];
                    for (int j = 0; j < 4; ++j) {
                        A[r*4+j] -= f * A[col*4+j];
                        I[r*4+j] -= f * I[col*4+j];
                    }
                }
        }

        // B (6x12): dN/dx = I[4+i], dN/dy = I[8+i], dN/dz = I[12+i]
        for (int i = 0; i < 4; ++i) {
            const double dNx = I[4  + i];
            const double dNy = I[8  + i];
            const double dNz = I[12 + i];
            B[0*width + 3*i  ] = dNx;
            B[1*width + 3*i+1] = dNy;
            B[2*width + 3*i+2] = dNz;
            B[3*width + 3*i+1] = dNz; B[3*width + 3*i+2] = dNy;
            B[4*width + 3*i  ] = dNz; B[4*width + 3*i+2] = dNx;
            B[5*width + 3*i  ] = dNy; B[5*width + 3*i+1] = dNx;
        }

        // D 3D isotropik (6x6)
        const double fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const double g   = (1.0 - 2.0 * nu) * 0.5;
        D[0]  = fac*(1.0-nu); D[1]  = fac*nu;       D[2]  = fac*nu;
        D[6]  = fac*nu;       D[7]  = fac*(1.0-nu); D[8]  = fac*nu;
        D[12] = fac*nu;       D[13] = fac*nu;       D[14] = fac*(1.0-nu);
        D[3*6+3] = fac*g; D[4*6+4] = fac*g; D[5*6+5] = fac*g;
    } else {
        const int n0 = conn[e*3+0], n1 = conn[e*3+1], n2 = conn[e*3+2];
        const double x0=coords[n0*2], y0=coords[n0*2+1];
        const double x1=coords[n1*2], y1=coords[n1*2+1];
        const double x2=coords[n2*2], y2=coords[n2*2+1];
        const double Area = 0.5 * fabs(x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1));
        vol = thick * Area;

        B[0*6 + 0] = (y1 - y2); B[0*6 + 2] = (y2 - y0); B[0*6 + 4] = (y0 - y1);
        B[1*6 + 1] = (x2 - x1); B[1*6 + 3] = (x0 - x2); B[1*6 + 5] = (x1 - x0);
        B[2*6 + 0] = (x2 - x1); B[2*6 + 1] = (y1 - y2);
        B[2*6 + 2] = (x0 - x2); B[2*6 + 3] = (y2 - y0);
        B[2*6 + 4] = (x1 - x0); B[2*6 + 5] = (y0 - y1);
        for (int k = 0; k < 18; ++k) B[k] /= (2.0 * Area);

        const double f = E / (1.0 - nu * nu);
        D[0*6+0] = f;      D[0*6+1] = f * nu;
        D[1*6+0] = f * nu; D[1*6+1] = f;
        D[2*6+2] = f * (1.0 - nu) * 0.5;
    }

    // S = D * B  (6 x width)
    double S[72];
    for (int m = 0; m < 6; ++m)
        for (int k = 0; k < width; ++k) {
            double s = 0.0;
            for (int j = 0; j < 6; ++j) s += D[m*6+j] * B[j*width+k];
            S[m*width+k] = s;
        }

    // Tulis triplet COO - urutan identik dengan versi CPU:
    //   p = ((i*npe + j)*dn + r)*dn + c
    //   Ke(l,k) = vol * sum_m B(m,l) * S(m,k)  [B^T D B]
    int p = 0;
    for (int i = 0; i < npe; ++i)
        for (int j = 0; j < npe; ++j)
            for (int r = 0; r < dn; ++r)
                for (int c = 0; c < dn; ++c) {
                    const int l = i*dn + r;
                    const int k = j*dn + c;
                    double s = 0.0;
                    for (int m = 0; m < 6; ++m)
                        s += B[m*width + l] * S[m*width + k];
                    rows[base + p] = conn[e*npe + i]*dn + r;
                    cols[base + p] = conn[e*npe + j]*dn + c;
                    vals[base + p] = vol * s;
                    ++p;
                }
}
)CLC";

// ====================== Struktur Data & Parser ======================
struct Node {
    int id;
    double x, y;
    double z = 0.0;
};

// CPS3 (2D, 3 node) / C3D4 (3D, 4 node)
struct Element {
    int id;
    int n1, n2, n3;
    int n4 = 0;
};

struct BoundaryCondition {
    int node_id;
    int dof; // 2D: 1=Ux, 2=Uy | 3D: 1=Ux, 2=Uy, 3=Uz
    double value;
};

struct NodalLoad {
    int node_id;
    int dof;
    double magnitude;
};

class FEMSolver {
public:
    const double E = 210.0e9;   // Modulus Young (Pa)
    const double nu = 0.3;      // Rasio Poisson
    const double t = 0.001;     // Ketebalan (m, hanya 2D)

    bool is3D = false;

    std::vector<Node> nodes;
    std::map<int, int> node_id_to_index;
    std::vector<Element> elements;
    std::vector<BoundaryCondition> bcs;
    std::vector<NodalLoad> loads;

    Eigen::VectorXd U;

    int dofPerNode() const { return is3D ? 3 : 2; }
    int nodesPerElement() const { return is3D ? 4 : 3; }

    bool parseINP(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Tidak dapat membuka file " << filename << std::endl;
            return false;
        }
        std::string line;
        enum Section { NONE, NODE, ELEMENT, BOUNDARY, CLOAD };
        Section currentSection = NONE;
        while (std::getline(file, line)) {
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            if (line.empty()) continue;
            if (line[0] == '*') {
                std::transform(line.begin(), line.end(), line.begin(), ::toupper);
                if (line.find("TYPE=C3D4") != std::string::npos) is3D = true;
                else if (line.find("TYPE=CPS3") != std::string::npos) is3D = false;
                if (line.find("*NODE") == 0) currentSection = NODE;
                else if (line.find("*ELEMENT") == 0) currentSection = ELEMENT;
                else if (line.find("*BOUNDARY") == 0) currentSection = BOUNDARY;
                else if (line.find("*CLOAD") == 0) currentSection = CLOAD;
                else currentSection = NONE;
                continue;
            }
            std::stringstream ss(line);
            std::string item;
            if (currentSection == NODE) {
                Node node;
                std::getline(ss, item, ','); node.id = std::stoi(item);
                std::getline(ss, item, ','); node.x = std::stod(item);
                std::getline(ss, item, ','); node.y = std::stod(item);
                node.z = 0.0;
                if (std::getline(ss, item, ','))
                    if (!item.empty()) node.z = std::stod(item);
                node_id_to_index[node.id] = nodes.size();
                nodes.push_back(node);
            }
            else if (currentSection == ELEMENT) {
                Element elem;
                std::getline(ss, item, ','); elem.id = std::stoi(item);
                std::getline(ss, item, ','); elem.n1 = std::stoi(item);
                std::getline(ss, item, ','); elem.n2 = std::stoi(item);
                std::getline(ss, item, ','); elem.n3 = std::stoi(item);
                elem.n4 = 0;
                if (std::getline(ss, item, ','))
                    if (!item.empty()) elem.n4 = std::stoi(item);
                elements.push_back(elem);
            }
            else if (currentSection == BOUNDARY) {
                BoundaryCondition bc;
                std::getline(ss, item, ','); bc.node_id = std::stoi(item);
                std::getline(ss, item, ','); int start_dof = std::stoi(item);
                std::getline(ss, item, ','); int end_dof = std::stoi(item);
                double val = 0.0;
                if (std::getline(ss, item, ',')) val = std::stod(item);
                for (int d = start_dof; d <= end_dof; ++d)
                    bcs.push_back({bc.node_id, d, val});
            }
            else if (currentSection == CLOAD) {
                NodalLoad load;
                std::getline(ss, item, ','); load.node_id = std::stoi(item);
                std::getline(ss, item, ','); load.dof = std::stoi(item);
                std::getline(ss, item, ','); load.magnitude = std::stod(item);
                loads.push_back(load);
            }
        }
        file.close();
        return true;
    }

    // ===== Perakitan CPU (fallback) =====
    Eigen::Matrix<double, 3, 3> getConstitutiveMatrix() const {
        double factor = E / (1.0 - nu * nu);
        Eigen::Matrix<double, 3, 3> D;
        D << 1.0, nu, 0.0,
             nu, 1.0, 0.0,
             0.0, 0.0, (1.0 - nu) / 2.0;
        return factor * D;
    }

    Eigen::Matrix<double, 6, 6> getConstitutiveMatrix3D() const {
        double factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
        double g = (1.0 - 2.0 * nu) * 0.5;
        Eigen::Matrix<double, 6, 6> D;
        D << 1.0 - nu, nu,      nu,      0.0, 0.0, 0.0,
             nu,      1.0 - nu, nu,      0.0, 0.0, 0.0,
             nu,      nu,       1.0 - nu, 0.0, 0.0, 0.0,
             0.0,     0.0,      0.0,     g,   0.0, 0.0,
             0.0,     0.0,      0.0,     0.0, g,   0.0,
             0.0,     0.0,      0.0,     0.0, 0.0, g;
        return factor * D;
    }

    Eigen::MatrixXd computeElementStiffness(const Element& elem) const {
        if (is3D) {
            const Node& n1 = nodes[node_id_to_index.at(elem.n1)];
            const Node& n2 = nodes[node_id_to_index.at(elem.n2)];
            const Node& n3 = nodes[node_id_to_index.at(elem.n3)];
            const Node& n4 = nodes[node_id_to_index.at(elem.n4)];
            Eigen::Matrix4d T;
            T << 1.0, n1.x, n1.y, n1.z,
                 1.0, n2.x, n2.y, n2.z,
                 1.0, n3.x, n3.y, n3.z,
                 1.0, n4.x, n4.y, n4.z;
            double V = std::abs(T.determinant()) / 6.0;
            Eigen::Matrix4d Ti = T.inverse();
            Eigen::Matrix<double, 6, 12> B = Eigen::Matrix<double, 6, 12>::Zero();
            for (int i = 0; i < 4; ++i) {
                double dNx = Ti(1, i), dNy = Ti(2, i), dNz = Ti(3, i);
                B(0, 3 * i) = dNx;
                B(1, 3 * i + 1) = dNy;
                B(2, 3 * i + 2) = dNz;
                B(3, 3 * i + 1) = dNz; B(3, 3 * i + 2) = dNy;
                B(4, 3 * i) = dNz;     B(4, 3 * i + 2) = dNx;
                B(5, 3 * i) = dNy;     B(5, 3 * i + 1) = dNx;
            }
            return V * B.transpose() * getConstitutiveMatrix3D() * B;
        }
        const Node& n1 = nodes[node_id_to_index.at(elem.n1)];
        const Node& n2 = nodes[node_id_to_index.at(elem.n2)];
        const Node& n3 = nodes[node_id_to_index.at(elem.n3)];
        double Area = 0.5 * std::abs(n1.x * (n2.y - n3.y) + n2.x * (n3.y - n1.y) + n3.x * (n1.y - n2.y));
        Eigen::Matrix<double, 3, 6> B;
        B << (n2.y - n3.y), 0.0, (n3.y - n1.y), 0.0, (n1.y - n2.y), 0.0,
             0.0, (n3.x - n2.x), 0.0, (n1.x - n3.x), 0.0, (n2.x - n1.x),
             (n3.x - n2.x), (n2.y - n3.y), (n1.x - n3.x), (n3.y - n1.y), (n2.x - n1.x), (n1.y - n2.y);
        B /= (2.0 * Area);
        return t * Area * B.transpose() * getConstitutiveMatrix() * B;
    }

    void assembleCPU(std::vector<Eigen::Triplet<double>>& triplets) const {
        const int dn = dofPerNode();
        const int npe = nodesPerElement();
        for (const auto& elem : elements) {
            Eigen::MatrixXd Ke = computeElementStiffness(elem);
            int en[4];
            en[0] = node_id_to_index.at(elem.n1);
            en[1] = node_id_to_index.at(elem.n2);
            en[2] = node_id_to_index.at(elem.n3);
            if (is3D) en[3] = node_id_to_index.at(elem.n4);
            for (int i = 0; i < npe; ++i)
                for (int j = 0; j < npe; ++j)
                    for (int r = 0; r < dn; ++r)
                        for (int c = 0; c < dn; ++c)
                            triplets.push_back(Eigen::Triplet<double>(
                                en[i] * dn + r, en[j] * dn + c,
                                Ke(i * dn + r, j * dn + c)));
        }
    }

    void solve(const OpenCL& ocl);
    void exportVTK(const std::string& filename);
};

// ====================== Solusi: GPU + Fallback ======================
void FEMSolver::solve(const OpenCL& ocl) {
    using T = Eigen::Triplet<double>;
    const int dn = dofPerNode();
    const int npe = nodesPerElement();
    const int nn = (int)nodes.size();
    const int ne = (int)elements.size();
    const int total_dofs = nn * dn;
    const long long nt = (long long)npe * npe * dn * dn;

    std::vector<T> triplets;
    double msUp = 0, msKer = 0, msDown = 0;
    bool gpuPath = ocl.ok;

    auto stamp = []() {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };

    if (gpuPath) {
        // --- Siapkan data host ---
        std::vector<double> coords((size_t)nn * dn);
        for (int i = 0; i < nn; ++i) {
            coords[(size_t)i * dn + 0] = nodes[i].x;
            coords[(size_t)i * dn + 1] = nodes[i].y;
            if (dn == 3) coords[(size_t)i * dn + 2] = nodes[i].z;
        }
        std::vector<cl_int> conn((size_t)ne * npe);
        for (int e = 0; e < ne; ++e) {
            conn[(size_t)e * npe + 0] = (cl_int)node_id_to_index.at(elements[e].n1);
            conn[(size_t)e * npe + 1] = (cl_int)node_id_to_index.at(elements[e].n2);
            conn[(size_t)e * npe + 2] = (cl_int)node_id_to_index.at(elements[e].n3);
            if (is3D) conn[(size_t)e * npe + 3] = (cl_int)node_id_to_index.at(elements[e].n4);
        }

        cl_int err = 0;
        cl_mem mCoords = ocl.clCreateBuffer(ocl.context, CL_MEM_READ_ONLY,
            coords.size() * sizeof(double), nullptr, &err);
        cl_mem mConn = ocl.clCreateBuffer(ocl.context, CL_MEM_READ_ONLY,
            conn.size() * sizeof(cl_int), nullptr, &err);
        cl_mem mRows = ocl.clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY,
            (size_t)nt * ne * sizeof(cl_int), nullptr, &err);
        cl_mem mCols = ocl.clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY,
            (size_t)nt * ne * sizeof(cl_int), nullptr, &err);
        cl_mem mVals = ocl.clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY,
            (size_t)nt * ne * sizeof(double), nullptr, &err);
        if (!mCoords || !mConn || !mRows || !mCols || !mVals) {
            std::cerr << "  [GPU] clCreateBuffer gagal -> fallback CPU" << std::endl;
            gpuPath = false;
        }

        if (gpuPath) {
            double t0 = stamp();
            ocl.clEnqueueWriteBuffer(ocl.queue, mCoords, CL_TRUE, 0,
                coords.size() * sizeof(double), coords.data(), 0, nullptr, nullptr);
            ocl.clEnqueueWriteBuffer(ocl.queue, mConn, CL_TRUE, 0,
                conn.size() * sizeof(cl_int), conn.data(), 0, nullptr, nullptr);
            ocl.clFinish(ocl.queue);
            double t1 = stamp();

            cl_int nElem = ne, is3Di = is3D ? 1 : 0;
            double Ee = E, nnu = nu, tt = t;
            ocl.clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &mCoords);
            ocl.clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), &mConn);
            ocl.clSetKernelArg(ocl.kernel, 2, sizeof(cl_mem), &mRows);
            ocl.clSetKernelArg(ocl.kernel, 3, sizeof(cl_mem), &mCols);
            ocl.clSetKernelArg(ocl.kernel, 4, sizeof(cl_mem), &mVals);
            ocl.clSetKernelArg(ocl.kernel, 5, sizeof(cl_int), &nElem);
            ocl.clSetKernelArg(ocl.kernel, 6, sizeof(cl_int), &is3Di);
            ocl.clSetKernelArg(ocl.kernel, 7, sizeof(double), &Ee);
            ocl.clSetKernelArg(ocl.kernel, 8, sizeof(double), &nnu);
            ocl.clSetKernelArg(ocl.kernel, 9, sizeof(double), &tt);

            size_t L = 128;
            if (ocl.maxWorkGroup > 0 && L > ocl.maxWorkGroup) L = ocl.maxWorkGroup;
            size_t G = ((size_t)ne + L - 1) / L * L;
            err = ocl.clEnqueueNDRangeKernel(ocl.queue, ocl.kernel, 1, nullptr, &G, &L, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                std::cerr << "  [GPU] enqueue kernel gagal (" << clErrStr(err) << ") -> fallback CPU" << std::endl;
                gpuPath = false;
            } else {
                ocl.clFinish(ocl.queue);
                double t2 = stamp();

                std::vector<cl_int> rows((size_t)nt * ne), cols((size_t)nt * ne);
                std::vector<double> vals((size_t)nt * ne);
                ocl.clEnqueueReadBuffer(ocl.queue, mRows, CL_TRUE, 0,
                    rows.size() * sizeof(cl_int), rows.data(), 0, nullptr, nullptr);
                ocl.clEnqueueReadBuffer(ocl.queue, mCols, CL_TRUE, 0,
                    cols.size() * sizeof(cl_int), cols.data(), 0, nullptr, nullptr);
                ocl.clEnqueueReadBuffer(ocl.queue, mVals, CL_TRUE, 0,
                    vals.size() * sizeof(double), vals.data(), 0, nullptr, nullptr);
                ocl.clFinish(ocl.queue);
                double t3 = stamp();

                msUp = t1 - t0; msKer = t2 - t1; msDown = t3 - t2;

                triplets.reserve((size_t)nt * ne);
                for (long long q = 0; q < nt * ne; ++q)
                    triplets.push_back(T(rows[q], cols[q], vals[q]));
            }
        }

        if (mCoords) ocl.clReleaseMemObject(mCoords);
        if (mConn)   ocl.clReleaseMemObject(mConn);
        if (mRows)   ocl.clReleaseMemObject(mRows);
        if (mCols)   ocl.clReleaseMemObject(mCols);
        if (mVals)   ocl.clReleaseMemObject(mVals);
    }

    if (!gpuPath) {
        std::cout << "  [Fallback CPU] " << (ocl.err.empty() ? "GPU tidak digunakan" : ocl.err)
                  << std::endl;
        double t0 = stamp();
        assembleCPU(triplets);
        msKer = stamp() - t0;
    }

    // --- Bangun matriks sparse dari triplets ---
    Eigen::SparseMatrix<double> K(total_dofs, total_dofs);
    K.setFromTriplets(triplets.begin(), triplets.end());
    triplets.clear();
    triplets.shrink_to_fit();

    // --- Vektor beban ---
    Eigen::VectorXd F = Eigen::VectorXd::Zero(total_dofs);
    for (const auto& load : loads) {
        int idx = node_id_to_index.at(load.node_id);
        F(idx * dn + (load.dof - 1)) += load.magnitude;
    }

    // --- Kondisi batas (metode penalti) ---
    Eigen::SparseMatrix<double> Kmod = K;
    for (const auto& bc : bcs) {
        int idx = node_id_to_index.at(bc.node_id);
        int g = idx * dn + (bc.dof - 1);
        Kmod.coeffRef(g, g) *= 1.0e10;
        F(g) = 0.0;
    }

    // --- Solve (CPU, Eigen SparseLU) ---
    double ts0 = stamp();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(Kmod);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Error: Faktorisasi LU gagal!" << std::endl;
        return;
    }
    U = solver.solve(F);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Error: Penyelesaian sistem gagal!" << std::endl;
        return;
    }
    double ts1 = stamp();

    if (gpuPath)
        std::printf("  [GPU] Upload %.1f ms | Kernel %.1f ms | Readback %.1f ms | Solve %.1f ms\n",
                    msUp, msKer, msDown, ts1 - ts0);
    else
        std::printf("  [CPU] Perakitan %.1f ms | Solve %.1f ms\n", msKer, ts1 - ts0);
    std::cout << "Penyelesaian persamaan FEM selesai diselesaikan." << std::endl;
}

// ====================== Ekspor VTK ======================
void FEMSolver::exportVTK(const std::string& filename) {
    std::ofstream vtkFile(filename);
    if (!vtkFile.is_open()) {
        std::cerr << "Error: Tidak dapat membuat file VTK!" << std::endl;
        return;
    }
    const int dn = dofPerNode();
    const int npe = nodesPerElement();

    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << (is3D ? "Hasil FEM 3D (C3D4)" : "Hasil FEM 2D (CPS3)") << " - GPU OpenCL\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET UNSTRUCTURED_GRID\n";

    vtkFile << "POINTS " << nodes.size() << " double\n";
    for (const auto& node : nodes)
        vtkFile << node.x << " " << node.y << " " << node.z << "\n";

    vtkFile << "CELLS " << elements.size() << " "
            << elements.size() * (npe + 1) << "\n";
    for (const auto& elem : elements) {
        int i1 = node_id_to_index.at(elem.n1);
        int i2 = node_id_to_index.at(elem.n2);
        int i3 = node_id_to_index.at(elem.n3);
        if (is3D) {
            int i4 = node_id_to_index.at(elem.n4);
            vtkFile << "4 " << i1 << " " << i2 << " " << i3 << " " << i4 << "\n";
        } else {
            vtkFile << "3 " << i1 << " " << i2 << " " << i3 << "\n";
        }
    }

    vtkFile << "CELL_TYPES " << elements.size() << "\n";
    for (size_t i = 0; i < elements.size(); ++i)
        vtkFile << (is3D ? 10 : 5) << "\n";

    vtkFile << "POINT_DATA " << nodes.size() << "\n";
    vtkFile << "VECTORS Displacement double\n";
    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        if (is3D)
            vtkFile << U(idx * 3) << " " << U(idx * 3 + 1) << " " << U(idx * 3 + 2) << "\n";
        else
            vtkFile << U(idx * 2) << " " << U(idx * 2 + 1) << " 0.0\n";
    }

    vtkFile << "SCALARS DisplacementMagnitude double 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        double mag = 0.0;
        for (int d = 0; d < dn; ++d)
            mag += U(idx * dn + d) * U(idx * dn + d);
        vtkFile << std::sqrt(mag) << "\n";
    }

    vtkFile.close();
    std::cout << "File VTK berhasil dibuat: " << filename << std::endl;
}

// ====================== Main ======================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Penggunaan: " << argv[0]
                  << " <file_input.inp> <file_output.vtk>" << std::endl;
        return 1;
    }
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];

    auto totalStart = std::chrono::steady_clock::now();

    // Inisialisasi OpenCL (dinamis, tanpa CUDA)
    OpenCL ocl;
    if (loadOpenCL(ocl) && pickDevice(ocl) && setupContextAndKernel(ocl, kSrc))
        ocl.ok = true;

    if (ocl.ok) {
        std::cout << "GPU terdeteksi: " << ocl.devName << std::endl;
        std::cout << "  Platform : " << ocl.platName << " (" << ocl.platVersion << ")" << std::endl;
        std::cout << "  CU       : " << ocl.computeUnits
                  << " | Memori global: " << (ocl.globalMem >> 20) << " MB"
                  << " | Max work-group: " << ocl.maxWorkGroup << std::endl;
    } else {
        std::cout << "OpenCL/GPU tidak tersedia (" << ocl.err << ")" << std::endl;
    }

    FEMSolver solver;
    if (!solver.parseINP(inputFile)) return 1;

    std::cout << "Menjalankan " << (solver.is3D ? "[MODE 3D: C3D4]" : "[MODE 2D: CPS3]")
              << (ocl.ok ? " dengan GPU (OpenCL)" : " dengan CPU (fallback)") << "..." << std::endl;

    solver.solve(ocl);
    solver.exportVTK(outputFile);

    auto totalEnd = std::chrono::steady_clock::now();
    std::cout << "Total waktu eksekusi: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count()
              << " ms" << std::endl;

    releaseOpenCL(ocl);
    return 0;
}
