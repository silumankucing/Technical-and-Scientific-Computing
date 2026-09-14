#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
enum class BCType { DIRICHLET, NEUMANN, ROBIN };

struct BoundaryCondition {
    BCType type = BCType::NEUMANN;
    double val1 = 0.0;
    double val2 = 0.0;
};

struct Config {
    int nx = 150, ny = 100, nz = 20;
    double dx = 0.0005, dy = 0.0005, dz = 0.0002;
    double dt = 0.00001;
    int max_steps = 10000;
    int output_interval = 500;
    double rho = 7850.0;
    double cp = 460.0;
    double k_ref = 45.0;
    double alpha_k = 0.0018;
    double T_ref = 293.15;
    double T_init = 293.15;
    double q_vol = 125000000.0;
    
    std::map<std::string, BoundaryCondition> bcs;
};

// POD Struct untuk GPU Constant Memory
struct DeviceConfig {
    int nx, ny, nz;
    double dx, dy, dz;
    double dt;
    double rho, cp;
    double k_ref, alpha_k, T_ref, q_vol;
    
    // Boundary conditions: 0:DIRICHLET, 1:NEUMANN, 2:ROBIN
    int bc_type[6];    // xmin, xmax, ymin, ymax, zmin, zmax
    double bc_val1[6];
    double bc_val2[6];
};

__constant__ DeviceConfig d_cfg;

// Device Function: Non-linear conductivity k(T)
__device__ inline double calc_k_gpu(double T) {
    return d_cfg.k_ref * (1.0 + d_cfg.alpha_k * (T - d_cfg.T_ref));
}

// 3D Indexing Function
__device__ inline size_t get_idx(int i, int j, int k, int nx, int ny) {
    return static_cast<size_t>(i) + static_cast<size_t>(nx) * (j + ny * k);
}

// -------------------------------------------------------------------
// CUDA KERNEL 1: 3D Stencil Thermal Conduction (Interior Nodes)
// -------------------------------------------------------------------
__global__ void heat_conduction_3d_kernel(const double* __restrict__ T, double* __restrict__ T_next) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = d_cfg.nx;
    int ny = d_cfg.ny;
    int nz = d_cfg.nz;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1) {
        size_t c = get_idx(i, j, k, nx, ny);

        double T_c = T[c];
        double k_c = calc_k_gpu(T_c);

        // Harmonic Mean Thermal Conductivity antar elemen
        double k_px = 2.0 * k_c * calc_k_gpu(T[get_idx(i+1, j, k, nx, ny)]) / (k_c + calc_k_gpu(T[get_idx(i+1, j, k, nx, ny)]));
        double k_nx = 2.0 * k_c * calc_k_gpu(T[get_idx(i-1, j, k, nx, ny)]) / (k_c + calc_k_gpu(T[get_idx(i-1, j, k, nx, ny)]));

        double k_py = 2.0 * k_c * calc_k_gpu(T[get_idx(i, j+1, k, nx, ny)]) / (k_c + calc_k_gpu(T[get_idx(i, j+1, k, nx, ny)]));
        double k_ny = 2.0 * k_c * calc_k_gpu(T[get_idx(i, j-1, k, nx, ny)]) / (k_c + calc_k_gpu(T[get_idx(i, j-1, k, nx, ny)]));

        double k_pz = 2.0 * k_c * calc_k_gpu(T[get_idx(i, j, k+1, nx, ny)]) / (k_c + calc_k_gpu(T[get_idx(i, j, k+1, nx, ny)]));
        double k_nz = 2.0 * k_c * calc_k_gpu(T[get_idx(i, j, k-1, nx, ny)]) / (k_c + calc_k_gpu(T[get_idx(i, j, k-1, nx, ny)]));

        // Central Finite Difference Second Derivatives
        double d2T_dx2 = (k_px * (T[get_idx(i+1, j, k, nx, ny)] - T_c) - k_nx * (T_c - T[get_idx(i-1, j, k, nx, ny)])) / (d_cfg.dx * d_cfg.dx);
        double d2T_dy2 = (k_py * (T[get_idx(i, j+1, k, nx, ny)] - T_c) - k_ny * (T_c - T[get_idx(i, j-1, k, nx, ny)])) / (d_cfg.dy * d_cfg.dy);
        double d2T_dz2 = (k_pz * (T[get_idx(i, j, k+1, nx, ny)] - T_c) - k_nz * (T_c - T[get_idx(i, j, k-1, nx, ny)])) / (d_cfg.dz * d_cfg.dz);

        // Explicit Time Stepping
        T_next[c] = T_c + (d_cfg.dt / (d_cfg.rho * d_cfg.cp)) * (d2T_dx2 + d2T_dy2 + d2T_dz2 + d_cfg.q_vol);
    }
}

// -------------------------------------------------------------------
// CUDA KERNEL 2: Parallel Boundary Conditions Enforcer
// -------------------------------------------------------------------
__global__ void apply_bc_kernel(double* __restrict__ T_next) {
    int idx_2d_1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_2d_2 = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = d_cfg.nx;
    int ny = d_cfg.ny;
    int nz = d_cfg.nz;

    // Boundary Faces X-Min dan X-Max
    if (idx_2d_1 < ny && idx_2d_2 < nz) {
        int j = idx_2d_1, k = idx_2d_2;
        // X-Min
        if (d_cfg.bc_type[0] == 0) T_next[get_idx(0, j, k, nx, ny)] = d_cfg.bc_val1[0];
        else if (d_cfg.bc_type[0] == 1) {
            double k_val = calc_k_gpu(T_next[get_idx(1, j, k, nx, ny)]);
            T_next[get_idx(0, j, k, nx, ny)] = T_next[get_idx(1, j, k, nx, ny)] + (d_cfg.bc_val1[0] * d_cfg.dx / k_val);
        } else if (d_cfg.bc_type[0] == 2) {
            double h = d_cfg.bc_val1[0], T_inf = d_cfg.bc_val2[0];
            double k_val = calc_k_gpu(T_next[get_idx(1, j, k, nx, ny)]);
            T_next[get_idx(0, j, k, nx, ny)] = (k_val * T_next[get_idx(1, j, k, nx, ny)] + h * d_cfg.dx * T_inf) / (k_val + h * d_cfg.dx);
        }
        // X-Max
        if (d_cfg.bc_type[1] == 0) T_next[get_idx(nx-1, j, k, nx, ny)] = d_cfg.bc_val1[1];
        else if (d_cfg.bc_type[1] == 1) {
            double k_val = calc_k_gpu(T_next[get_idx(nx-2, j, k, nx, ny)]);
            T_next[get_idx(nx-1, j, k, nx, ny)] = T_next[get_idx(nx-2, j, k, nx, ny)] + (d_cfg.bc_val1[1] * d_cfg.dx / k_val);
        } else if (d_cfg.bc_type[1] == 2) {
            double h = d_cfg.bc_val1[1], T_inf = d_cfg.bc_val2[1];
            double k_val = calc_k_gpu(T_next[get_idx(nx-2, j, k, nx, ny)]);
            T_next[get_idx(nx-1, j, k, nx, ny)] = (k_val * T_next[get_idx(nx-2, j, k, nx, ny)] + h * d_cfg.dx * T_inf) / (k_val + h * d_cfg.dx);
        }
    }

    // Boundary Faces Y-Min dan Y-Max
    if (idx_2d_1 < nx && idx_2d_2 < nz) {
        int i = idx_2d_1, k = idx_2d_2;
        // Y-Min
        if (d_cfg.bc_type[2] == 0) T_next[get_idx(i, 0, k, nx, ny)] = d_cfg.bc_val1[2];
        else if (d_cfg.bc_type[2] == 1) {
            double k_val = calc_k_gpu(T_next[get_idx(i, 1, k, nx, ny)]);
            T_next[get_idx(i, 0, k, nx, ny)] = T_next[get_idx(i, 1, k, nx, ny)] + (d_cfg.bc_val1[2] * d_cfg.dy / k_val);
        } else if (d_cfg.bc_type[2] == 2) {
            double h = d_cfg.bc_val1[2], T_inf = d_cfg.bc_val2[2];
            double k_val = calc_k_gpu(T_next[get_idx(i, 1, k, nx, ny)]);
            T_next[get_idx(i, 0, k, nx, ny)] = (k_val * T_next[get_idx(i, 1, k, nx, ny)] + h * d_cfg.dy * T_inf) / (k_val + h * d_cfg.dy);
        }
        // Y-Max
        if (d_cfg.bc_type[3] == 0) T_next[get_idx(i, ny-1, k, nx, ny)] = d_cfg.bc_val1[3];
        else if (d_cfg.bc_type[3] == 1) {
            double k_val = calc_k_gpu(T_next[get_idx(i, ny-2, k, nx, ny)]);
            T_next[get_idx(i, ny-1, k, nx, ny)] = T_next[get_idx(i, ny-2, k, nx, ny)] + (d_cfg.bc_val1[3] * d_cfg.dy / k_val);
        } else if (d_cfg.bc_type[3] == 2) {
            double h = d_cfg.bc_val1[3], T_inf = d_cfg.bc_val2[3];
            double k_val = calc_k_gpu(T_next[get_idx(i, ny-2, k, nx, ny)]);
            T_next[get_idx(i, ny-1, k, nx, ny)] = (k_val * T_next[get_idx(i, ny-2, k, nx, ny)] + h * d_cfg.dy * T_inf) / (k_val + h * d_cfg.dy);
        }
    }

    // Boundary Faces Z-Min dan Z-Max
    if (idx_2d_1 < nx && idx_2d_2 < ny) {
        int i = idx_2d_1, j = idx_2d_2;
        // Z-Min
        if (d_cfg.bc_type[4] == 0) T_next[get_idx(i, j, 0, nx, ny)] = d_cfg.bc_val1[4];
        else if (d_cfg.bc_type[4] == 1) {
            double k_val = calc_k_gpu(T_next[get_idx(i, j, 1, nx, ny)]);
            T_next[get_idx(i, j, 0, nx, ny)] = T_next[get_idx(i, j, 1, nx, ny)] + (d_cfg.bc_val1[4] * d_cfg.dz / k_val);
        } else if (d_cfg.bc_type[4] == 2) {
            double h = d_cfg.bc_val1[4], T_inf = d_cfg.bc_val2[4];
            double k_val = calc_k_gpu(T_next[get_idx(i, j, 1, nx, ny)]);
            T_next[get_idx(i, j, 0, nx, ny)] = (k_val * T_next[get_idx(i, j, 1, nx, ny)] + h * d_cfg.dz * T_inf) / (k_val + h * d_cfg.dz);
        }
        // Z-Max
        if (d_cfg.bc_type[5] == 0) T_next[get_idx(i, j, nz-1, nx, ny)] = d_cfg.bc_val1[5];
        else if (d_cfg.bc_type[5] == 1) {
            double k_val = calc_k_gpu(T_next[get_idx(i, j, nz-2, nx, ny)]);
            T_next[get_idx(i, j, nz-1, nx, ny)] = T_next[get_idx(i, j, nz-2, nx, ny)] + (d_cfg.bc_val1[5] * d_cfg.dz / k_val);
        } else if (d_cfg.bc_type[5] == 2) {
            double h = d_cfg.bc_val1[5], T_inf = d_cfg.bc_val2[5];
            double k_val = calc_k_gpu(T_next[get_idx(i, j, nz-2, nx, ny)]);
            T_next[get_idx(i, j, nz-1, nx, ny)] = (k_val * T_next[get_idx(i, j, nz-2, nx, ny)] + h * d_cfg.dz * T_inf) / (k_val + h * d_cfg.dz);
        }
    }
}

// Host Config Parser
Config parseInputFile(const std::string& filename) {
    Config cfg;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(file, line)) {
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);
        std::istringstream is_line(line);
        std::string key, eq, val;
        if (is_line >> key >> eq >> val && eq == "=") kv[key] = val;
    }
    auto get_double = [&](const std::string& k, double d) { return kv.count(k) ? std::stod(kv[k]) : d; };
    auto get_int = [&](const std::string& k, int d) { return kv.count(k) ? std::stoi(kv[k]) : d; };

    cfg.nx = get_int("nx", 150); cfg.ny = get_int("ny", 100); cfg.nz = get_int("nz", 20);
    cfg.dx = get_double("dx", 0.0005); cfg.dy = get_double("dy", 0.0005); cfg.dz = get_double("dz", 0.0002);
    cfg.dt = get_double("dt", 0.00001); cfg.max_steps = get_int("max_steps", 10000);
    cfg.output_interval = get_int("output_interval", 500);
    cfg.rho = get_double("rho", 7850.0); cfg.cp = get_double("cp", 460.0);
    cfg.k_ref = get_double("k_ref", 45.0); cfg.alpha_k = get_double("alpha_k", 0.0018);
    cfg.T_ref = get_double("T_ref", 293.15); cfg.T_init = get_double("T_init", 293.15);
    cfg.q_vol = get_double("q_vol", 125000000.0);

    std::vector<std::string> faces = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    for (const auto& face : faces) {
        BoundaryCondition bc;
        std::string type_str = kv["bc_" + face + "_type"];
        if (type_str == "DIRICHLET") bc.type = BCType::DIRICHLET;
        else if (type_str == "NEUMANN") bc.type = BCType::NEUMANN;
        else if (type_str == "ROBIN") bc.type = BCType::ROBIN;
        bc.val1 = get_double("bc_" + face + "_val1", 0.0);
        bc.val2 = get_double("bc_" + face + "_val2", 0.0);
        cfg.bcs[face] = bc;
    }
    return cfg;
}

void writeVTK(int step, const Config& cfg, const std::vector<double>& T) {
    std::string filename = "thermal_gpu_output_" + std::to_string(step) + ".vtk";
    std::ofstream out(filename);
    if (!out.is_open()) return;

    out << "# vtk DataFile Version 3.0\nCUDA GPU Thermal Conduction\nASCII\nDATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << cfg.nx << " " << cfg.ny << " " << cfg.nz << "\n";
    out << "ORIGIN 0 0 0\nSPACING " << cfg.dx << " " << cfg.dy << " " << cfg.dz << "\n";
    out << "POINT_DATA " << cfg.nx * cfg.ny * cfg.nz << "\nSCALARS Temperature double 1\nLOOKUP_TABLE default\n";
    for (size_t i = 0; i < T.size(); ++i) out << T[i] << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file.inp>\n";
        return 1;
    }

    Config cfg = parseInputFile(argv[1]);
    const size_t total_nodes = static_cast<size_t>(cfg.nx) * cfg.ny * cfg.nz;
    const size_t bytes = total_nodes * sizeof(double);

    DeviceConfig h_dev_cfg;
    h_dev_cfg.nx = cfg.nx; h_dev_cfg.ny = cfg.ny; h_dev_cfg.nz = cfg.nz;
    h_dev_cfg.dx = cfg.dx; h_dev_cfg.dy = cfg.dy; h_dev_cfg.dz = cfg.dz;
    h_dev_cfg.dt = cfg.dt; h_dev_cfg.rho = cfg.rho; h_dev_cfg.cp = cfg.cp;
    h_dev_cfg.k_ref = cfg.k_ref; h_dev_cfg.alpha_k = cfg.alpha_k;
    h_dev_cfg.T_ref = cfg.T_ref; h_dev_cfg.q_vol = cfg.q_vol;

    std::vector<std::string> faces = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    for (int i = 0; i < 6; ++i) {
        const auto& bc = cfg.bcs[faces[i]];
        h_dev_cfg.bc_type[i] = static_cast<int>(bc.type);
        h_dev_cfg.bc_val1[i] = bc.val1;
        h_dev_cfg.bc_val2[i] = bc.val2;
    }

    cudaMemcpyToSymbol(d_cfg, &h_dev_cfg, sizeof(DeviceConfig));

    std::vector<double> h_T(total_nodes, cfg.T_init);
    double *d_T, *d_T_next;
    cudaMalloc(&d_T, bytes);
    cudaMalloc(&d_T_next, bytes);
    cudaMemcpy(d_T, h_T.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_next, h_T.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8, 4); // 256 threads per block
    dim3 numBlocks((cfg.nx + 7)/8, (cfg.ny + 7)/8, (cfg.nz + 3)/4);
    dim3 bcThreads(16, 16);
    dim3 bcBlocks((std::max(cfg.nx, cfg.ny) + 15)/16, (std::max(cfg.ny, cfg.nz) + 15)/16);

    std::cout << "Running CUDA Simulation on GPU (" << total_nodes << " nodes)...\n";
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step <= cfg.max_steps; ++step) {
        if (step % cfg.output_interval == 0) {
            cudaMemcpy(h_T.data(), d_T, bytes, cudaMemcpyDeviceToHost);
            writeVTK(step, cfg, h_T);
        }
        heat_conduction_3d_kernel<<<numBlocks, threadsPerBlock>>>(d_T, d_T_next);
        apply_bc_kernel<<<bcBlocks, bcThreads>>>(d_T_next);
        std::swap(d_T, d_T_next);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Simulation Completed in: " << std::chrono::duration<double>(end - start).count() << " seconds.\n";

    cudaFree(d_T);
    cudaFree(d_T_next);
    return 0;
}