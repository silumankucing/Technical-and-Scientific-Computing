#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

enum class BCType { DIRICHLET, NEUMANN, ROBIN };

struct BoundaryCondition {
    BCType type = BCType::NEUMANN;
    double val1 = 0.0; // T_fixed (Dirichlet), flux (Neumann), h (Robin)
    double val2 = 0.0; // T_inf (Robin)
};

struct Config {
    int nx = 50, ny = 50, nz = 50;
    double dx = 0.001, dy = 0.001, dz = 0.001;
    double dt = 0.0001;
    int max_steps = 1000;
    int output_interval = 100;
    double rho = 7850.0;
    double cp = 490.0;
    double k_ref = 50.0;
    double alpha_k = 0.0;
    double T_ref = 300.0;
    double T_init = 300.0;
    double q_vol = 0.0;
    
    std::map<std::string, BoundaryCondition> bcs;
};

// Parser untuk membaca file input.inp
Config parseInputFile(const std::string& filename) {
    Config cfg;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Gagal membuka file input: " + filename);
    }

    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(file, line)) {
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);
        
        std::istringstream is_line(line);
        std::string key, eq, val;
        if (is_line >> key >> eq >> val && eq == "=") {
            kv[key] = val;
        }
    }

    auto get_double = [&](const std::string& key, double def) {
        return kv.count(key) ? std::stod(kv[key]) : def;
    };
    auto get_int = [&](const std::string& key, int def) {
        return kv.count(key) ? std::stoi(kv[key]) : def;
    };

    cfg.nx = get_int("nx", 50);
    cfg.ny = get_int("ny", 50);
    cfg.nz = get_int("nz", 50);
    cfg.dx = get_double("dx", 0.001);
    cfg.dy = get_double("dy", 0.001);
    cfg.dz = get_double("dz", 0.001);
    cfg.dt = get_double("dt", 0.0001);
    cfg.max_steps = get_int("max_steps", 1000);
    cfg.output_interval = get_int("output_interval", 100);
    cfg.rho = get_double("rho", 7850.0);
    cfg.cp = get_double("cp", 490.0);
    cfg.k_ref = get_double("k_ref", 50.0);
    cfg.alpha_k = get_double("alpha_k", 0.0);
    cfg.T_ref = get_double("T_ref", 300.0);
    cfg.T_init = get_double("T_init", 300.0);
    cfg.q_vol = get_double("q_vol", 0.0);

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
    std::string filename = "thermal_output_" + std::to_string(step) + ".vtk";
    std::ofstream out(filename);
    if (!out.is_open()) return;

    out << "# vtk DataFile Version 3.0\n";
    out << "Thermal Conduction Output\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << cfg.nx << " " << cfg.ny << " " << cfg.nz << "\n";
    out << "ORIGIN 0 0 0\n";
    out << "SPACING " << cfg.dx << " " << cfg.dy << " " << cfg.dz << "\n";
    out << "POINT_DATA " << cfg.nx * cfg.ny * cfg.nz << "\n";
    out << "SCALARS Temperature double 1\n";
    out << "LOOKUP_TABLE default\n";

    for (size_t i = 0; i < T.size(); ++i) {
        out << T[i] << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Error: File input tidak ditentukan!\n";
        std::cerr << "Penggunaan: " << argv[0] << " <input_file.inp>\n";
        std::cerr << "Contoh: ./Heat_Conduction input.inp\n";
        return 1;
    }

    std::string input_filename = argv[1];

    std::cout << "========================================================\n";
    std::cout << "  3D Thermal Conduction Simulator (OpenMP)\n";
    std::cout << "  Membaca konfigurasi dari: " << input_filename << "\n";
    std::cout << "========================================================\n";

    Config cfg;
    try {
        cfg = parseInputFile(input_filename);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    const int nx = cfg.nx, ny = cfg.ny, nz = cfg.nz;
    const size_t total_nodes = static_cast<size_t>(nx) * ny * nz;
    
    std::vector<double> T(total_nodes, cfg.T_init);
    std::vector<double> T_next(total_nodes, cfg.T_init);

    auto idx = [nx, ny](int i, int j, int k) -> size_t {
        return static_cast<size_t>(i) + nx * (j + ny * k);
    };

    auto calc_k = [&cfg](double Temp) {
        return cfg.k_ref * (1.0 + cfg.alpha_k * (Temp - cfg.T_ref));
    };

    double alpha_max = (cfg.k_ref * (1.0 + std::abs(cfg.alpha_k) * 1000.0)) / (cfg.rho * cfg.cp);
    double dt_max = 0.5 / (alpha_max * (1.0/(cfg.dx*cfg.dx) + 1.0/(cfg.dy*cfg.dy) + 1.0/(cfg.dz*cfg.dz)));
    
    std::cout << "Jumlah Thread OpenMP: " << omp_get_max_threads() << "\n";
    std::cout << "Ukuran Grid: " << nx << " x " << ny << " x " << nz << " (" << total_nodes << " node)\n";
    std::cout << "Time step (dt): " << cfg.dt << " s (Batas maksimal dt: " << dt_max << " s)\n";
    
    if (cfg.dt > dt_max) {
        std::cout << "PERINGATAN: dt melebihi batas Von Neumann, simulasi bisa mengalami divergensi!\n";
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step <= cfg.max_steps; ++step) {

        if (step % cfg.output_interval == 0) {
            writeVTK(step, cfg, T);
        }

        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    size_t c = idx(i, j, k);

                    double T_c = T[c];
                    double k_c = calc_k(T_c);

                    double k_px = 2.0 * k_c * calc_k(T[idx(i+1, j, k)]) / (k_c + calc_k(T[idx(i+1, j, k)]));
                    double k_nx = 2.0 * k_c * calc_k(T[idx(i-1, j, k)]) / (k_c + calc_k(T[idx(i-1, j, k)]));
                    
                    double k_py = 2.0 * k_c * calc_k(T[idx(i, j+1, k)]) / (k_c + calc_k(T[idx(i, j+1, k)]));
                    double k_ny = 2.0 * k_c * calc_k(T[idx(i, j-1, k)]) / (k_c + calc_k(T[idx(i, j-1, k)]));
                    
                    double k_pz = 2.0 * k_c * calc_k(T[idx(i, j, k+1)]) / (k_c + calc_k(T[idx(i, j, k+1)]));
                    double k_nz = 2.0 * k_c * calc_k(T[idx(i, j, k-1)]) / (k_c + calc_k(T[idx(i, j, k-1)]));

                    double d2T_dx2 = (k_px * (T[idx(i+1, j, k)] - T_c) - k_nx * (T_c - T[idx(i-1, j, k)])) / (cfg.dx * cfg.dx);
                    double d2T_dy2 = (k_py * (T[idx(i, j+1, k)] - T_c) - k_ny * (T_c - T[idx(i, j-1, k)])) / (cfg.dy * cfg.dy);
                    double d2T_dz2 = (k_pz * (T[idx(i, j, k+1)] - T_c) - k_nz * (T_c - T[idx(i, j, k-1)])) / (cfg.dz * cfg.dz);

                    T_next[c] = T_c + (cfg.dt / (cfg.rho * cfg.cp)) * (d2T_dx2 + d2T_dy2 + d2T_dz2 + cfg.q_vol);
                }
            }
        }

        #pragma omp parallel
        {
            #pragma omp for collapse(2) schedule(static)
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                    const auto& bc_xmin = cfg.bcs.at("xmin");
                    if (bc_xmin.type == BCType::DIRICHLET) {
                        T_next[idx(0, j, k)] = bc_xmin.val1;
                    } else if (bc_xmin.type == BCType::NEUMANN) {
                        T_next[idx(0, j, k)] = T_next[idx(1, j, k)] + (bc_xmin.val1 * cfg.dx / calc_k(T_next[idx(1, j, k)]));
                    } else if (bc_xmin.type == BCType::ROBIN) {
                        double h = bc_xmin.val1, T_inf = bc_xmin.val2, k_val = calc_k(T_next[idx(1, j, k)]);
                        T_next[idx(0, j, k)] = (k_val * T_next[idx(1, j, k)] + h * cfg.dx * T_inf) / (k_val + h * cfg.dx);
                    }

                    const auto& bc_xmax = cfg.bcs.at("xmax");
                    if (bc_xmax.type == BCType::DIRICHLET) {
                        T_next[idx(nx-1, j, k)] = bc_xmax.val1;
                    } else if (bc_xmax.type == BCType::NEUMANN) {
                        T_next[idx(nx-1, j, k)] = T_next[idx(nx-2, j, k)] + (bc_xmax.val1 * cfg.dx / calc_k(T_next[idx(nx-2, j, k)]));
                    } else if (bc_xmax.type == BCType::ROBIN) {
                        double h = bc_xmax.val1, T_inf = bc_xmax.val2, k_val = calc_k(T_next[idx(nx-2, j, k)]);
                        T_next[idx(nx-1, j, k)] = (k_val * T_next[idx(nx-2, j, k)] + h * cfg.dx * T_inf) / (k_val + h * cfg.dx);
                    }
                }
            }

            #pragma omp for collapse(2) schedule(static)
            for (int k = 0; k < nz; ++k) {
                for (int i = 0; i < nx; ++i) {
                    const auto& bc_ymin = cfg.bcs.at("ymin");
                    if (bc_ymin.type == BCType::DIRICHLET) {
                        T_next[idx(i, 0, k)] = bc_ymin.val1;
                    } else if (bc_ymin.type == BCType::NEUMANN) {
                        T_next[idx(i, 0, k)] = T_next[idx(i, 1, k)] + (bc_ymin.val1 * cfg.dy / calc_k(T_next[idx(i, 1, k)]));
                    } else if (bc_ymin.type == BCType::ROBIN) {
                        double h = bc_ymin.val1, T_inf = bc_ymin.val2, k_val = calc_k(T_next[idx(i, 1, k)]);
                        T_next[idx(i, 0, k)] = (k_val * T_next[idx(i, 1, k)] + h * cfg.dy * T_inf) / (k_val + h * cfg.dy);
                    }

                    const auto& bc_ymax = cfg.bcs.at("ymax");
                    if (bc_ymax.type == BCType::DIRICHLET) {
                        T_next[idx(i, ny-1, k)] = bc_ymax.val1;
                    } else if (bc_ymax.type == BCType::NEUMANN) {
                        T_next[idx(i, ny-1, k)] = T_next[idx(i, ny-2, k)] + (bc_ymax.val1 * cfg.dy / calc_k(T_next[idx(i, ny-2, k)]));
                    } else if (bc_ymax.type == BCType::ROBIN) {
                        double h = bc_ymax.val1, T_inf = bc_ymax.val2, k_val = calc_k(T_next[idx(i, ny-2, k)]);
                        T_next[idx(i, ny-1, k)] = (k_val * T_next[idx(i, ny-2, k)] + h * cfg.dy * T_inf) / (k_val + h * cfg.dy);
                    }
                }
            }

            #pragma omp for collapse(2) schedule(static)
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const auto& bc_zmin = cfg.bcs.at("zmin");
                    if (bc_zmin.type == BCType::DIRICHLET) {
                        T_next[idx(i, j, 0)] = bc_zmin.val1;
                    } else if (bc_zmin.type == BCType::NEUMANN) {
                        T_next[idx(i, j, 0)] = T_next[idx(i, j, 1)] + (bc_zmin.val1 * cfg.dz / calc_k(T_next[idx(i, j, 1)]));
                    } else if (bc_zmin.type == BCType::ROBIN) {
                        double h = bc_zmin.val1, T_inf = bc_zmin.val2, k_val = calc_k(T_next[idx(i, j, 1)]);
                        T_next[idx(i, j, 0)] = (k_val * T_next[idx(i, j, 1)] + h * cfg.dz * T_inf) / (k_val + h * cfg.dz);
                    }

                    const auto& bc_zmax = cfg.bcs.at("zmax");
                    if (bc_zmax.type == BCType::DIRICHLET) {
                        T_next[idx(i, j, nz-1)] = bc_zmax.val1;
                    } else if (bc_zmax.type == BCType::NEUMANN) {
                        T_next[idx(i, j, nz-1)] = T_next[idx(i, j, nz-2)] + (bc_zmax.val1 * cfg.dz / calc_k(T_next[idx(i, j, nz-2)]));
                    } else if (bc_zmax.type == BCType::ROBIN) {
                        double h = bc_zmax.val1, T_inf = bc_zmax.val2, k_val = calc_k(T_next[idx(i, j, nz-2)]);
                        T_next[idx(i, j, nz-1)] = (k_val * T_next[idx(i, j, nz-2)] + h * cfg.dz * T_inf) / (k_val + h * cfg.dz);
                    }
                }
            }
        }

        T.swap(T_next);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Simulasi selesai dalam waktu " << elapsed.count() << " detik.\n";
    std::cout << "File ekspor VTK berhasil dibuat.\n";

    return 0;
}