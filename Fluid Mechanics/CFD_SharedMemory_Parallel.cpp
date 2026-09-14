#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <omp.h>

// Struct Node 3D universal
struct Node {
    int id;
    double x, y, z;
};

// Struct Cell dinamis (2D Quad/Tri & 3D Hex/Tet)
struct Cell {
    int id;
    int dim;                   // 2 untuk 2D, 3 untuk 3D
    std::vector<int> node_ids; // Fleksibel menampung N-node
    double xc, yc, zc;         // Pusat sel
    double volume;             // Luas (2D) atau Volume (3D)
};

// Struct Boundary Element
struct BoundaryElement {
    int id;
    int physical_tag;          // Tag boundary (1: inlet, 2: outflow, 3: wall)
    std::vector<int> node_ids;
};

class UnifiedCFDSolver {
public:
    std::vector<Node> nodes;
    std::vector<Cell> cells;
    std::vector<BoundaryElement> boundaries;
    
    int domain_dim = 2; // Otomatis terdeteksi (2 atau 3)

    // Properti Fluida (Air)
    const double rho = 998.2;    // kg/m^3
    const double nu = 1.004e-6;  // m^2/s

    // Field Variabel 3D Universal
    std::vector<double> u, v, w, p;
    std::vector<double> u_next, v_next, w_next;

    bool loadMesh(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Gagal membuka file " << filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line == "$Nodes") {
                int num_nodes;
                file >> num_nodes;
                nodes.resize(num_nodes + 1);
                for (int i = 0; i < num_nodes; ++i) {
                    int id;
                    double x, y, z;
                    file >> id >> x >> y >> z;
                    nodes[id] = {id, x, y, z};
                }
            } else if (line == "$Elements") {
                int num_elements;
                file >> num_elements;
                for (int i = 0; i < num_elements; ++i) {
                    int id, type, num_tags, ptag, etag;
                    file >> id >> type >> num_tags >> ptag >> etag;

                    // Mengabaikan tag tambahan jika ada
                    for (int t = 2; t < num_tags; ++t) {
                        int dummy_tag;
                        file >> dummy_tag;
                    }

                    // Handlers Tipe Elemen Gmsh
                    if (type == 1 || type == 2) {
                        // Line atau Triangle boundary
                        int num_pts = (type == 1) ? 2 : 3;
                        BoundaryElement b;
                        b.id = id;
                        b.physical_tag = ptag;
                        b.node_ids.resize(num_pts);
                        for (int k = 0; k < num_pts; ++k) file >> b.node_ids[k];
                        boundaries.push_back(b);

                    } else if (type == 3) { 
                        // Quad Element (bisa boundary 3D atau domain 2D)
                        if (domain_dim == 3) {
                            BoundaryElement b;
                            b.id = id;
                            b.physical_tag = ptag;
                            b.node_ids.resize(4);
                            for (int k = 0; k < 4; ++k) file >> b.node_ids[k];
                            boundaries.push_back(b);
                        } else {
                            parseCell(id, 2, 4, file);
                        }

                    } else if (type == 5) { 
                        // Elemen Domain 3D Hexahedron (8 Node)
                        domain_dim = 3;
                        parseCell(id, 3, 8, file);

                    } else if (type == 4) {
                        // Elemen Domain 3D Tetrahedron (4 Node)
                        domain_dim = 3;
                        parseCell(id, 3, 4, file);

                    } else {
                        // Skip tipe elemen tidak dikenal
                        std::string dummy;
                        std::getline(file, dummy);
                    }
                }
            }
        }

        // Alokasi memori field variabel
        size_t n = cells.size();
        u.assign(n, 0.0); v.assign(n, 0.0); w.assign(n, 0.0); p.assign(n, 0.0);
        u_next.assign(n, 0.0); v_next.assign(n, 0.0); w_next.assign(n, 0.0);

        std::cout << "[Mesh Loaded Successfully]\n"
                  << "  - Dimensi Domain : " << domain_dim << "D\n"
                  << "  - Jumlah Node    : " << nodes.size() - 1 << "\n"
                  << "  - Jumlah Sel     : " << cells.size() << "\n"
                  << "  - Elemen Boundary: " << boundaries.size() << std::endl;
        return true;
    }

private:
    void parseCell(int id, int dim, int num_nodes, std::ifstream& file) {
        Cell c;
        c.id = id;
        c.dim = dim;
        c.node_ids.resize(num_nodes);
        
        double sum_x = 0, sum_y = 0, sum_z = 0;
        for (int k = 0; k < num_nodes; ++k) {
            file >> c.node_ids[k];
            sum_x += nodes[c.node_ids[k]].x;
            sum_y += nodes[c.node_ids[k]].y;
            sum_z += nodes[c.node_ids[k]].z;
        }

        c.xc = sum_x / num_nodes;
        c.yc = sum_y / num_nodes;
        c.zc = sum_z / num_nodes;
        c.volume = 1.0; 
        cells.push_back(c);
    }

public:
    void solveParallel(int max_steps, double dt) {
        std::cout << "\nMemulai OpenMP Solver (" << domain_dim << "D Mode) dengan " 
                  << omp_get_max_threads() << " threads..." << std::endl;

        for (int step = 0; step < max_steps; ++step) {

            // 1. Solusi Momentum Pararel (Mendukung Kecepatan u, v, w)
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < cells.size(); ++i) {
                double du_dx = 0.01 * u[i];
                double dv_dy = 0.01 * v[i];
                double dw_dz = (domain_dim == 3) ? 0.01 * w[i] : 0.0;

                u_next[i] = u[i] + dt * (-u[i] * du_dx);
                v_next[i] = v[i] + dt * (-v[i] * dv_dy);
                if (domain_dim == 3) {
                    w_next[i] = w[i] + dt * (-w[i] * dw_dz);
                }
            }

            // 2. Aplikasi Boundary Conditions Paralel
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < cells.size(); ++i) {
                // Inlet Condition (X mendekati 0) -> Kecepatan masuk U = 1.0 m/s
                if (cells[i].xc < 0.05) {
                    u_next[i] = 1.0;
                    v_next[i] = 0.0;
                    w_next[i] = 0.0;
                }
                // No-Slip Wall Condition
                if (cells[i].yc < 0.05 || (domain_dim == 3 && cells[i].zc < 0.05)) {
                    u_next[i] = 0.0;
                    v_next[i] = 0.0;
                    w_next[i] = 0.0;
                }
            }

            // 3. Update Variabel Utama
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < cells.size(); ++i) {
                u[i] = u_next[i];
                v[i] = v_next[i];
                if (domain_dim == 3) w[i] = w_next[i];
            }

            if (step % 100 == 0) {
                #pragma omp single
                {
                    std::cout << "Step: " << step << " / " << max_steps << " selesai." << std::endl;
                }
            }
        }
        std::cout << "Simulasi CFD " << domain_dim << "D Berhasil!" << std::endl;
    }

    // Fungsi Export Hasil ke Format VTK ASCII (ParaView Compatible)
    void writeVTK(const std::string& output_filename) {
        std::ofstream vtk_file(output_filename);
        if (!vtk_file.is_open()) {
            std::cerr << "Error: Tidak dapat membuat file VTK " << output_filename << std::endl;
            return;
        }

        vtk_file << "# vtk DataFile Version 3.0\n";
        vtk_file << "CFD Result Output\n";
        vtk_file << "ASCII\n";
        vtk_file << "DATASET UNSTRUCTURED_GRID\n";

        // Tulis Nodes
        vtk_file << "POINTS " << nodes.size() - 1 << " double\n";
        for (size_t i = 1; i < nodes.size(); ++i) {
            vtk_file << nodes[i].x << " " << nodes[i].y << " " << nodes[i].z << "\n";
        }

        // Tulis Cells
        size_t total_cell_entries = 0;
        for (const auto& c : cells) {
            total_cell_entries += (c.node_ids.size() + 1);
        }

        vtk_file << "CELLS " << cells.size() << " " << total_cell_entries << "\n";
        for (const auto& c : cells) {
            vtk_file << c.node_ids.size();
            for (int nid : c.node_ids) {
                vtk_file << " " << (nid - 1); // Indeks 0-based untuk VTK
            }
            vtk_file << "\n";
        }

        // Tulis Cell Types
        vtk_file << "CELL_TYPES " << cells.size() << "\n";
        for (const auto& c : cells) {
            if (c.dim == 2) {
                vtk_file << "9\n";  // VTK_QUAD = 9
            } else {
                vtk_file << (c.node_ids.size() == 8 ? "12\n" : "10\n"); // VTK_HEXAHEDRON = 12, VTK_TETRA = 10
            }
        }

        // Tulis Cell Data (Kecepatan dan Tekanan di pusat sel)
        vtk_file << "CELL_DATA " << cells.size() << "\n";
        
        // Vektor Kecepatan
        vtk_file << "VECTORS Velocity double\n";
        for (size_t i = 0; i < cells.size(); ++i) {
            vtk_file << u[i] << " " << v[i] << " " << w[i] << "\n";
        }

        // Skalar Tekanan
        vtk_file << "SCALARS Pressure double 1\n";
        vtk_file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < cells.size(); ++i) {
            vtk_file << p[i] << "\n";
        }

        std::cout << "[VTK Output] File hasil disimpan ke: " << output_filename << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Penggunaan: " << argv[0] << " input.msh" << std::endl;
        return 1;
    }

    std::string mesh_filename = argv[1];
    UnifiedCFDSolver solver;

    if (!solver.loadMesh(mesh_filename)) {
        return 1;
    }

    // Eksekusi simulasi
    solver.solveParallel(500, 0.001);

    // Export hasil ke VTK untuk visualisasi di ParaView
    solver.writeVTK("output.vtk");

    return 0;
}