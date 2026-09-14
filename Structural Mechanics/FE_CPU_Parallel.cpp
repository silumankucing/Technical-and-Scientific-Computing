// =====================================================================
// FE_CPU_Parallel.cpp
// FEM Shared Memory Parallelism (std::thread)
// - 2D: Plane Stress CPS3 (segitiga 3-node)
// - 3D: C3D4 (tetrahedron 4-node), terdeteksi otomatis dari TYPE=C3D4
//
// Penggunaan:
//   ./FE_CPU_Parallel.exe <jumlah_core> <file_input.inp> <file_output.vtk>
// Contoh:
//   ./FE_CPU_Parallel.exe 4 input.inp output.vtk
//   ./FE_CPU_Parallel.exe 8 complex3d.inp complex3d.vtk
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
#include <thread>
#include <mutex>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

// Struktur Node
struct Node {
    int id;
    double x, y;
    double z = 0.0;
};

// Element Segitiga 3-Node (CPS3, 2D) / Tetrahedron 4-Node (C3D4, 3D)
struct Element {
    int id;
    int n1, n2, n3;
    int n4 = 0; // dipakai hanya untuk C3D4
};

// Struktur Batas (Constraint)
struct BoundaryCondition {
    int node_id;
    int dof; // 2D: 1=Ux, 2=Uy | 3D: 1=Ux, 2=Uy, 3=Uz
    double value;
};

// Struktur Beban Nodal
struct NodalLoad {
    int node_id;
    int dof;
    double magnitude;
};

class FEMSolver {
public:
    // Properti Material Baja
    const double E = 210.0e9;   // Modulus Young (Pa)
    const double nu = 0.3;      // Rasio Poisson
    const double t = 0.001;     // Ketebalan (m, hanya 2D)

    bool is3D = false;          // true jika elemen C3D4

    std::vector<Node> nodes;
    std::map<int, int> node_id_to_index;
    std::vector<Element> elements;
    std::vector<BoundaryCondition> bcs;
    std::vector<NodalLoad> loads;

    Eigen::VectorXd U; // Vector Perpindahan

    int dofPerNode() const { return is3D ? 3 : 2; }
    int nodesPerElement() const { return is3D ? 4 : 3; }

    // Parser File .inp Abaqus/CalculiX (tetap sekuensial - I/O bound)
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
                // Koordinat z opsional (ada pada mesh 3D)
                node.z = 0.0;
                if (std::getline(ss, item, ',')) {
                    if (!item.empty()) node.z = std::stod(item);
                }

                node_id_to_index[node.id] = nodes.size();
                nodes.push_back(node);
            }
            else if (currentSection == ELEMENT) {
                Element elem;
                std::getline(ss, item, ','); elem.id = std::stoi(item);
                std::getline(ss, item, ','); elem.n1 = std::stoi(item);
                std::getline(ss, item, ','); elem.n2 = std::stoi(item);
                std::getline(ss, item, ','); elem.n3 = std::stoi(item);
                // Node ke-4 opsional (C3D4)
                elem.n4 = 0;
                if (std::getline(ss, item, ',')) {
                    if (!item.empty()) elem.n4 = std::stoi(item);
                }
                elements.push_back(elem);
            }
            else if (currentSection == BOUNDARY) {
                BoundaryCondition bc;
                std::getline(ss, item, ','); bc.node_id = std::stoi(item);
                std::getline(ss, item, ','); int start_dof = std::stoi(item);
                std::getline(ss, item, ','); int end_dof = std::stoi(item);
                double val = 0.0;
                if (std::getline(ss, item, ',')) val = std::stod(item);

                for (int d = start_dof; d <= end_dof; ++d) {
                    bcs.push_back({bc.node_id, d, val});
                }
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

    // Matriks Konstitutif D (2D Plane Stress)
    Eigen::Matrix<double, 3, 3> getConstitutiveMatrix() const {
        double factor = E / (1.0 - nu * nu);
        Eigen::Matrix<double, 3, 3> D;
        D << 1.0, nu, 0.0,
             nu, 1.0, 0.0,
             0.0, 0.0, (1.0 - nu) / 2.0;
        return factor * D;
    }

    // Matriks Konstitutif D 3D (Isotropik)
    Eigen::Matrix<double, 6, 6> getConstitutiveMatrix3D() const {
        double factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
        double g = (1.0 - 2.0 * nu) / 2.0;
        Eigen::Matrix<double, 6, 6> D;
        D << 1.0 - nu, nu,      nu,      0.0, 0.0, 0.0,
             nu,      1.0 - nu, nu,      0.0, 0.0, 0.0,
             nu,      nu,       1.0 - nu, 0.0, 0.0, 0.0,
             0.0,     0.0,      0.0,     g,   0.0, 0.0,
             0.0,     0.0,      0.0,     0.0, g,   0.0,
             0.0,     0.0,      0.0,     0.0, 0.0, g;
        return factor * D;
    }

    // Matriks Kekakuan Element: CPS3 -> 6x6, C3D4 -> 12x12
    Eigen::MatrixXd computeElementStiffness(const Element& elem) const {
        if (is3D) {
            // Tetrahedron linear 4-node: Ke = V * B^T * D * B
            const Node& n1 = nodes[node_id_to_index.at(elem.n1)];
            const Node& n2 = nodes[node_id_to_index.at(elem.n2)];
            const Node& n3 = nodes[node_id_to_index.at(elem.n3)];
            const Node& n4 = nodes[node_id_to_index.at(elem.n4)];

            Eigen::Matrix4d T;
            T << 1.0, n1.x, n1.y, n1.z,
                 1.0, n2.x, n2.y, n2.z,
                 1.0, n3.x, n3.y, n3.z,
                 1.0, n4.x, n4.y, n4.z;

            double V = std::abs(T.determinant()) / 6.0; // Volume tetra
            Eigen::Matrix4d Ti = T.inverse();

            // B (6x12): urutan regangan eps_x, eps_y, eps_z, gam_yz, gam_xz, gam_xy
            Eigen::Matrix<double, 6, 12> B = Eigen::Matrix<double, 6, 12>::Zero();
            for (int i = 0; i < 4; ++i) {
                double dNx = Ti(1, i);
                double dNy = Ti(2, i);
                double dNz = Ti(3, i);
                B(0, 3 * i)     = dNx;
                B(1, 3 * i + 1) = dNy;
                B(2, 3 * i + 2) = dNz;
                B(3, 3 * i + 1) = dNz; B(3, 3 * i + 2) = dNy;
                B(4, 3 * i)     = dNz; B(4, 3 * i + 2) = dNx;
                B(5, 3 * i)     = dNy; B(5, 3 * i + 1) = dNx;
            }

            Eigen::Matrix<double, 6, 6> D = getConstitutiveMatrix3D();
            return V * B.transpose() * D * B;
        }

        // 2D: Segitiga Plane Stress CPS3
        const Node& n1 = nodes[node_id_to_index.at(elem.n1)];
        const Node& n2 = nodes[node_id_to_index.at(elem.n2)];
        const Node& n3 = nodes[node_id_to_index.at(elem.n3)];

        double Area = 0.5 * std::abs(n1.x * (n2.y - n3.y) + n2.x * (n3.y - n1.y) + n3.x * (n1.y - n2.y));

        Eigen::Matrix<double, 3, 6> B;
        B << (n2.y - n3.y), 0.0, (n3.y - n1.y), 0.0, (n1.y - n2.y), 0.0,
             0.0, (n3.x - n2.x), 0.0, (n1.x - n3.x), 0.0, (n2.x - n1.x),
             (n3.x - n2.x), (n2.y - n3.y), (n1.x - n3.x), (n3.y - n1.y), (n2.x - n1.x), (n1.y - n2.y);
        B /= (2.0 * Area);

        Eigen::Matrix<double, 3, 3> D = getConstitutiveMatrix();
        return t * Area * B.transpose() * D * B;
    }

    // Helper Pembagian Rentang untuk Multi-threading
    static void parallelRange(int total, int num_threads, int thread_id, int& start, int& end) {
        int chunk = total / num_threads;
        int remainder = total % num_threads;
        start = thread_id * chunk + (thread_id < remainder ? thread_id : remainder);
        end = start + chunk + (thread_id < remainder ? 1 : 0);
    }

    void solve(int num_threads);
    void exportVTK(const std::string& filename);
};

// Implementasi Solver Paralel
void FEMSolver::solve(int num_threads) {
    using T = Eigen::Triplet<double>;

    const int dn = dofPerNode();          // DOF per node (2 atau 3)
    const int npe = nodesPerElement();    // node per elemen (3 atau 4)
    const int total_dofs = (int)nodes.size() * dn;

    std::vector<T> tripletList;
    tripletList.reserve((size_t)elements.size() * npe * npe * dn * dn);

    std::vector<std::vector<T>> local_triplets(num_threads);
    std::vector<std::thread> threads;
    int num_elements = (int)elements.size();

    // ================================================
    // FASE 1: Perakitan Matriks Kekakuan Global (Paralel)
    // ================================================
    auto assembly_start = std::chrono::high_resolution_clock::now();

    for (int tid = 0; tid < num_threads; ++tid) {
        threads.emplace_back([&, tid]() {
            int start, end;
            parallelRange(num_elements, num_threads, tid, start, end);
            std::vector<T>& local = local_triplets[tid];

            for (int e = start; e < end; ++e) {
                const Element& elem = elements[e];
                Eigen::MatrixXd Ke = computeElementStiffness(elem);

                int element_nodes[4];
                element_nodes[0] = node_id_to_index.at(elem.n1);
                element_nodes[1] = node_id_to_index.at(elem.n2);
                element_nodes[2] = node_id_to_index.at(elem.n3);
                if (is3D) element_nodes[3] = node_id_to_index.at(elem.n4);

                for (int i = 0; i < npe; ++i) {
                    for (int j = 0; j < npe; ++j) {
                        for (int r = 0; r < dn; ++r) {
                            for (int c = 0; c < dn; ++c) {
                                local.push_back(T(
                                    element_nodes[i] * dn + r,
                                    element_nodes[j] * dn + c,
                                    Ke(i * dn + r, j * dn + c)));
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto& t : threads) t.join();
    threads.clear();

    for (int tid = 0; tid < num_threads; ++tid) {
        tripletList.insert(tripletList.end(),
                           local_triplets[tid].begin(),
                           local_triplets[tid].end());
    }

    Eigen::SparseMatrix<double> K(total_dofs, total_dofs);
    K.setFromTriplets(tripletList.begin(), tripletList.end());

    auto assembly_end = std::chrono::high_resolution_clock::now();
    auto assembly_duration = std::chrono::duration_cast<std::chrono::milliseconds>(assembly_end - assembly_start);

    // ================================================
    // FASE 2: Pemasangan Vector Beban F
    // ================================================
    Eigen::VectorXd F = Eigen::VectorXd::Zero(total_dofs);
    for (const auto& load : loads) {
        int idx = node_id_to_index.at(load.node_id);
        F(idx * dn + (load.dof - 1)) += load.magnitude;
    }

    // ================================================
    // FASE 3: Penerapan Kondisi Batas (Penalti)
    // ================================================
    Eigen::SparseMatrix<double> K_modified = K;
    for (const auto& bc : bcs) {
        int idx = node_id_to_index.at(bc.node_id);
        int global_dof = idx * dn + (bc.dof - 1);
        K_modified.coeffRef(global_dof, global_dof) *= 1e10;
        F(global_dof) = 0.0;
    }

    // ================================================
    // FASE 4: Penyelesaian Persamaan K*U = F (Solver Langsung)
    // ================================================
    auto solve_start = std::chrono::high_resolution_clock::now();

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(K_modified);
    solver.factorize(K_modified);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Error: Faktorisasi LU gagal!" << std::endl;
        return;
    }
    U = solver.solve(F);

    auto solve_end = std::chrono::high_resolution_clock::now();
    auto solve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - solve_start);

    std::cout << "  [Perakitan K] " << tripletList.size() << " entri dari "
              << num_elements << " elemen, durasi " << assembly_duration.count() << " ms" << std::endl;
    std::cout << "  [Solve LU] durasi " << solve_duration.count() << " ms" << std::endl;
    std::cout << "Penyelesaian persamaan FEM selesai diselesaikan." << std::endl;
}

// Ekspor Hasil ke Format VTK Legacy (Paraview)
void FEMSolver::exportVTK(const std::string& filename) {
    std::ofstream vtkFile(filename);
    if (!vtkFile.is_open()) {
        std::cerr << "Error: Tidak dapat membuat file VTK!" << std::endl;
        return;
    }

    const int dn = dofPerNode();
    const int npe = nodesPerElement();

    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << (is3D ? "Hasil FEM 3D (C3D4)" : "Hasil FEM 2D (CPS3)") << "\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET UNSTRUCTURED_GRID\n";

    // Titik (Node)
    vtkFile << "POINTS " << nodes.size() << " double\n";
    for (const auto& node : nodes) {
        vtkFile << node.x << " " << node.y << " " << node.z << "\n";
    }

    // Element
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

    // Tipe Cell VTK: 5 = VTK_TRIANGLE, 10 = VTK_TETRA
    vtkFile << "CELL_TYPES " << elements.size() << "\n";
    for (size_t i = 0; i < elements.size(); ++i) {
        vtkFile << (is3D ? 10 : 5) << "\n";
    }

    // Data Perpindahan
    vtkFile << "POINT_DATA " << nodes.size() << "\n";
    vtkFile << "VECTORS Displacement double\n";
    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        if (is3D) {
            vtkFile << U(idx * 3) << " " << U(idx * 3 + 1) << " " << U(idx * 3 + 2) << "\n";
        } else {
            vtkFile << U(idx * 2) << " " << U(idx * 2 + 1) << " 0.0\n";
        }
    }

    // Magnitude Perpindahan
    vtkFile << "SCALARS DisplacementMagnitude double 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        double mag = 0.0;
        for (int d = 0; d < dn; ++d) {
            mag += U(idx * dn + d) * U(idx * dn + d);
        }
        vtkFile << std::sqrt(mag) << "\n";
    }

    vtkFile.close();
    std::cout << "File VTK berhasil dibuat: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Penggunaan: " << argv[0]
                  << " <jumlah_core> <file_input.inp> <file_output.vtk>" << std::endl;
        return 1;
    }

    int num_threads = std::atoi(argv[1]);
    std::string input_file = argv[2];
    std::string output_file = argv[3];

    if (num_threads <= 0) num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads <= 0) num_threads = 1;

    auto total_start = std::chrono::high_resolution_clock::now();

    FEMSolver solver;

    if (!solver.parseINP(input_file)) return 1;

    std::cout << "Menjalankan dengan " << num_threads << " thread (shared memory)..."
              << (solver.is3D ? " [MODE 3D: C3D4]" : " [MODE 2D: CPS3]") << std::endl;

    solver.solve(num_threads);
    solver.exportVTK(output_file);

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "Total waktu eksekusi: " << total_duration.count() << " ms" << std::endl;

    return 0;
}
