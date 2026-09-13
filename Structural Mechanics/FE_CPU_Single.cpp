#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

// Struktur Node
struct Node {
    int id;
    double x, y;
};

// Element Segitiga 3-Node (CPS3)
struct Element {
    int id;
    int n1, n2, n3;
};

// Struktur Batas (Constraint)
struct BoundaryCondition {
    int node_id;
    int dof; // 1 = Ux, 2 = Uy
    double value;
};

// Struktur Beban Nodal
struct NodalLoad {
    int node_id;
    int dof; // 1 = Fx, 2 = Fy
    double magnitude;
};

class FEMSolver {
public:
    // Properti Material Baja (Steel - Plane Stress)
    const double E = 210.0e9;   // Modulus Young (Pa)
    const double nu = 0.3;      // Rasio Poisson
    const double t = 0.001;     // Ketebalan (m)

    std::vector<Node> nodes;
    std::map<int, int> node_id_to_index; // Pemetaan ID Node -> Indeks Matriks (0 ke N-1)
    std::vector<Element> elements;
    std::vector<BoundaryCondition> bcs;
    std::vector<NodalLoad> loads;
    
    Eigen::VectorXd U; // Vector Perpindahan (Displacement)

    // Parser File .inp Abaqus/CalculiX
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
                
                node_id_to_index[node.id] = nodes.size();
                nodes.push_back(node);
            }
            else if (currentSection == ELEMENT) {
                Element elem;
                std::getline(ss, item, ','); elem.id = std::stoi(item);
                std::getline(ss, item, ','); elem.n1 = std::stoi(item);
                std::getline(ss, item, ','); elem.n2 = std::stoi(item);
                std::getline(ss, item, ','); elem.n3 = std::stoi(item);
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

    // Matriks Konstitutif D (Plane Stress)
    Eigen::Matrix3d getConstitutiveMatrix() const {
        Eigen::Matrix3d D;
        double factor = E / (1.0 - nu * nu);
        D << 1.0,  nu,   0.0,
             nu,   1.0,  0.0,
             0.0,  0.0,  (1.0 - nu) / 2.0;
        return factor * D;
    }

    // Matriks Kekakuan Element
    Eigen::Matrix<double, 6, 6> computeElementStiffness(const Element& elem) const {
        const Node& n1 = nodes[node_id_to_index.at(elem.n1)];
        const Node& n2 = nodes[node_id_to_index.at(elem.n2)];
        const Node& n3 = nodes[node_id_to_index.at(elem.n3)];

        double Area = 0.5 * std::abs(n1.x * (n2.y - n3.y) + n2.x * (n3.y - n1.y) + n3.x * (n1.y - n2.y));

        double b1 = n2.y - n3.y, c1 = n3.x - n2.x;
        double b2 = n3.y - n1.y, c2 = n1.x - n3.x;
        double b3 = n1.y - n2.y, c3 = n2.x - n1.x;

        Eigen::Matrix<double, 3, 6> B;
        B << b1,  0, b2,  0, b3,  0,
              0, c1,  0, c2,  0, c3,
             c1, b1, c2, b2, c3, b3;
        B /= (2.0 * Area);

        Eigen::Matrix3d D = getConstitutiveMatrix();
        return t * Area * B.transpose() * D * B;
    }

    void solve() {
        int num_nodes = nodes.size();
        int total_dofs = num_nodes * 2;

        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& elem : elements) {
            Eigen::Matrix<double, 6, 6> Ke = computeElementStiffness(elem);
            int element_nodes[3] = {
                node_id_to_index[elem.n1],
                node_id_to_index[elem.n2],
                node_id_to_index[elem.n3]
            };

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int r = 0; r < 2; ++r) {
                        for (int c = 0; c < 2; ++c) {
                            int row = element_nodes[i] * 2 + r;
                            int col = element_nodes[j] * 2 + c;
                            triplets.push_back(Eigen::Triplet<double>(row, col, Ke(i * 2 + r, j * 2 + c)));
                        }
                    }
                }
            }
        }

        Eigen::SparseMatrix<double> K(total_dofs, total_dofs);
        K.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::VectorXd F = Eigen::VectorXd::Zero(total_dofs);
        for (const auto& load : loads) {
            int idx = node_id_to_index[load.node_id];
            int global_dof = idx * 2 + (load.dof - 1);
            F(global_dof) += load.magnitude;
        }

        // Terapkan Boundary Conditions
        for (const auto& bc : bcs) {
            int idx = node_id_to_index[bc.node_id];
            int global_dof = idx * 2 + (bc.dof - 1);

            for (Eigen::SparseMatrix<double>::InnerIterator it(K, global_dof); it; ++it) {
                it.valueRef() = (it.row() == global_dof) ? 1.0 : 0.0;
            }
            F(global_dof) = bc.value;
        }

        // Solve K * U = F
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(K);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Penguraian matriks gagal!" << std::endl;
            return;
        }

        U = solver.solve(F);
        std::cout << "Penyelesaian persamaan FEM selesai diselesaikan." << std::endl;
    }

    // Eksport Hasil ke Format VTK (.vtk)
    void exportVTK(const std::string& filename) {
        std::ofstream vtkFile(filename);
        if (!vtkFile.is_open()) {
            std::cerr << "Gagal membuat file VTK: " << filename << std::endl;
            return;
        }

        // 1. Header VTK
        vtkFile << "# vtk DataFile Version 3.0\n";
        vtkFile << "FEM 2D Result Output\n";
        vtkFile << "ASCII\n";
        vtkFile << "DATASET UNSTRUCTURED_GRID\n";

        // 2. Koordinat Node (Points)
        vtkFile << "POINTS " << nodes.size() << " double\n";
        for (const auto& node : nodes) {
            vtkFile << node.x << " " << node.y << " 0.0\n"; // z=0 untuk 2D
        }

        // 3. Elemen (Cells)
        // Format VTK cell: [Jumlah Node per Cell, idx1, idx2, idx3]
        vtkFile << "\nCELLS " << elements.size() << " " << elements.size() * 4 << "\n";
        for (const auto& elem : elements) {
            vtkFile << "3 " 
                    << node_id_to_index[elem.n1] << " "
                    << node_id_to_index[elem.n2] << " "
                    << node_id_to_index[elem.n3] << "\n";
        }

        // 4. Tipe Elemen (Cell Types)
        // VTK_TRIANGLE = Type 5
        vtkFile << "\nCELL_TYPES " << elements.size() << "\n";
        for (size_t i = 0; i < elements.size(); ++i) {
            vtkFile << "5\n"; 
        }

        // 5. Data Node (POINT_DATA) -> Displacement
        vtkFile << "\nPOINT_DATA " << nodes.size() << "\n";
        
        // Vektor Displacement (Ux, Uy, Uz=0)
        vtkFile << "VECTORS Displacement double\n";
        for (const auto& node : nodes) {
            int idx = node_id_to_index[node.id];
            vtkFile << U(idx * 2) << " " << U(idx * 2 + 1) << " 0.0\n";
        }

        // Skalar Magnitude Displacement
        vtkFile << "\nSCALARS DisplacementMagnitude double 1\n";
        vtkFile << "LOOKUP_TABLE default\n";
        for (const auto& node : nodes) {
            int idx = node_id_to_index[node.id];
            double ux = U(idx * 2);
            double uy = U(idx * 2 + 1);
            double mag = std::sqrt(ux * ux + uy * uy);
            vtkFile << mag << "\n";
        }

        vtkFile.close();
        std::cout << "File VTK berhasil dibuat: " << filename << std::endl;
    }
};

int main() {
    FEMSolver solver;
    if (solver.parseINP("input.inp")) {
        solver.solve();
        solver.exportVTK("output.vtk");
    }
    return 0;
}