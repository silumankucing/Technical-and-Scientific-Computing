#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

struct Node {
    int id;
    double x, y, z;
    double temp = 293.15; // Field Suhu
    double u_x = 0.0, u_y = 0.0, u_z = 0.0; // Field Pergeseran (Displacement)
};

struct Element3D {
    int id;
    int nodes[8]; // Hexahedral C3D8R / VTK_HEXAHEDRON
};

struct MaterialProps {
    double density = 0.0;
    double cp = 0.0;
    double conductivity = 0.0;
    double expansion = 0.0;
};

// Parser Abaqus INP
void parseAbaqusINP(const std::string& filename, std::vector<Node>& nodes, std::vector<Element3D>& elements, MaterialProps& mat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Tidak dapat membuka file '" << filename << "'" << std::endl;
        exit(1);
    }

    std::string line;
    std::string current_section = "";

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.find("**") == 0) continue;

        if (line[0] == '*') {
            std::string upperLine = line;
            std::transform(upperLine.begin(), upperLine.end(), upperLine.begin(), ::toupper);

            if (upperLine.rfind("*NODE", 0) == 0) current_section = "NODE";
            else if (upperLine.rfind("*ELEMENT", 0) == 0) current_section = "ELEMENT";
            else if (upperLine.rfind("*DENSITY", 0) == 0) current_section = "DENSITY";
            else if (upperLine.rfind("*SPECIFIC HEAT", 0) == 0) current_section = "CP";
            else if (upperLine.rfind("*CONDUCTIVITY", 0) == 0) current_section = "CONDUCTIVITY";
            else if (upperLine.rfind("*EXPANSION", 0) == 0) current_section = "EXPANSION";
            else current_section = "OTHER";
            continue;
        }

        std::stringstream ss(line);
        std::string val;

        if (current_section == "NODE") {
            Node n;
            if (std::getline(ss, val, ',')) {
                n.id = std::stoi(val);
                std::getline(ss, val, ','); n.x = std::stod(val);
                std::getline(ss, val, ','); n.y = std::stod(val);
                std::getline(ss, val, ','); n.z = std::stod(val);
                nodes.push_back(n);
            }
        } 
        else if (current_section == "ELEMENT") {
            Element3D elem;
            if (std::getline(ss, val, ',')) {
                elem.id = std::stoi(val);
                for (int i = 0; i < 8; ++i) {
                    if (std::getline(ss, val, ',')) {
                        elem.nodes[i] = std::stoi(val);
                    }
                }
                elements.push_back(elem);
            }
        } 
        else if (current_section == "DENSITY") {
            mat.density = std::stod(line);
        } 
        else if (current_section == "CP") {
            mat.cp = std::stod(line);
        } 
        else if (current_section == "CONDUCTIVITY") {
            mat.conductivity = std::stod(line);
        } 
        else if (current_section == "EXPANSION") {
            mat.expansion = std::stod(line);
        }
    }
    file.close();
}

// Fungsi Eksportir Hasil ke Format VTK (Legacy Unstructured Grid)
void exportToVTK(const std::string& vtkFilename, const std::vector<Node>& nodes, const std::vector<Element3D>& elements) {
    std::ofstream vtkFile(vtkFilename);
    if (!vtkFile.is_open()) {
        std::cerr << "Error: Gagal membuat file VTK '" << vtkFilename << "'" << std::endl;
        return;
    }

    // 1. Header VTK
    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "Thermo-Mechanical Simulation Output\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET UNSTRUCTURED_GRID\n\n";

    // 2. Koordinat Node (POINTS)
    vtkFile << "POINTS " << nodes.size() << " double\n";
    for (const auto& node : nodes) {
        vtkFile << node.x << " " << node.y << " " << node.z << "\n";
    }
    vtkFile << "\n";

    // 3. Topologi Elemen (CELLS: jumlah_cell, total_integer_data)
    // 8 node per elemen hex + 1 count integer = 9 nilai per elemen
    vtkFile << "CELLS " << elements.size() << " " << elements.size() * 9 << "\n";
    for (const auto& elem : elements) {
        vtkFile << "8";
        for (int i = 0; i < 8; ++i) {
            // VTK menggunakan 0-based index
            vtkFile << " " << (elem.nodes[i] - 1);
        }
        vtkFile << "\n";
    }
    vtkFile << "\n";

    // 4. Tipe Elemen VTK (CELL_TYPES)
    // 12 = VTK_HEXAHEDRON
    vtkFile << "CELL_TYPES " << elements.size() << "\n";
    for (size_t i = 0; i < elements.size(); ++i) {
        vtkFile << "12\n";
    }
    vtkFile << "\n";

    // 5. Data Field Pada Node (POINT_DATA)
    vtkFile << "POINT_DATA " << nodes.size() << "\n";

    // Scalar Field: Suhu (Temperature)
    vtkFile << "SCALARS Temperature double 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& node : nodes) {
        vtkFile << node.temp << "\n";
    }
    vtkFile << "\n";

    // Vector Field: Pergeseran (Displacement)
    vtkFile << "VECTORS Displacement double\n";
    for (const auto& node : nodes) {
        vtkFile << node.u_x << " " << node.u_y << " " << node.z << "\n";
    }

    vtkFile.close();
    std::cout << "File VTK berhasil dibuat: " << vtkFilename << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Penggunaan: " << argv[0] << " <file_input.inp>" << std::endl;
        return 1;
    }

    std::string inputFilename = argv[1];
    std::vector<Node> nodes;
    std::vector<Element3D> elements;
    MaterialProps mat;

    // 1. Baca File Abaqus INP
    parseAbaqusINP(inputFilename, nodes, elements, mat);

    if (nodes.empty()) {
        std::cerr << "Error: Tidak ada data node yang terbaca!" << std::endl;
        return 1;
    }

    // 2. Contoh Dummy Perhitungan Thermo-Mechanical (Simulasi Sederhana)
    for (size_t i = 0; i < nodes.size(); ++i) {
        // Memberikan kontur suhu contoh (misal gradien sepanjang sumbu X)
        nodes[i].temp = 293.15 + nodes[i].x * 1000.0;
        // Simulasi pergeseran ekspansi termal contoh
        nodes[i].u_x = mat.expansion * (nodes[i].temp - 293.15) * nodes[i].x;
    }

    // 3. Export Hasil ke File VTK
    std::string vtkOutput = "output_result.vtk";
    exportToVTK(vtkOutput, nodes, elements);

    return 0;
}