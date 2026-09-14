#include <iostream>
#include <fstream>
#include <string>

// CGAL Surface Mesh & Polyhedron
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>

// CGAL 2D Surface Remesh
#include <CGAL/Polygon_mesh_processing/isotropic_remeshing.h>

// CGAL 3D Mesh Generation
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Polyhedron_3_to_inria_mesh_3.h>
#include <CGAL/make_mesh_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3>                     Surface_mesh;
typedef CGAL::Polyhedron_3<K>                               Polyhedron;

// Domain & Criteria for 3D Mesh
typedef CGAL::Polyhedral_mesh_domain_3<Polyhedron, K>       Mesh_domain;
typedef CGAL::Sequential_tag                                CGAL_Tag;
typedef CGAL::Mesh_triangulation_3<Mesh_domain, CGAL::Default, CGAL_Tag>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr>         C3t3;
typedef CGAL::Mesh_criteria_3<Tr>                           Mesh_criteria;

namespace PMP = CGAL::Polygon_mesh_processing;

// Function for 2D Surface Remeshing
void generate2DMesh(const std::string& input_stl, const std::string& output_off, double target_edge_length) {
    std::cout << "\n[2D] Processing Surface Mesh Generation..." << std::endl;

    Surface_mesh mesh;
    if (!PMP::IO::read_polygon_mesh(input_stl, mesh)) {
        std::cerr << "Error: Gagal membaca file STL: " << input_stl << std::endl;
        return;
    }

    // Remeshing permukaan secara konsisten (Isotropic Remeshing)
    unsigned int nb_iterations = 3;
    PMP::isotropic_remeshing(faces(mesh), target_edge_length, mesh,
                            PMP::parameters::number_of_iterations(nb_iterations));

    // Save Output
    std::ofstream out(output_off);
    out << mesh;
    std::cout << "[2D] Selesai! Result saved to: " << output_off << std::endl;
}

// Function for 3D Volume Mesh Generation
void generate3DMesh(const std::string& input_stl, const std::string& output_mesh, double edge_size, double facet_size, double cell_size) {
    std::cout << "\n[3D] Processing Volume Mesh Generation (Tetrahedrons)..." << std::endl;

    Polyhedron polyhedron;
    if (!PMP::IO::read_polygon_mesh(input_stl, polyhedron)) {
        std::cerr << "Error: Gagal membaca file STL ke Polyhedron: " << input_stl << std::endl;
        return;
    }

    if (!CGAL::is_closed(polyhedron)) {
        std::cerr << "Warning: STL input tidak tertutup (not a closed watertight manifold!). Mesh 3D mungkin gagal." << std::endl;
    }

    // Define 3D Mesh Domain
    Mesh_domain domain(polyhedron);

    // Set Mesh Criteria (Size and Quality Constraints)
    using namespace CGAL::parameters;
    Mesh_criteria criteria(facet_angle = 25,
                          facet_size = facet_size,
                          facet_distance = 0.008,
                          cell_radius_edge_ratio = 3,
                          cell_size = cell_size);

    // Generate 3D Mesh
    C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);

    // Save Output in INRIA .mesh format (dapat dibaca oleh ParaView/GMSH)
    std::ofstream output(output_mesh);
    CGAL::IO::output_to_medit(output, c3t3);
    std::cout << "[3D] Selesai! Result saved to: " << output_mesh << std::endl;
}

int main(int argc, char* argv[]) {
    std::string input_stl;

    if (argc > 1) {
        input_stl = argv[1];
    } else {
        std::cout << "Masukkan path file STL input (misal: input.stl): ";
        std::cin >> input_stl;
    }

    int choice;
    std::cout << "\nPilih Jenis Mesh Generation:\n";
    std::cout << "1. 2D Surface Mesh (Triangulated Surface)\n";
    std::cout << "2. 3D Volume Mesh (Tetrahedral Mesh)\n";
    std::cout << "3. Generasi Keduanya (2D & 3D)\n";
    std::cout << "Pilihan (1-3): ";
    std::cin >> choice;

    if (choice == 1 || choice == 3) {
        double edge_len;
        std::cout << "\n[2D] Masukkan Target Panjang Edge (misal: 1.0): ";
        std::cin >> edge_len;
        generate2DMesh(input_stl, "output_mesh_2d.off", edge_len);
    }

    if (choice == 2 || choice == 3) {
        double edge_size, facet_size, cell_size;
        std::cout << "\n[3D] Masukkan Max Facet Size (misal: 1.5): ";
        std::cin >> facet_size;
        std::cout << "[3D] Masukkan Max Cell/Tetrahedral Size (misal: 2.0): ";
        std::cin >> cell_size;
        
        generate3DMesh(input_stl, "output_mesh_3d.mesh", facet_size, facet_size, cell_size);
    }

    return 0;
}