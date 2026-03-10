// cfd2d_openmp.cpp
// Simulasi CFD 2D sederhana menggunakan OpenMP
// Kompilasi: g++ -fopenmp cfd2d_openmp.cpp -o cfd2d_openmp
// Jalankan: ./cfd2d_openmp
#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>

const int nx = 50;
const int ny = 50;
const int nt = 100;
const double dx = 2.0 / (nx - 1);
const double dy = 2.0 / (ny - 1);
const double nu = 0.1;
const double dt = 0.001;

int main() {
    std::vector<double> u(nx * ny, 1.0);
    std::vector<double> u_old(nx * ny, 1.0);

    // Gangguan di tengah domain
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = j * nx + i;
            if (i >= nx/4 && i < nx/2 && j >= ny/4 && j < ny/2) {
                u[idx] = 2.0;
            }
        }
    }

    for (int n = 0; n < nt; ++n) {
        u_old = u;
        // Update interior points (paralel dengan OpenMP)
        #pragma omp parallel for collapse(2)
        for (int j = 1; j < ny-1; ++j) {
            for (int i = 1; i < nx-1; ++i) {
                int idx = j * nx + i;
                int idx_ip = j * nx + (i+1);
                int idx_im = j * nx + (i-1);
                int idx_jp = (j+1) * nx + i;
                int idx_jm = (j-1) * nx + i;
                u[idx] = u_old[idx]
                    - u_old[idx] * dt / dx * (u_old[idx] - u_old[idx_im])
                    - u_old[idx] * dt / dy * (u_old[idx] - u_old[idx_jm])
                    + nu * dt / (dx*dx) * (u_old[idx_ip] - 2*u_old[idx] + u_old[idx_im])
                    + nu * dt / (dy*dy) * (u_old[idx_jp] - 2*u_old[idx] + u_old[idx_jm]);
            }
        }
        // Boundary condition
        for (int i = 0; i < nx; ++i) {
            u[i] = 1.0; // bottom
            u[(ny-1)*nx + i] = 1.0; // top
        }
        for (int j = 0; j < ny; ++j) {
            u[j*nx] = 1.0; // left
            u[j*nx + nx-1] = 1.0; // right
        }
    }

    // Simpan hasil ke file
    std::ofstream fout("cfd2d_openmp_u.dat");
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            fout << u[j*nx + i] << (i == nx-1 ? "\n" : " ");
        }
    }
    fout.close();
    std::cout << "Simulasi selesai. Hasil disimpan di cfd2d_openmp_u.dat" << std::endl;
    return 0;
}
