#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

const int nx = 50;
const int ny = 50;
const int nt = 100; // jumlah langkah waktu
double dx = 2.0 / (nx - 1);
double dy = 2.0 / (ny - 1);
double nu = 0.1; // viskositas kinematik
double dt = 0.001; // langkah waktu

void save_field(const std::vector<std::vector<double>>& u, const std::string& filename) {
    std::ofstream fout(filename);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            fout << std::setw(8) << u[j][i] << " ";
        }
        fout << "\n";
    }
    fout.close();
}

int main() {
    std::vector<std::vector<double>> u(ny, std::vector<double>(nx, 1.0));
    std::vector<std::vector<double>> v(ny, std::vector<double>(nx, 1.0));
    // Gangguan di tengah domain
    for (int j = ny/4; j < ny/2; ++j) {
        for (int i = nx/4; i < nx/2; ++i) {
            u[j][i] = 2.0;
            v[j][i] = 2.0;
        }
    }
    std::vector<std::vector<double>> un = u;
    std::vector<std::vector<double>> vn = v;
    for (int n = 0; n < nt; ++n) {
        un = u;
        vn = v;
        for (int j = 1; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                u[j][i] = un[j][i]
                    - un[j][i] * dt / dx * (un[j][i] - un[j][i-1])
                    - vn[j][i] * dt / dy * (un[j][i] - un[j-1][i])
                    + nu * dt / (dx*dx) * (un[j][i+1 < nx ? i+1 : i] - 2*un[j][i] + un[j][i-1])
                    + nu * dt / (dy*dy) * (un[j+1 < ny ? j+1 : j][i] - 2*un[j][i] + un[j-1][i]);
                v[j][i] = vn[j][i]
                    - un[j][i] * dt / dx * (vn[j][i] - vn[j][i-1])
                    - vn[j][i] * dt / dy * (vn[j][i] - vn[j-1][i])
                    + nu * dt / (dx*dx) * (vn[j][i+1 < nx ? i+1 : i] - 2*vn[j][i] + vn[j][i-1])
                    + nu * dt / (dy*dy) * (vn[j+1 < ny ? j+1 : j][i] - 2*vn[j][i] + vn[j-1][i]);
            }
        }
        // Boundary condition
        for (int i = 0; i < nx; ++i) {
            u[0][i] = 1.0; u[ny-1][i] = 1.0;
            v[0][i] = 1.0; v[ny-1][i] = 1.0;
        }
        for (int j = 0; j < ny; ++j) {
            u[j][0] = 1.0; u[j][nx-1] = 1.0;
            v[j][0] = 1.0; v[j][nx-1] = 1.0;
        }
    }
    save_field(u, "cfd2d_u.dat");
    std::cout << "Simulasi selesai. Hasil disimpan di cfd2d_u.dat" << std::endl;
    return 0;
}
