// cfd2d_petsc.cpp
// Simulasi CFD 2D sederhana menggunakan PETSc
// Kompilasi: mpicxx -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include cfd2d_petsc.cpp -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc -o cfd2d_petsc
// Jalankan: mpirun -np 4 ./cfd2d_petsc
#include <petscksp.h>

const PetscInt nx = 50;
const PetscInt ny = 50;
const PetscInt nt = 100;
const PetscScalar dx = 2.0 / (nx - 1);
const PetscScalar dy = 2.0 / (ny - 1);
const PetscScalar nu = 0.1;
const PetscScalar dt = 0.001;

int main(int argc, char **args) {
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &args, NULL, NULL); if (ierr) return ierr;

    Vec u, u_old;
    PetscInt size = nx * ny;
    ierr = VecCreate(PETSC_COMM_WORLD, &u); CHKERRQ(ierr);
    ierr = VecSetSizes(u, PETSC_DECIDE, size); CHKERRQ(ierr);
    ierr = VecSetFromOptions(u); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &u_old); CHKERRQ(ierr);

    // Inisialisasi u = 1, gangguan di tengah domain u = 2
    PetscScalar *u_array;
    ierr = VecGetArray(u, &u_array); CHKERRQ(ierr);
    for (PetscInt j = 0; j < ny; ++j) {
        for (PetscInt i = 0; i < nx; ++i) {
            PetscInt idx = j * nx + i;
            u_array[idx] = 1.0;
            if (i >= nx/4 && i < nx/2 && j >= ny/4 && j < ny/2) {
                u_array[idx] = 2.0;
            }
        }
    }
    ierr = VecRestoreArray(u, &u_array); CHKERRQ(ierr);

    // Time stepping
    for (PetscInt n = 0; n < nt; ++n) {
        ierr = VecCopy(u, u_old); CHKERRQ(ierr);
        ierr = VecGetArray(u, &u_array); CHKERRQ(ierr);
        const PetscScalar *u_old_array;
        ierr = VecGetArrayRead(u_old, &u_old_array); CHKERRQ(ierr);
        for (PetscInt j = 1; j < ny-1; ++j) {
            for (PetscInt i = 1; i < nx-1; ++i) {
                PetscInt idx = j * nx + i;
                PetscInt idx_ip = j * nx + (i+1);
                PetscInt idx_im = j * nx + (i-1);
                PetscInt idx_jp = (j+1) * nx + i;
                PetscInt idx_jm = (j-1) * nx + i;
                u_array[idx] = u_old_array[idx]
                    - u_old_array[idx] * dt / dx * (u_old_array[idx] - u_old_array[idx_im])
                    - u_old_array[idx] * dt / dy * (u_old_array[idx] - u_old_array[idx_jm])
                    + nu * dt / (dx*dx) * (u_old_array[idx_ip] - 2*u_old_array[idx] + u_old_array[idx_im])
                    + nu * dt / (dy*dy) * (u_old_array[idx_jp] - 2*u_old_array[idx] + u_old_array[idx_jm]);
            }
        }
        ierr = VecRestoreArrayRead(u_old, &u_old_array); CHKERRQ(ierr);
        ierr = VecRestoreArray(u, &u_array); CHKERRQ(ierr);
        // Boundary condition
        ierr = VecGetArray(u, &u_array); CHKERRQ(ierr);
        for (PetscInt i = 0; i < nx; ++i) {
            u_array[i] = 1.0; // bottom
            u_array[(ny-1)*nx + i] = 1.0; // top
        }
        for (PetscInt j = 0; j < ny; ++j) {
            u_array[j*nx] = 1.0; // left
            u_array[j*nx + nx-1] = 1.0; // right
        }
        ierr = VecRestoreArray(u, &u_array); CHKERRQ(ierr);
    }

    // Simpan hasil ke file
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "cfd2d_petsc_u.dat", &viewer); CHKERRQ(ierr);
    ierr = VecView(u, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    ierr = VecDestroy(&u); CHKERRQ(ierr);
    ierr = VecDestroy(&u_old); CHKERRQ(ierr);
    ierr = PetscFinalize();
    return 0;
}
