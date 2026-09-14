clang++.exe -std=c++17 -O3 -I "C:\Portable\eigen-5.0.0" FE_CPU_Single.cpp -o FE_CPU_Single.exe
clang++.exe -std=c++17 -O3 -I "C:\Portable\eigen-5.0.0" FE_CPU_Parallel.cpp -o FE_CPU_Parallel.exe
clang++.exe -std=c++17 -O3 -I "C:\Portable\eigen-5.0.0" FE_GPU.cpp -o FE_GPU.exe
clang++.exe -std=c++17 -O3 -I "C:\Portable\eigen-5.0.0" FE_GPU_CPU_Hybrid.cpp -o FE_GPU_CPU_Hybrid.exe

clang++.exe -std=c++17 -O3 -I "C:\Portable\eigen-5.0.0" FD_CPU_Single.cpp -o FD_CPU_Single.exe
clang++.exe -std=c++17 -O3 -I "C:\Portable\eigen-5.0.0" FD_CPU_Parallel.cpp -o FD_CPU_Parallel.exe

./FE_CPU_Parallel.exe 4 complex3d.inp output.vtk

pada FE_GPU_CPU_HYBRID.cpp buat computing pada CPU (parallel) dan GPU secara bersamaan