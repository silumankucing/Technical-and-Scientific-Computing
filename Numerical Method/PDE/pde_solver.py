"""
Kelas dasar untuk menyelesaikan PDE secara numerik.
Mendukung metode beda hingga, elemen hingga, dan spektral (kerangka dasar).
"""
import numpy as np

class PDESolver:
    def __init__(self, domain, nx):
        self.domain = domain
        self.nx = nx
        self.x = np.linspace(domain[0], domain[1], nx)

    def solve(self):
        raise NotImplementedError("Implementasikan metode ini di subclass.")
