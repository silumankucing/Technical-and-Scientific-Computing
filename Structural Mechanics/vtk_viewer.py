# Simple VTK Viewer App for Structural Mechanics
# Requires: PyQt5, numpy, matplotlib
# Usage: python vtk_viewer.py

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout, QWidget, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class VTKViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Simple VTK Viewer')
        self.setGeometry(100, 100, 600, 500)
        self.initUI()

    def initUI(self):
        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.add_subplot(111)
        self.label = QLabel('No file loaded')
        btn = QPushButton('Open VTK File')
        btn.clicked.connect(self.open_file)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(btn)
        layout.addWidget(self.canvas)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open VTK File', '', 'VTK Files (*.vtk)')
        if fname:
            self.label.setText('Loaded: {}'.format(fname))
            self.plot_vtk(fname)

    def plot_vtk(self, fname):
        points, polygons, scalars = self.read_vtk(fname)
        self.ax.clear()
        if points is not None and polygons is not None:
            for poly in polygons:
                pts = points[poly]
                pts = np.vstack([pts, pts[0]])  # close the polygon
                self.ax.plot(pts[:,0], pts[:,1], 'k-')
            if scalars is not None:
                scatter = self.ax.scatter(points[:,0], points[:,1], c=scalars, cmap='viridis', s=80)
                self.canvas.figure.colorbar(scatter, ax=self.ax, label='Scalar Value')
            else:
                self.ax.scatter(points[:,0], points[:,1], color='r', s=80)
            self.ax.set_aspect('equal')
            self.ax.set_title('VTK Polydata Visualization')
            self.canvas.draw()

    def read_vtk(self, fname):
        with open(fname) as f:
            lines = f.readlines()
        points, polygons, scalars = None, [], None
        for i, line in enumerate(lines):
            if line.startswith('POINTS'):
                npts = int(line.split()[1])
                pts = []
                for j in range(npts):
                    vals = list(map(float, lines[i+1+j].split()))
                    pts.append(vals[:2])
                points = np.array(pts)
            if line.startswith('POLYGONS'):
                npoly = int(line.split()[1])
                for j in range(npoly):
                    idxs = list(map(int, lines[i+1+j].split()[1:]))
                    polygons.append(idxs)
            if line.startswith('SCALARS'):
                nscal = int(lines[i-1].split()[1])
                scalars = np.array([float(lines[i+3+k]) for k in range(nscal)])
        return points, polygons, scalars

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = VTKViewer()
    viewer.show()
    sys.exit(app.exec_())
