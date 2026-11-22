"""
top88.py -- Python (NumPy/SciPy/Matplotlib) port of the 88-line MATLAB
topology optimization code by Andreassen, Clausen, Schevenels, Lazarov
and Sigmund (Struct Multidisc Optim, 2011; DTU 2010).

Usage (from a terminal):
    python top88.py                       # default: 60 x 20 half MBB beam
    python top88.py 120 40 0.5 3.0 4.8 2  # nelx nely volfrac penal rmin ft

ft = 1 -> sensitivity filter, ft = 2 -> density filter.
Requires: numpy, scipy, matplotlib   (pip install numpy scipy matplotlib)
"""
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def top88(nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=1.5, ft=1,
          maxiter=200, show=True, save_png=None,
          E0=1.0, Emin=1e-9, nu=0.3):
    # ---------------- MATERIAL PROPERTIES ----------------
    # E0: solid stiffness, Emin: "void" stiffness, nu: Poisson's ratio
    # (keyword arguments so they can be changed for experiments)

    # ---------------- PREPARE FINITE ELEMENT ANALYSIS ----------------
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    KE = 1 / (1 - nu**2) / 24 * (np.block([[A11, A12], [A12.T, A11]])
                                  + nu * np.block([[B11, B12], [B12.T, B11]]))

    nel = nelx * nely
    ndof = 2 * (nelx + 1) * (nely + 1)
    # Node numbers, column-major like MATLAB (order="F"); 0-based here
    nodenrs = np.arange((1 + nelx) * (1 + nely)).reshape((1 + nely, 1 + nelx), order="F")
    edofVec = (2 * nodenrs[:-1, :-1] + 2).reshape(nel, order="F")
    edofMat = edofVec[:, None] + np.array([0, 1, 2 * nely + 2, 2 * nely + 3,
                                           2 * nely, 2 * nely + 1, -2, -1])
    iK = np.kron(edofMat, np.ones((8, 1), dtype=int)).ravel()
    jK = np.kron(edofMat, np.ones((1, 8), dtype=int)).ravel()

    # ---------------- LOADS AND SUPPORTS (HALF MBB BEAM) ----------------
    F = np.zeros(ndof)
    F[1] = -1.0                                   # MATLAB dof 2 -> index 1
    U = np.zeros(ndof)
    fixeddofs = np.union1d(np.arange(0, 2 * (nely + 1), 2), [ndof - 1])
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    # ---------------- PREPARE FILTER ----------------
    r = int(np.ceil(rmin)) - 1
    iH, jH, sH = [], [], []
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1 * nely + j1
            for i2 in range(max(i1 - r, 0), min(i1 + r + 1, nelx)):
                for j2 in range(max(j1 - r, 0), min(j1 + r + 1, nely)):
                    e2 = i2 * nely + j2
                    iH.append(e1); jH.append(e2)
                    sH.append(max(0.0, rmin - np.hypot(i1 - i2, j1 - j2)))
    H = coo_matrix((sH, (iH, jH)), shape=(nel, nel)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()

    # ---------------- INITIALIZE ITERATION ----------------
    x = np.full(nel, volfrac)          # stored as a flat, column-major vector
    xPhys = x.copy()
    loop, change = 0, 1.0
    history = []

    if show:
        plt.ion()
        fig, ax = plt.subplots()
        im = ax.imshow(-xPhys.reshape((nely, nelx), order="F"), cmap="gray",
                       vmin=-1, vmax=0, interpolation="none")
        ax.axis("off")

    # ---------------- START ITERATION ----------------
    while change > 0.01 and loop < maxiter:
        loop += 1
        # FE-ANALYSIS
        sK = (KE.ravel()[:, None] * (Emin + xPhys**penal * (E0 - Emin))).ravel(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = (K + K.T) / 2
        U[freedofs] = spsolve(K[freedofs, :][:, freedofs], F[freedofs])

        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        Ue = U[edofMat]                               # (nel, 8)
        ce = np.sum((Ue @ KE) * Ue, axis=1)
        c = np.sum((Emin + xPhys**penal * (E0 - Emin)) * ce)
        dc = -penal * (E0 - Emin) * xPhys**(penal - 1) * ce
        dv = np.ones(nel)

        # FILTERING / MODIFICATION OF SENSITIVITIES
        if ft == 1:
            dc = H @ (x * dc) / Hs / np.maximum(1e-3, x)
        elif ft == 2:
            dc = H @ (dc / Hs)
            dv = H @ (dv / Hs)

        # OPTIMALITY CRITERIA UPDATE (bisection on Lagrange multiplier)
        l1, l2, move = 0.0, 1e9, 0.2
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0, np.maximum(x - move, np.minimum(1, np.minimum(
                x + move, x * np.sqrt(-dc / dv / lmid)))))
            if ft == 1:
                xPhys = xnew
            elif ft == 2:
                xPhys = (H @ xnew) / Hs
            if xPhys.sum() > volfrac * nel:
                l1 = lmid
            else:
                l2 = lmid
        change = np.max(np.abs(xnew - x))
        x = xnew

        history.append(c)
        print(f" It.:{loop:5d} Obj.:{c:11.4f} Vol.:{xPhys.mean():7.3f} ch.:{change:7.3f}")

        if show:
            im.set_data(-xPhys.reshape((nely, nelx), order="F"))
            fig.canvas.draw(); plt.pause(0.01)

    img = xPhys.reshape((nely, nelx), order="F")
    if save_png:
        plt.imsave(save_png, 1 - img, cmap="gray", vmin=0, vmax=1)
    if show:
        plt.ioff(); plt.show()
    return img, history


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        nelx, nely = int(args[0]), int(args[1])
        volfrac, penal, rmin = float(args[2]), float(args[3]), float(args[4])
        ft = int(args[5])
        top88(nelx, nely, volfrac, penal, rmin, ft)
    else:
        top88()
