from dataclasses import dataclass
import scipy.io
from scipy.sparse import csc_matrix
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


@dataclass
class MeshHyperelasticity3D:
    # mesh attributes
    dim: int
    level: int
    nn: int
    ne: int
    elems2nodes: np.ndarray
    nodes2coord: np.ndarray
    volumes: np.ndarray
    dphi: list
    Hstr: csc_matrix
    nodesDirichlet: list
    nodesMinim: np.ndarray
    dofsDirichlet: np.ndarray
    dofsMinim: np.ndarray
    dofsMinim_local: np.ndarray
    bfaces2elems: np.ndarray
    bfaces2nodes: np.ndarray

    # params attributes
    lx: float
    ly: float
    lz: float
    E: float
    nu: float
    turning: int
    timeSteps: int
    # draw: list
    visualizeLevels: np.ndarray
    showFullDirichlet: int
    showDirichletTwo: int
    showDirichletOne: int
    # graphs: list
    azimuth: int
    elevation: int
    freq: int
    animations_count: int
    delay: float
    delay_first: float
    delay_last: float
    epsFDSS: float
    max_iters: int
    disp: str
    tf: float
    nbfn: int
    T: int
    lambda_: float
    mu: float
    K: float
    C1: float
    D1: float
    evaluation: int

    def plot_solution(self, u, u0):
        res = np.array(u0)
        res[self.dofsMinim] = u

        # Create a 3D figure
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(projection='3d')  # type: ignore

        ax.view_init(elev=0, azim=90)

        # Plot the wire mesh
        ax.plot_trisurf(res[::3], res[1::3], res[2::3], triangles=self.bfaces2nodes,
                        color="b", edgecolor='k', linewidth=0.5, antialiased=True, shade=True)

        # The closest thing to "equal" scaling:
        # Scale the axes equally
        data_ranges = [np.ptp(a) for a in [res[::3], res[1::3], res[2::3]]]

        ax.set_box_aspect(data_ranges)  # Aspect ratio is 1:1:1
        ax.set_xlim(ax.get_xlim()[::-1])  # type: ignore
        plt.show()

    @classmethod
    def load_mesh_hyperelasticity_3d(cls, level: int) -> 'MeshHyperelasticity3D':
        filename = f"hyperelasticity_mesh_level_{level}.mat"
        data = scipy.io.loadmat(filename)

        # Extracting mesh and params from the loaded data
        mesh_data = data['mesh'][0, 0]
        params_data = data['params'][0, 0]

        # Creating and returning the MeshHyperelasticity3D instance
        return cls(
            # mesh attributes
            dim=mesh_data[0][0, 0],
            level=mesh_data[1][0, 0],
            nn=mesh_data[2][0, 0],
            ne=mesh_data[3][0, 0],
            elems2nodes=mesh_data[4] - 1,
            nodes2coord=mesh_data[5],
            volumes=mesh_data[6],
            dphi=mesh_data[7][0].tolist(),
            Hstr=mesh_data[8],
            nodesDirichlet=[arr - 1 for arr in mesh_data[9][0].tolist()],
            nodesMinim=mesh_data[10] - 1,
            dofsDirichlet=mesh_data[11] - 1,
            dofsMinim=mesh_data[12] - 1,
            dofsMinim_local=mesh_data[13] - 1,
            bfaces2elems=mesh_data[14] - 1,
            bfaces2nodes=mesh_data[15] - 1,
            # params attributes
            lx=params_data[0][0, 0],
            ly=params_data[1][0, 0],
            lz=params_data[2][0, 0],
            E=params_data[3][0, 0],
            nu=params_data[4][0, 0],
            turning=params_data[5][0, 0],
            timeSteps=params_data[6][0, 0],
            # draw=params_data[7].tolist(),
            visualizeLevels=params_data[8],
            showFullDirichlet=params_data[9][0, 0],
            showDirichletTwo=params_data[10][0, 0],
            showDirichletOne=params_data[11][0, 0],
            # graphs=params_data[12].tolist(),
            azimuth=params_data[13][0, 0],
            elevation=params_data[14][0, 0],
            freq=params_data[15][0, 0],
            animations_count=params_data[16][0, 0],
            delay=params_data[17][0, 0],
            delay_first=params_data[18][0, 0],
            delay_last=params_data[19][0, 0],
            epsFDSS=params_data[20][0, 0],
            max_iters=params_data[21][0, 0],
            disp=params_data[22][0],
            tf=params_data[23][0, 0],
            nbfn=params_data[24][0, 0],
            T=params_data[25][0, 0],
            lambda_=params_data[26][0, 0],
            mu=params_data[27][0, 0],
            K=params_data[28][0, 0],
            C1=params_data[29][0, 0],
            D1=params_data[30][0, 0],
            evaluation=params_data[31][0, 0]
        )
