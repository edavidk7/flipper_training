from typing import Literal, List, Optional, Tuple, Dict
import torch
import numpy as np
from pathlib import Path
import pyvista as pv
import pyacvd
import yaml
from dataclasses import dataclass
from flipper_training.utils.numerical import IntegrationMode
from flipper_training.utils.geometry import points_in_oriented_box

ROOT = Path(__file__).parent.parent
MESHDIR = ROOT / "meshes"
YAMLDIR = ROOT / "robots"
IMPLEMENTED_ROBOTS = ["marv"]


def extract_surface_from_mesh(mesh: pv.PolyData, n_points: int = 100, **clus_opts) -> torch.Tensor:
    """
    Extracts the surface of a mesh and clusters it to n_points.

    First, the delauany triangulation is computed and the surface is extracted.
    Then, the surface is clustered using the pyacvd library.

    Args:
        mesh (pv.PolyData): mesh object.
        n_points (int, optional): number of points extracted. Defaults to 100.

    Returns:
        torch.Tensor: extracted points.
    """
    delaunay = mesh.delaunay_3d()
    surf = delaunay.extract_surface()
    clus: pyacvd.Clustering = pyacvd.Clustering(surf)
    clus.cluster(n_points, **clus_opts)
    return torch.tensor(clus.cluster_centroid)


def voxelize_mesh(mesh: pv.PolyData, voxel_size: float) -> torch.Tensor:
    """
    Voxelizes a mesh and returns the voxelized points.

    Args:
        mesh (pv.PolyData): mesh object.
        voxel_size (float): size of the voxel.

    Returns:
        torch.Tensor: voxelized points

    """
    mesh = pv.voxelize(mesh, voxel_size)
    return torch.tensor(mesh.points)


def extract_submesh_by_mask(mesh: pv.PolyData, mask: torch.Tensor) -> pv.PolyData:
    """
    Extracts a submesh from a mesh using a boolean mask.

    Args:
        mesh (pv.PolyData): mesh object.
        mask (torch.Tensor): boolean mask of the points to extract.

    Returns:
        pv.PolyData: extracted submesh.
    """
    indices = mask.nonzero().flatten().numpy()
    return mesh.extract_points(indices, adjacent_cells=False, include_cells=True)


@dataclass
class RobotModelConfig:
    """
    Configuration of the robot model. Contains the physical constants of the robot, its mass and geometry. 
    The input mesh is subsampled to a voxel grid with a specified voxel size.

    Attributes:
        robot_mass (float): mass of the robot in kg.
        joint_positions (Dict[str, List[float]]): positions of the joints in the robot's local frame.
        robot_points (torch.Tensor): position of the robot points in their local frame.
        mesh (o3d.geometry.TriangleMesh): mesh object of the robot.
        driving_parts (List[torch.Tensor]): list of masks for the driving parts of the robot, e.g. the tracks
        vel_max (float): maximum linear velocity of the robot in m/s.
        omega_max (float): maximum angular velocity of the robot in rad/s.
    """
    robot_type: Literal['tradr', 'marv', 'husky']
    voxel_size: float = 0.1

    def __post_init__(self):
        assert self.robot_type in IMPLEMENTED_ROBOTS, f"Robot {self.robot_type} not supported. Available robots: {IMPLEMENTED_ROBOTS}"
        self.load_robot_params_from_yaml()
        self.robot_points, self.driving_parts, self.mesh = self.create_robot_geometry()

    def load_robot_params_from_yaml(self) -> None:
        """
        Loads the robot parameters from a yaml file.

        Parameters:
            robot_type (str): Name of the robot.

        Returns:
            None
        """
        with open(YAMLDIR / f"{self.robot_type}.yaml", "r") as file:
            robot_params = yaml.safe_load(file)
        self.robot_mass = robot_params["robot_mass"]
        self.joint_positions = torch.tensor(robot_params["joint_positions"])
        self.vel_max = robot_params["vel_max"]
        self.omega_max = robot_params["omega_max"]
        self.driving_part_bboxes = [torch.tensor(bbox) for bbox in robot_params["driving_part_bboxes"]]
        self.body_bbox = torch.tensor(robot_params["body_bbox"])

    def create_robot_geometry(self) -> tuple[torch.Tensor, torch.Tensor, pv.PolyData]:
        """
        Returns the geometry of the robot model.

        Returns:
            torch.Tensor: Point cloud as vertices of the robot mesh (downsampled by voxel_size).
            torch.Tensor: Masks for the driving parts of the robot.
            o3d.geometry.TriangleMesh: Mesh object.
        """
        # Load the mesh and voxelized mesh
        mesh = pv.read(MESHDIR / f"{self.robot_type}.obj")
        robot_points = torch.tensor(mesh.points)
        driving_part_points = []
        # Create voxelized meshes for the driving parts
        for box in self.driving_part_bboxes:
            mask = points_in_oriented_box(robot_points[:, :2], box)
            driving_mesh = extract_submesh_by_mask(mesh, mask)
            driving_points = extract_surface_from_mesh(driving_mesh.delaunay_3d(), n_points=100)
            driving_part_points.append(driving_points)
        # Create voxelized mesh for the body
        body_mask = points_in_oriented_box(robot_points[:, :2], self.body_bbox)
        robot_body = extract_submesh_by_mask(mesh, body_mask)
        robot_body_points = voxelize_mesh(robot_body.delaunay_3d(), self.voxel_size)
        # Combine all voxelized meshes
        robot_points = torch.cat(driving_part_points + [robot_body_points])
        # Create masks for the driving parts
        driving_parts = []
        s = 0
        for i in range(len(driving_part_points)):
            mask = torch.zeros(robot_points.shape[0], dtype=torch.bool)
            mask[s:s + driving_part_points[i].shape[0]] = True
            s += driving_part_points[i].shape[0]
            driving_parts.append(mask)
        driving_parts = torch.stack(driving_parts, dim=0)
        return robot_points, driving_parts, mesh

    def visualize_robot(self, robot_points: Optional[torch.Tensor] = None, show_mesh: bool = True, return_geoms: bool = False) -> Optional[List[pv.PolyData]]:
        """
        Visualizes the robot in 3D using PyVista.
        """
        if robot_points is None:
            robot_points = self.robot_points

        driving_part_points = torch.cat([robot_points[mask] for mask in self.driving_parts])
        other_points = robot_points[torch.sum(self.driving_parts, dim=0) == 0]

        # Create PyVista point clouds
        driving_pcd = pv.PolyData(driving_part_points.numpy())
        other_pcd = pv.PolyData(other_points.numpy())

        # Create a PyVista Plotter
        plotter = pv.Plotter()
        if show_mesh:
            mesh = pv.PolyData(np.asarray(self.mesh.vertices), np.asarray(self.mesh.faces))
            plotter.add_mesh(mesh, color='white', opacity=0.5)

        # Add point clouds with colors
        plotter.add_mesh(other_pcd, color='blue', point_size=5, render_points_as_spheres=True)
        plotter.add_mesh(driving_pcd, color='red', point_size=5, render_points_as_spheres=True)

        # Add joint spheres
        for joint in self.joint_positions:
            sphere = pv.Sphere(center=joint.numpy(), radius=0.01)
            plotter.add_mesh(sphere, color='green')

        # Add grid lines
        grid_size = 1.0
        grid_spacing = 0.1
        for i in np.arange(-grid_size, grid_size + grid_spacing, grid_spacing):
            line_x = pv.Line(pointa=[i, -grid_size, 0], pointb=[i, grid_size, 0])
            line_y = pv.Line(pointa=[-grid_size, i, 0], pointb=[grid_size, i, 0])
            plotter.add_mesh(line_x, color='black', line_width=0.5)
            plotter.add_mesh(line_y, color='black', line_width=0.5)

        # Add coordinate axes
        plotter.show_axes()

        if return_geoms:
            # Collect all geometries if needed
            geometries = [self.mesh, other_pcd, driving_pcd]
            for joint in self.joint_positions:
                sphere = pv.Sphere(center=joint.numpy(), radius=0.01)
                geometries.append(sphere)
            return geometries
        else:
            print("X: red, Y: green, Z: blue")
            print("Robot has {} points".format(robot_points.shape[0]))
            plotter.show()


if __name__ == "__main__":
    robot_model = RobotModelConfig(robot_type="marv", voxel_size=0.1)
    robot_model.visualize_robot(show_mesh=False)
