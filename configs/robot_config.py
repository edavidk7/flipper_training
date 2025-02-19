from typing import Literal, Optional
import torch
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from flipper_training.configs.base_config import BaseConfig
import pyvista as pv
import yaml
import hashlib
from flipper_training.utils.geometry import (
    points_in_oriented_box,
    pointcloud_bounding_volume,
    extract_top_plane_from_box,
)
from flipper_training.utils.meshes import (
    extract_surface_from_mesh,
    extract_submesh_by_mask,
    voxelize_mesh,
)
from flipper_training.utils.flipper_modeling import (
    get_flipper_pointwise_vels,
    FlipperWheels,
)

np.random.seed(0)

ROOT = Path(__file__).parent.parent
MESHDIR = ROOT / "meshes"
YAMLDIR = ROOT / "robots"
POINTCACHE = ROOT / ".robot_cache"
IMPLEMENTED_ROBOTS = ["marv"]


@dataclass
class RobotModelConfig(BaseConfig):
    """
    Configuration of the robot model. Contains the physical constants of the robot, its mass and geometry.
    The input mesh is subsampled to a voxel grid with a specified voxel size.

    Attributes:
        mass (float): mass of the robot in kg.
        pointwise_mass (torch.Tensor): mass of each point in the robot.
        joint_positions (torch.Tensor): positions of the joints in the robot's local frame.
        robot_points (torch.Tensor): position of the robot points in their local frame.
        driving_parts (torch.Tensor): tensor of masks for the driving parts of the robot, e.g. the tracks
        driving_part_density (float): density of the driving parts relative to the body.
        omega_max (float): maximum angular velocity of the robot in rad/s.
        vel_max (float): maximum linear velocity of the robot's tracks in m/s.
        driving_part_bboxes (torch.Tensor): bounding boxes of the driving parts.
        radius (float): radius of the robot.
        body_bbox (torch.Tensor): bounding box of the robot body.
        robot_type (Literal['tradr', 'marv', 'husky']): type of the robot.
        voxel_size (float): size of the voxel grid for the body.
        points_per_driving_part (int): number of points per driving part for clustering.
        bounding_volume_eps (float): epsilon for bounding volume computation.
    """

    robot_type: Literal["tradr", "marv", "husky"]
    voxel_size: float = 0.08
    points_per_driving_part: int = 192
    bounding_volume_eps: float = 1e-2
    wheel_assignment_margin: float = 0.02

    def __post_init__(self):
        assert self.robot_type in IMPLEMENTED_ROBOTS, (
            f"Robot {self.robot_type} not supported. Available robots: {IMPLEMENTED_ROBOTS}"
        )
        self.load_robot_params_from_yaml()
        self.create_robot_geometry()

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
            canonical = yaml.dump(robot_params, sort_keys=True)  # ensure consistent order
        self.yaml_hash = hashlib.sha256(canonical.encode()).hexdigest()
        self.mass = robot_params["mass"]
        self.driving_part_density = robot_params["driving_part_density"]
        self.joint_positions = torch.tensor(robot_params["joint_positions"])
        self.vel_max = robot_params["vel_max"]
        self.omega_max = self._compute_omega_max()
        self.joint_limits = torch.tensor(robot_params["joint_limits"]).T  # shape (2, n_joints)
        self.joint_movable_mask = torch.tensor(
            robot_params["joint_movable_mask"], dtype=torch.bool
        )  # shape (n_joints,)
        self.joint_vel_limits = torch.tensor(robot_params["joint_vel_limits"])  # shape (n_joints,)
        self.driving_part_bboxes = torch.stack(
            [torch.tensor(bbox) for bbox in robot_params["driving_part_bboxes"]], dim=0
        )
        self.body_bbox = torch.tensor(robot_params["body_bbox"])
        self.flipper_wheels = [
            FlipperWheels(
                wheel_radii=torch.tensor(wheel["radius"]),
                wheel_positions=torch.tensor(wheel["position"]),
                rotation_vector=torch.tensor(wheel["rot_axis"]),
            )
            for wheel in robot_params["wheels"]
        ]
        self.driving_direction = torch.tensor(robot_params["forward_direction"])

    @property
    def _descr_str(self) -> str:
        return f"{self.robot_type}_{self.voxel_size:.3f}_{self.points_per_driving_part}.pt"

    def print_info(self) -> None:
        print(f"Robot has {self.robot_points.shape[0]} points")

    def _compute_omega_max(self) -> float:
        joint_y = self.joint_positions[..., 1]
        w = abs(joint_y.max() - joint_y.min())
        return 2 * self.vel_max / w

    def load_from_cache(self) -> bool:
        """
        Loads the robot parameters from a cache file.

        Returns:
            bool: True if the cache file exists, False otherwise.
        """
        confpath = POINTCACHE / self._descr_str
        if confpath.exists():
            print(f"Loading robot model from cache: {confpath}")
            confdict = torch.load(confpath)
            if confdict["yaml_hash"] != self.yaml_hash:
                print("Hash mismatch, re-creating robot model")
                return False
            for key, val in confdict.items():
                setattr(self, key, val)
            self.print_info()
            return True
        return False

    def save_to_cache(self) -> None:
        """
        Saves the robot parameters to a cache file.
        """
        confpath = POINTCACHE / self._descr_str
        if not confpath.parent.exists():
            confpath.parent.mkdir(parents=True)
        print(f"Saving robot model to cache: {confpath}")
        confdict = {
            "robot_points": self.robot_points,
            "driving_part_masks": self.driving_part_masks,
            "body_mask": self.body_mask,
            "pointwise_mass": self.pointwise_mass,
            "joint_positions": self.joint_positions,
            "body_bbox": self.body_bbox,
            "driving_part_bboxes": self.driving_part_bboxes,
            "radius": self.radius,
            "yaml_hash": self.yaml_hash,
            "body_bounding_volume": self.body_bounding_volume,
            "driving_part_bounding_volumes": self.driving_part_bounding_volumes,
        }
        torch.save(confdict, confpath)

    def get_controls(self, vw: torch.Tensor) -> torch.Tensor:
        """Compute the speeds of the driving parts based on the robot's linear and angular velocities.

        The output is clamped to the maximum velocity of the robot's tracks.

        Args:
            vw (torch.Tensor): linear and angular velocity of the robot. shape (n_robots, 2)

        Returns:
            torch.Tensor: speeds of the driving parts. shape (n_robots, n_driving_parts)
        """
        joint_y = self.joint_positions[..., 1]
        return torch.clamp(vw[..., 0] + joint_y * vw[..., 1], min=-self.vel_max, max=self.vel_max)

    def create_robot_geometry(self) -> None:
        """
        Returns the geometry of the robot model.

        Returns:
            torch.Tensor: Point cloud as vertices of the robot mesh (downsampled by voxel_size).
            torch.Tensor: Masks for the driving parts of the robot.
            o3d.geometry.TriangleMesh: Mesh object.
        """
        if self.load_from_cache():
            return
        # Load the mesh and voxelized mesh
        mesh = pv.read(MESHDIR / f"{self.robot_type}.obj")
        robot_points = torch.tensor(mesh.points)
        driving_part_points = []
        # Create surface meshes for the driving parts
        for box in self.driving_part_bboxes:
            mask = points_in_oriented_box(robot_points[:, :2], box)
            driving_mesh = extract_submesh_by_mask(mesh, mask)
            driving_points = extract_surface_from_mesh(
                driving_mesh, n_points=self.points_per_driving_part
            )
            driving_part_points.append(driving_points)
        # Create voxelized mesh for the body
        body_mask = points_in_oriented_box(robot_points[:, :2], self.body_bbox)
        robot_body = extract_submesh_by_mask(mesh, body_mask)
        robot_body_points = voxelize_mesh(robot_body.delaunay_3d(), self.voxel_size)
        # Combine all points
        robot_points = torch.cat(driving_part_points + [robot_body_points]).float()
        # Create masks for the driving parts
        driving_part_masks = []
        s = 0
        pointwise_relative_density = torch.ones(robot_points.shape[0])
        for i in range(len(driving_part_points)):
            mask = torch.zeros(robot_points.shape[0], dtype=torch.bool)
            mask[s : s + driving_part_points[i].shape[0]] = True
            s += driving_part_points[i].shape[0]
            driving_part_masks.append(mask)
            pointwise_relative_density[mask] = self.driving_part_density
        driving_part_masks = torch.stack(driving_part_masks, dim=0)
        # Calculate pointwise mass and center of gravity
        pointwise_mass = self.mass * pointwise_relative_density / pointwise_relative_density.sum()
        # Set
        self.body_bounding_volume = pointcloud_bounding_volume(
            robot_body_points, eps=self.bounding_volume_eps
        ).float()
        self.driving_part_bounding_volumes = torch.stack(
            [
                pointcloud_bounding_volume(p, eps=self.bounding_volume_eps)
                for p in driving_part_points
            ],
            dim=0,
        ).float()
        self.pointwise_mass = pointwise_mass
        self.robot_points = robot_points
        self.driving_part_masks = driving_part_masks.float()
        self.body_mask = (torch.sum(driving_part_masks, dim=0) == 0).float()
        self.radius = torch.sqrt((robot_points**2).sum(dim=1).max())
        # Save to cache
        self.save_to_cache()
        self.print_info()

    @property
    def num_joints(self) -> int:
        return self.joint_positions.shape[0]

    @property
    def upper_bounding_planes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The upper bounding planes of the robot's bounding volumes.

        The planes are returned per-point in format (normal_vectors (n_points, 3), point_on_plane (n_points, 3)).
        """
        normals = torch.zeros_like(self.robot_points)
        points = torch.zeros_like(self.robot_points)
        body_normal, body_point = extract_top_plane_from_box(self.body_bounding_volume)
        normals[self.body_mask.bool()] = body_normal
        points[self.body_mask.bool()] = body_point
        for i, mask in enumerate(self.driving_part_masks):
            normal, point = extract_top_plane_from_box(self.driving_part_bounding_volumes[i])
            normals[mask.bool()] = normal
            points[mask.bool()] = point
        return normals, points

    def visualize_robot(
        self,
        robot_points: Optional[torch.Tensor] = None,
        grid_size: float = 1.0,
        grid_spacing: float = 0.1,
        return_plotter: bool = False,
    ) -> None:
        """
        Visualizes the robot in 3D using PyVista.

        Parameters:
            robot_points (torch.Tensor): Point cloud as vertices of the robot mesh (downsampled by voxel_size).
            grid_size (float): Size of the grid.
            grid_spacing (float): Spacing of the grid.
        """
        if robot_points is None:
            robot_points = self.robot_points
        robot_points = robot_points.cpu()
        driving_part_points = torch.cat(
            [robot_points[mask] for mask in self.driving_part_masks.cpu().bool()]
        )
        other_points = robot_points[torch.sum(self.driving_part_masks.cpu(), dim=0) == 0]
        driving_pcd = pv.PolyData(driving_part_points.numpy())
        other_pcd = pv.PolyData(other_points.numpy())
        plotter = pv.Plotter()
        # body and driving parts
        plotter.add_mesh(other_pcd, color="blue", point_size=5, render_points_as_spheres=True)
        plotter.add_mesh(driving_pcd, color="red", point_size=5, render_points_as_spheres=True)
        # joint positions
        for joint in self.joint_positions.cpu():
            sphere = pv.Sphere(center=joint.numpy(), radius=0.01)
            plotter.add_mesh(sphere, color="green")
        # bounding volumes - body
        body_bounding_volume_corners = self.body_bounding_volume.cpu().detach()
        body_bounding_volume_mesh = pv.PolyData(body_bounding_volume_corners.numpy()).delaunay_3d()
        plotter.add_mesh(body_bounding_volume_mesh, color="blue", line_width=2, opacity=0.05)
        # bounding volumes - driving parts
        for (
            driving_part_bounding_volume_corners
        ) in self.driving_part_bounding_volumes.cpu().detach():
            driving_part_bounding_volume_mesh = pv.PolyData(
                driving_part_bounding_volume_corners.numpy()
            ).delaunay_3d()
            plotter.add_mesh(
                driving_part_bounding_volume_mesh, color="red", line_width=2, opacity=0.05
            )
        # origin of robot's local frame
        sphere = pv.Sphere(center=np.zeros(3), radius=0.025)
        plotter.add_mesh(sphere, color="yellow")
        # grid
        for i in np.arange(-grid_size, grid_size + grid_spacing, grid_spacing):
            line_x = pv.Line(pointa=[i, -grid_size, 0], pointb=[i, grid_size, 0])
            line_y = pv.Line(pointa=[-grid_size, i, 0], pointb=[grid_size, i, 0])
            plotter.add_mesh(line_x, color="black", line_width=0.5)
            plotter.add_mesh(line_y, color="black", line_width=0.5)
        # show
        print("Robot has {} points".format(robot_points.shape[0]))
        plotter.show_axes()
        if not return_plotter:
            plotter.show()
        else:
            return plotter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_type", type=str, default="marv", help="Type of the robot")
    parser.add_argument("--voxel_size", type=float, default=0.08, help="Voxel size")
    parser.add_argument(
        "--points_per_driving_part", type=int, default=192, help="Number of points per driving part"
    )
    args = parser.parse_args()
    robot_model = RobotModelConfig(
        robot_type=args.robot_type,
        voxel_size=args.voxel_size,
        points_per_driving_part=args.points_per_driving_part,
    )
    robot_model.visualize_robot()
