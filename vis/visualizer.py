import torch
import time
from typing import Iterable, Dict
import mayavi.mlab as mlab
from flipper_training.engine.engine_state import PhysicsState, AuxEngineInfo, vectorize_iter_of_tensor_tuples
from flipper_training.configs import WorldConfig, PhysicsEngineConfig


def animate_trajectory(world_config: WorldConfig,
                       engine_config: PhysicsEngineConfig,
                       states: Iterable[PhysicsState],
                       aux_info: Iterable[AuxEngineInfo],
                       robot_index: int = 0,
                       vis_opts: Dict = {}):
    # Vectorize the states and aux_info
    states_vec = vectorize_iter_of_tensor_tuples(states)
    aux_info_vec = vectorize_iter_of_tensor_tuples(aux_info)
    # Figure
    window_size = vis_opts.get("window_size", (1280, 720))
    freq = vis_opts.get("freq", None)
    f = mlab.figure(size=window_size)
    # Static terrain
    terrain_vis = mlab.mesh(world_config.x_grid[robot_index], world_config.y_grid[robot_index], world_config.z_grid[robot_index], colormap="terrain", opacity=0.8)
    # Unpack the states
    xs = states_vec.x[:, robot_index].cpu().numpy()
    robot_points = aux_info_vec.global_robot_points[:, robot_index].cpu().numpy()
    fspring = aux_info_vec.F_spring[:, robot_index].cpu().numpy()
    contact_masks = aux_info_vec.in_contact[:, robot_index].bool().squeeze(-1).cpu().numpy()
    # Visualize the trajectory
    traj_vis = mlab.plot3d(xs[:, 0], xs[:, 1], xs[:, 2], color=(0, 1, 0), line_width=0.2)
    # Visualize the contact points and normals
    non_contact_point_vis = mlab.points3d(robot_points[0, :, 0], robot_points[0, :, 1], robot_points[0, :, 2], scale_factor=0.03, color=(0, 0, 0))
    contact_point_vis = mlab.points3d(robot_points[0, :, 0], robot_points[0, :, 1], robot_points[0, :, 2], scale_factor=0.03, color=(1, 0, 0))
    spring_forces_vis = mlab.quiver3d(robot_points[0, :, 0], robot_points[0, :, 1], robot_points[0, :, 2], fspring[0, :, 0], fspring[0, :, 1], fspring[0, :, 2], line_width=3.0, scale_factor=0.1, color=(1, 0, 0))
    # Animate the trajectory
    for i, (robot_point, fspring, contact_mask) in enumerate(zip(robot_points, fspring, contact_masks)):
        # Zero out non-contact points
        non_contact_points = robot_point.copy()
        non_contact_points[contact_mask] = 0
        contact_points = robot_point.copy()
        contact_points[~contact_mask] = 0
        fspring[~contact_mask] = 0
        # Update the visualization
        non_contact_point_vis.mlab_source.set(x=non_contact_points[:, 0], y=non_contact_points[:, 1], z=non_contact_points[:, 2])
        contact_point_vis.mlab_source.set(x=contact_points[:, 0], y=contact_points[:, 1], z=contact_points[:, 2])
        spring_forces_vis.mlab_source.set(x=contact_points[:, 0], y=contact_points[:, 1], z=contact_points[:, 2], u=fspring[:, 0], v=fspring[:, 1], w=fspring[:, 2])
        traj_vis.mlab_source.reset(x=xs[:i + 1, 0], y=xs[:i + 1, 1], z=xs[:i + 1, 2])
        mlab.process_ui_events()
        if freq is not None:
            if freq == "realtime":
                time.sleep(engine_config.dt)
            else:
                time.sleep(1 / freq)
    mlab.show()
