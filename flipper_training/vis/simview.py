import torch
from simview import SimViewTerrain, SimViewBody, BodyShapeType, OptionalBodyStateAttribute, SimViewBodyState
from flipper_training.configs.terrain_config import TerrainConfig
from flipper_training.configs.robot_config import RobotModelConfig
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.geometry import compose_quaternion_affine, euler_to_quaternion

__all__ = ["simview_terrain_from_config", "simview_bodies_from_robot_config", "physics_state_to_simview_body_states"]


def simview_terrain_from_config(config: TerrainConfig, is_singleton: bool = False) -> SimViewTerrain:
    """
    Create a SimViewTerrain object from a TerrainConfig object.

    Args:
        config (TerrainConfig): The TerrainConfig object containing the terrain configuration.
        is_singleton (bool): If True, the terrain is considered just one for all robots. If False, each robot has its own terrain.
    """
    return SimViewTerrain.create(
        heightmap=config.z_grid,
        normals=config.normals,
        x_lim=(-config.max_coord, config.max_coord),
        y_lim=(-config.max_coord, config.max_coord),
        is_singleton=is_singleton,
    )


def simview_bodies_from_robot_config(config: RobotModelConfig) -> list[SimViewBody]:
    # Main robot body
    robot_body = SimViewBody.create(
        name="Body",
        body_type=BodyShapeType.POINTCLOUD,
        points=config.body_points,
        available_attributes=[
            OptionalBodyStateAttribute.ANGULAR_VELOCITY,
            OptionalBodyStateAttribute.VELOCITY,
            OptionalBodyStateAttribute.TORQUE,
            OptionalBodyStateAttribute.CONTACTS,
            OptionalBodyStateAttribute.FORCE,
        ],
    )
    bodies = [robot_body]
    # Driving parts
    for i in range(config.num_driving_parts):
        driving_part = SimViewBody.create(
            name=config.driving_part_names[i],
            body_type=BodyShapeType.POINTCLOUD,
            points=config.joint_local_driving_part_pts[i],
            available_attributes=[
                OptionalBodyStateAttribute.ANGULAR_VELOCITY,
                OptionalBodyStateAttribute.VELOCITY,
                OptionalBodyStateAttribute.CONTACTS,
            ],
        )
        bodies.append(driving_part)
    return bodies


def physics_state_to_simview_body_states(
    robot_config: RobotModelConfig,
    physics_state: PhysicsState,
    physics_state_der: PhysicsStateDer,
    control: torch.Tensor,
) -> list[SimViewBodyState]:
    """
    Convert a PhysicsState object to a list of SimViewBodyState objects.

    Args:
        robot_config (RobotModelConfig): The RobotModelConfig object containing the robot configuration.
        physics_state (PhysicsState): The PhysicsState object containing the physics state.

    Returns:
        list[dict]: A list of dictionaries representing the SimViewBodyState objects.
    """
    # Split contacts
    driving_part_contacts, body_contacts = torch.split(
        physics_state_der.in_contact.squeeze().bool(), robot_config.num_driving_parts * robot_config.points_per_driving_part, dim=1
    )
    # Body state
    t_body = physics_state.x
    q_body = physics_state.q
    body_state = SimViewBodyState(
        body_name="Body",
        position=t_body,
        orientation=q_body,
        optional_attributes={
            OptionalBodyStateAttribute.FORCE: physics_state_der.xdd * robot_config.total_mass,
            OptionalBodyStateAttribute.VELOCITY: physics_state.xd,
            OptionalBodyStateAttribute.ANGULAR_VELOCITY: physics_state.omega,
            OptionalBodyStateAttribute.TORQUE: physics_state_der.torque,
            OptionalBodyStateAttribute.CONTACTS: body_contacts,
        },
    )
    states = [body_state]
    # Driving part states
    driving_part_contacts = driving_part_contacts.view(-1, robot_config.num_driving_parts, robot_config.points_per_driving_part)
    rot_y_vec = torch.tensor([[0.0, 1.0, 0.0]], device=physics_state.x.device)  # shape (1,3)
    for i in range(robot_config.num_driving_parts):
        theta = physics_state.thetas[..., i]
        theta_d = physics_state_der.thetas_d[..., i]
        q_local = euler_to_quaternion(torch.zeros_like(theta), theta, torch.zeros_like(theta))
        t_local = robot_config.joint_positions[i].unsqueeze(0)
        t_part, q_part = compose_quaternion_affine(
            t_body,
            q_body,
            t_local,
            q_local,
        )
        driving_part_state = SimViewBodyState(
            body_name=robot_config.driving_part_names[i],
            position=t_part,
            orientation=q_part,
            optional_attributes={
                OptionalBodyStateAttribute.VELOCITY: robot_config.driving_direction.unsqueeze(0) * control[..., i, None],
                OptionalBodyStateAttribute.ANGULAR_VELOCITY: rot_y_vec * theta_d[..., None],
                OptionalBodyStateAttribute.CONTACTS: driving_part_contacts[:, i],
            },
        )
        states.append(driving_part_state)
    return states
