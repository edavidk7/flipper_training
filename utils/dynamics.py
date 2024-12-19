import torch

__all__ = ['inertia_tensor']


@torch.compile
def inertia_tensor(pointwise_mass: torch.Tensor, points: torch.Tensor):
    """
        Compute the inertia tensor for a rigid body represented by point masses.

        Parameters:

            mass (torch.Tensor): masses of the points in shape (N).
            points (torch.Tensor): A tensor of shape (B, N, 3) representing the points of the body.
                                Each point contributes equally to the total mass.

        Returns:
            torch.Tensor: A 3x3 inertia tensor matrix.
        """
    points2mass = points * points * pointwise_mass[:, None]  # fuse this operation
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    x2m = points2mass[..., 0]
    y2m = points2mass[..., 1]
    z2m = points2mass[..., 2]
    Ixx = (y2m + z2m).sum(dim=-1)
    Iyy = (x2m + z2m).sum(dim=-1)
    Izz = (x2m + y2m).sum(dim=-1)
    Ixy = -(pointwise_mass[None, :] * x * y).sum(dim=-1)
    Ixz = -(pointwise_mass[None, :] * x * z).sum(dim=-1)
    Iyz = -(pointwise_mass[None, :] * y * z).sum(dim=-1)
    # Construct the inertia tensor matrix
    I = torch.stack([
        torch.stack([Ixx, Ixy, Ixz], dim=-1),
        torch.stack([Ixy, Iyy, Iyz], dim=-1),
        torch.stack([Ixz, Iyz, Izz], dim=-1)
    ], dim=-2)  #
    return I
