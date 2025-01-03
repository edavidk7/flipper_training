import torch
import pyvista as pv
import pyacvd

__all__ = ["cluster_points", "extract_surface_from_mesh", "voxelize_mesh", "extract_submesh_by_mask"]


def cluster_points(points: torch.Tensor, n_points: int, **clus_opts) -> torch.Tensor:
    """
    Clusters a point cloud to n_points using the pyacvd library.

    Args:
        points (torch.Tensor): point cloud.
        n_points (int): number of points to cluster to.

    Returns:
        torch.Tensor: clustered points.
    """
    surf = pv.PolyData(points.numpy()).delaunay_3d(progress_bar=True)
    surf = surf.extract_geometry().triangulate()
    clus = pyacvd.Clustering(surf)
    clus.cluster(n_points, **clus_opts)
    return torch.tensor(clus.cluster_centroid)


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
