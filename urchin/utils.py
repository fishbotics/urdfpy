"""Utilities for URDF parsing.
"""
import os

from lxml import etree as ET
import numpy as np
import trimesh

from scipy.spatial.transform import Rotation 

def rpy_to_matrix(coords):
    """Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix.

    The roll-pitch-yaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are extrinsic transformation according to ROS:
    https://wiki.ros.org/geometry2/RotationMethods


    Parameters
    ----------
    coords : (3,) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).

    Returns
    -------
    R : (3,3) float
        The corresponding homogenous 3x3 rotation matrix.
    """
    return Rotation.from_euler("xyz", coords).as_matrix()


def matrix_to_rpy(R):
    """Convert a 3x3 transform matrix to roll-pitch-yaw coordinates.

    The roll-pitchRyaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are extrinsic transformation according to ROS:
    https://wiki.ros.org/geometry2/RotationMethods

    Parameters
    ----------
    R : (3,3) float
        A 3x3 homogenous rotation matrix.
    
    Returns
    -------
    coords : (3,) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
    """
    return Rotation.from_matrix(coords).as_euler("xyz")


def matrix_to_xyz_rpy(matrix):
    """Convert a 4x4 homogenous matrix to xyzrpy coordinates.

    Parameters
    ----------
    matrix : (4,4) float
        The homogenous transform matrix.

    Returns
    -------
    xyz_rpy : (6,) float
        The xyz_rpy vector.
    """
    xyz = matrix[:3,3]
    rpy = matrix_to_rpy(matrix[:3,:3])
    return np.hstack((xyz, rpy))


def xyz_rpy_to_matrix(xyz_rpy):
    """Convert xyz_rpy coordinates to a 4x4 homogenous matrix.

    Parameters
    ----------
    xyz_rpy : (6,) float
        The xyz_rpy vector.

    Returns
    -------
    matrix : (4,4) float
        The homogenous transform matrix.
    """
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3,3] = xyz_rpy[:3]
    matrix[:3,:3] = rpy_to_matrix(xyz_rpy[3:])
    return matrix


def parse_origin(node):
    """Find the ``origin`` subelement of an XML node and convert it
    into a 4x4 homogenous transformation matrix.

    Parameters
    ----------
    node : :class`lxml.etree.Element`
        An XML node which (optionally) has a child node with the ``origin``
        tag.

    Returns
    -------
    matrix : (4,4) float
        The 4x4 homogneous transform matrix that corresponds to this node's
        ``origin`` child. Defaults to the identity matrix if no ``origin``
        child was found.
    """
    matrix = np.eye(4, dtype=np.float64)
    origin_node = node.find('origin')
    if origin_node is not None:
        if 'xyz' in origin_node.attrib:
            matrix[:3,3] = np.fromstring(origin_node.attrib['xyz'], sep=' ')
        if 'rpy' in origin_node.attrib:
            rpy = np.fromstring(origin_node.attrib['rpy'], sep=' ')
            matrix[:3,:3] = rpy_to_matrix(rpy)
    return matrix


def unparse_origin(matrix):
    """Turn a 4x4 homogenous matrix into an ``origin`` XML node.

    Parameters
    ----------
    matrix : (4,4) float
        The 4x4 homogneous transform matrix to convert into an ``origin``
        XML node.

    Returns
    -------
    node : :class`lxml.etree.Element`
        An XML node whose tag is ``origin``. The node has two attributes:

        - ``xyz`` - A string with three space-delimited floats representing
          the translation of the origin.
        - ``rpy`` - A string with three space-delimited floats representing
          the rotation of the origin.
    """
    node = ET.Element('origin')
    node.attrib['xyz'] = '{} {} {}'.format(*matrix[:3,3])
    node.attrib['rpy'] = '{} {} {}'.format(*matrix_to_rpy(matrix[:3,:3]))
    return node


def get_filename(base_path, file_path, makedirs=False):
    """Formats a file path correctly for URDF loading.

    Parameters
    ----------
    base_path : str
        The base path to the URDF's folder.
    file_path : str
        The path to the file.
    makedirs : bool, optional
        If ``True``, the directories leading to the file will be created
        if needed.

    Returns
    -------
    resolved : str
        The resolved filepath -- just the normal ``file_path`` if it was an
        absolute path, otherwise that path joined to ``base_path``.
    """
    fn = file_path
    if not os.path.isabs(file_path):
        fn = os.path.join(base_path, file_path)
    if makedirs:
        d, _ = os.path.split(fn)
        if not os.path.exists(d):
            os.makedirs(d)
    return fn


def load_meshes(filename):
    """Loads triangular meshes from a file.

    Parameters
    ----------
    filename : str
        Path to the mesh file.

    Returns
    -------
    meshes : list of :class:`~trimesh.base.Trimesh`
        The meshes loaded from the file.
    """
    meshes = trimesh.load(filename)

    # If we got a scene, dump the meshes
    if isinstance(meshes, trimesh.Scene):
        meshes = list(meshes.dump())
        meshes = [g for g in meshes if isinstance(g, trimesh.Trimesh)]

    if isinstance(meshes, (list, tuple, set)):
        meshes = list(meshes)
        if len(meshes) == 0:
            raise ValueError('At least one mesh must be pmeshesent in file')
        for r in meshes:
            if not isinstance(r, trimesh.Trimesh):
                raise TypeError('Could not load meshes from file')
    elif isinstance(meshes, trimesh.Trimesh):
        meshes = [meshes]
    else:
        raise ValueError('Unable to load mesh from file')

    return meshes


def configure_origin(value):
    """Convert a value into a 4x4 transform matrix.

    Parameters
    ----------
    value : None, (6,) float, or (4,4) float
        The value to turn into the matrix.
        If (6,), interpreted as xyzrpy coordinates.

    Returns
    -------
    matrix : (4,4) float or None
        The created matrix.
    """
    if value is None:
        value = np.eye(4, dtype=np.float64)
    elif isinstance(value, (list, tuple, np.ndarray)):
        value = np.asanyarray(value, dtype=np.float64)
        if value.shape == (6,):
            value = xyz_rpy_to_matrix(value)
        elif value.shape != (4,4):
            raise ValueError('Origin must be specified as a 4x4 '
                             'homogenous transformation matrix')
    else:
        raise TypeError('Invalid type for origin, expect 4x4 matrix')
    return value
