import numpy as np
import matplotlib.pyplot as plt

def Rx(angle):
    """
    Rotation matrix about the x-axis.

    Parameters
    ----------
    angle : float
        Angle of rotation in radians.

    Returns
    -------
    numpy.ndarray
        Rotation matrix.
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), np.sin(angle)],
        [0, -np.sin(angle), np.cos(angle)]
    ])

def Ry(theta):
    """
    Rotation matrix about the y-axis.

    Parameters
    ----------
    theta : float
        Angle of rotation in radians.

    Returns
    -------
    numpy.ndarray
        Rotation matrix.
    """
    
    return np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0, 1, 0],
        [np.sin(theta), 0, np.cos(theta)]
    ])

def Rz(phi):
    """
    Rotation matrix about the z-axis.

    Parameters
    ----------
    phi : float
        Angle of rotation in radians.

    Returns
    -------
    numpy.ndarray
        Rotation matrix.
    """
    return np.array([[np.cos(phi), np.sin(phi), 0],
                     [-np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])

def ellipsoid_surface(
    center_x: float, center_y: float, center_z: float,
    radius_x: float, radius_y: float, radius_z: float,
    theta: float, phi: float, rotation: float,
    angle_units: str = 'degrees'
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the surface coordinates of an ellipsoid.

    Parameters
    ----------
    center_x, center_y, center_z : float
        Coordinates of the center of the ellipsoid.
    radius_x, radius_y, radius_z : float
        Radii of the ellipsoid.
    dip, strike : float
        Spherical angles for rotation.
    angle_units : str, optional
        Units of the dip and strike angles. Default is 'degrees'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Coordinates of the ellipsoid. (x, y, z)
    """
    
    if angle_units not in ('degrees', 'radians'):
        raise ValueError(f"angle_units: must be one of ['degrees', 'radians'].")

    if angle_units.lower() == 'degrees':
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        rotation = np.deg2rad(rotation)

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = radius_x * np.outer(np.cos(u), np.sin(v))
    y = radius_y * np.outer(np.sin(u), np.sin(v))
    z = radius_z * np.outer(np.ones_like(u), np.cos(v))

    coordinates = np.c_[np.reshape(x, [x.size, 1]),
                        np.reshape(y, [y.size, 1]),
                        np.reshape(z, [z.size, 1])]

    rotated_coordinates = coordinates @ Rz(-rotation) @ Ry(theta) @ Rz(-phi)
    rotated_coordinates += np.c_[[center_x], [center_y], [center_z]]

    x = np.reshape(rotated_coordinates[:, 0], (100, 100))
    y = np.reshape(rotated_coordinates[:, 1], (100, 100))
    z = np.reshape(rotated_coordinates[:, 2], (100, 100))

    return x, y, z

def set_limits(x_coords, y_coords, z_coords):
    
    # Calculate the axis limits to ensure the aspect ratio is 1:1:1
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()

    x_mid = (x_max + x_min)/2
    y_mid = (y_max + y_min)/2
    z_mid = (z_max + z_min)/2

    # Find the maximum range
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

    xlim=(x_mid - max_range, x_mid + max_range)
    ylim=(y_mid - max_range, y_mid + max_range)
    zlim=(z_mid - max_range, z_mid + max_range)

    return xlim, ylim, zlim

def plot_ellipsoid(x_coords, y_coords, z_coords):

    # plot the ellipse
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'proj_type': 'ortho'})
    
    ax.view_init(azim=-135)
    ax.plot_surface(x_coords, y_coords, z_coords, rstride=5, cstride=5, cmap='plasma', linewidth=0.25, edgecolor='k')

    x_lim, y_lim, z_lim = set_limits(x_coords, y_coords, z_coords)
    ax.set(xlabel='X [km]', ylabel='Y [km]', zlabel='Z [km]', xlim=x_lim, ylim=y_lim, zlim=z_lim)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.get_label().set_fontsize(8)
        axis.set_tick_params(labelsize=8, top=True, right=True)

    # Set equal aspect ratio
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.zaxis._axinfo['juggled'] = (1,2,0)
    
    plt.tight_layout()
    plt.show()

def plot_ellipsoid_planview(x_coords, y_coords, z_coords):

    # plot the ellipse
    fig, axs = plt.subplots(1, 3, figsize=plt.figaspect(1/3), subplot_kw={'projection': '3d', 'proj_type': 'ortho'})

    for ax in axs:
        ax.plot_surface(x_coords, y_coords, z_coords, rstride=5, cstride=5, cmap='plasma', linewidth=0.25, edgecolor='k')

        x_lim, y_lim, z_lim = set_limits(x_coords, y_coords, z_coords)
        ax.set(xlabel='X [km]', ylabel='Y [km]', zlabel='Z [km]', xlim=x_lim, ylim=y_lim, zlim=z_lim)

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.get_label().set_fontsize(8)
            axis.set_tick_params(labelsize=8, top=True, right=True)

        ax.set_box_aspect([1.0, 1.0, 1.0])

    # Specific customizations for each subplot
    # First subplot: Z on the vertical axis, X on the horizontal axis
    axs[0].view_init(azim=-90, elev=90)
    axs[0].set(zlabel='')
    axs[0].set_zticks([])
    axs[0].zaxis.set_ticklabels([])
    axs[0].xaxis._axinfo['juggled'] = (1, 0, 2)
    axs[0].yaxis._axinfo['juggled'] = (0, 1, 2)

    # Second subplot: Y on the vertical axis, X on the horizontal axis
    axs[1].view_init(azim=-90, elev=0)
    axs[1].set(ylabel='')
    axs[1].set_yticks([])
    axs[1].yaxis.set_ticklabels([])
    axs[1].xaxis._axinfo['juggled'] = (1, 0, 2)
    axs[1].zaxis._axinfo['juggled'] = (0, 2, 1)

    # Third subplot: Y on the vertical axis, Z on the horizontal axis
    axs[2].view_init(azim=0, elev=0)
    axs[2].set(xlabel='')
    axs[2].set_xticks([])
    axs[2].xaxis.set_ticklabels([])
    axs[2].yaxis._axinfo['juggled'] = (0, 1, 2)
    axs[2].zaxis._axinfo['juggled'] = (0, 2, 1)

    plt.tight_layout()
    plt.show()