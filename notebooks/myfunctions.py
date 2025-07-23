# This file contains the common functions used in many notebooks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def get_coordinates_displacement(df_nodes, df_defs, dp):
    """"
    Get node coordinates and displacements for a particular design point
    """
    
    # Get nodes and displacements of the mode
    nodes = df_nodes[df_nodes['dp_no'] == dp]
    defs = df_defs[df_defs['dp_no'] == dp]

     # Merge node coordinates and displacements
    uy_nodes = pd.merge(nodes, defs, how='inner', on='node_no', suffixes=['', '_1'])   

    return uy_nodes


def get_mode_frequency(df_modes, dp, mode_no):
    frequency = df_modes.loc[(df_modes['dp_no'] == dp) & (df_modes['mode_no'] == mode_no), 'freq'].item()
    return frequency


def plot_displacement(df_modes, df_nodes, df_defs, dp, mode_no, set_zlim=False):
    """
    Visualization of the displacement of the output surface
    Plot 1: Contour plot 2D
    Plot 2: Contour plot 3D
    # Arguments
    dp:         design point
    mode_no:    mode number
    set_zlim:   set min z to 0
    """

    # Get frequency of the mode
    frequency = get_mode_frequency(df_modes, dp, mode_no)

    # Column name of the displacement
    col_disp = 'mode' + str(mode_no)

    # Get nodes and displacements of the mode
    uy_nodes = get_coordinates_displacement(df_nodes, df_defs, dp)

    # two-dimensional interpolation with scipy.interpolate.griddata
    # https://scipython.com/book/chapter-8-scipy/examples/two-dimensional-interpolation-with-scipyinterpolategriddata/
    px = uy_nodes['x_coord']
    pz = uy_nodes['z_coord']
    disp = np.array(uy_nodes[col_disp])
    x = np.linspace(px.min(), px.max(), 50)
    z = np.linspace(pz.min(), pz.max(), 50)
    X, Z = np.meshgrid(x, z)
    Disp = griddata((px, pz), disp, (X, Z), method='cubic')
    title = format('Design point %i\n Mode #%i (%0.0f Hz)' % (dp, mode_no, frequency))

    # 2d plot
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis('equal')
    cs = ax1.contourf(X, Z, Disp, levels=15, cmap='jet')
    ax1.set_title(title)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    
    # 3d plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Z, Disp, cmap='jet', shade=True, rstride=1, cstride=1)
    ax2.set_title(title)
    if set_zlim:
        ax2.set_zlim([0, max(uy_nodes[col_disp])])
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Z [mm]')
    plt.colorbar(cs)
    plt.show()

    return X, Z, Disp


def plot_2d_norm_disp(df_modes, df_nodes, df_defs, dp, mode_no, set_zlim=False):
    """
    Visualization of the displacement of the output square [-1:1] as heatmap only
    # Arguments
    dp:         design point
    mode_no:    mode number
    set_zlim:   set min z to 0
    """

    # Get frequency of the mode
    frequency = get_mode_frequency(df_modes, dp, mode_no)

    # Column name of the displacement
    col_disp = 'mode' + str(mode_no)

    # Get nodes and displacements of the mode
    uy_nodes = get_coordinates_displacement(df_nodes, df_defs, dp)
    px = uy_nodes['x_coord_n']
    pz = uy_nodes['z_coord_n']
    disp = np.array(uy_nodes[col_disp])
    # If the displacement is negative at the first node, reverse the displacement vector (makes sense particularly for the longitudinal mode)
    if disp[0] < 0:
        disp = -disp

    # two-dimensional interpolation with scipy.interpolate.griddata
    x = np.linspace(px.min(), px.max(), 20)
    z = np.linspace(pz.min(), pz.max(), 20)
    X, Z = np.meshgrid(x, z)
    Disp_near = griddata((px, pz), disp, (X, Z), method='nearest')
    Disp_cub = griddata((px, pz), disp, (X, Z), method='cubic')
    title = format('Design point %i\n Mode #%i (%0.0f Hz)' % (dp, mode_no, frequency))

    # 2d plot
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis('equal')
    cs = ax1.contourf(X, Z, Disp_cub, levels=15, cmap='jet')
    ax1.set_title(title)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')

    # 3d plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.axis('equal')
    ax2.plot_surface(X, Z, Disp_cub, cmap='jet', shade=True, rstride=1, cstride=1)
    ax2.set_title(title)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_xlim([-1.1, 1.1])
    ax2.set_ylim([-1.1, 1.1])
    if set_zlim:
        ax2.set_zlim([0, max(disp)])
    else:
        ax2.set_zlim([min(disp), max(disp)])
    plt.colorbar(cs)
    plt.show()

    return X, Z, Disp_near
    
    