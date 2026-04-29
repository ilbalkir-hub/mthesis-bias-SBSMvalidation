"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Provides interactive and static visualisations for criticality spaces.
    Plots are sliced to two free dimensions; remaining dimensions are fixed at
    user-specified values. Output formats include interactive Plotly HTML and
    static Matplotlib PNG/PDF.

    - key role: Visual inspection of criticality surfaces, selected sample
                distributions, discretised class boundaries, and boundary masks.
    - dependency: space.Space, matplotlib, plotly, numpy
    - output: Interactive Plotly figures and/or Matplotlib figures; optional
              file export (PNG, PDF, HTML).

Available plots:
    plot_3d_two_varied          — 3-D continuous surface (Plotly)
    plot_3d_two_varied_discrete — 3-D discrete class surface (Plotly)
    plot_3d_three_varied        — 3-D scatter coloured by criticality (Matplotlib)
    plot_top_down_2d            — 2-D heatmap with sample overlay (Matplotlib)
    plot_2d_decision_boundary   — 2-D class contours with sample overlay (Matplotlib)
    plot_boundary_mask          — binary boundary region mask (Matplotlib)

Usage:
    # Accessed via the lazy property on Space:
    space.visualizer.plot_3d_two_varied(show_critplane=True, fixed_values=[None, None, 5])
    space.visualizer.plot_top_down_2d(save_path="output/heatmap", format="png")
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib
import plotly.graph_objects as go
import numpy as np
import warnings
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from space import Space

class SpaceVisualizer:
    """
    Visualisation helper for a criticality Space instance.
    All plot methods operate on the space's meshgrid and selected_points;
    high-dimensional spaces are projected to 2-D slices by fixing non-plotted
    dimensions at specified values.
    """

    def __init__(self, space: "Space"):
        """
        Parameters
        ----------
        space : Space
            The criticality space instance to visualise.
            Accessed via the lazy ``space.visualizer`` property.
        """
        self.space = space

    def plot_3d_two_varied(self, dim1=0, dim2=1, fixed_values=None, show_samples=True, show_critplane=True, criticality_thresholds: List[float] = None, save_path=None, format='html', top_down=True, simulate_bias=False):
        """
        Interactive 3-D surface plot of the continuous criticality surface.
        Two dimensions are varied freely; all others are fixed.

        Parameters
        ----------
        dim1, dim2 : int
            Indices of the two free dimensions (X and Y axes).
        fixed_values : list, optional
            Per-dimension fixed values for non-plotted axes. Use None for
            free dimensions. Defaults to index 0 of the meshgrid.
        show_samples : bool
            Overlay selected sample points projected onto the surface.
        show_critplane : bool
            Render semi-transparent horizontal planes at each criticality threshold.
        criticality_thresholds : list of float, optional
            Override the space's default thresholds for threshold planes.
        save_path : str, optional
            File path (without extension) for export. If None, no file is saved.
        format : str
            Export format: ``'html'`` (interactive) or ``'png'`` (static).
        """
        if criticality_thresholds is None:
            crit_tres = self.space.criticality_thresholds
        else:
            crit_tres = sorted(criticality_thresholds)

        if len(self.space.dimensions) < 2:
            raise ValueError("At least two dimensions are required for 2D plotting.")
        if dim1 >= len(self.space.dimensions) or dim2 >= len(self.space.dimensions):
            raise ValueError("Chosen dimensions are higher than available dimensions")

        # Prepare data for plotting
        indices = [slice(None)] * len(self.space.dimensions)

        # Set fixed values for other dimensions
        if fixed_values is None:
            fixed_values = [0] * len(self.space.dimensions)

        for dim in range(len(indices)):
            if dim != dim1 and dim != dim2:
                indices[dim] = fixed_values[dim] if dim < len(fixed_values) else 0

        Z = self.space.values[tuple(indices)]
        X = self.space.meshgrid[dim1][tuple(indices)]
        Y = self.space.meshgrid[dim2][tuple(indices)]
        
        # ==========================================
        # NEU: BIAS AUF DIE OBERFLÄCHE (Z) ANWENDEN
        # ==========================================
        if simulate_bias and getattr(self.space, "bias", None) is not None:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    pt = [0] * len(self.space.dimensions)
                    pt[dim1] = X[i, j]
                    pt[dim2] = Y[i, j]
                    for d in range(len(pt)):
                        if d != dim1 and d != dim2:
                            pt[d] = fixed_values[d] if d < len(fixed_values) else 0
                    
                    # Hier ist der saubere Aufruf:
                    Z[i, j] = self.space.bias.apply(Z[i, j], pt)
        # ==========================================

        fig = go.Figure()

        # Plot surface
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', name='Criticality Surface'))

        # Plot multiple criticality threshold planes
        if show_critplane and self.space.criticality_thresholds != None:
            for i, th in enumerate(crit_tres):
                color = f"rgba({255 - i*50}, {100 + i*30}, 0, 0.4)"  # stepped colour per threshold
                crit_plane = np.full_like(Z, th)
                fig.add_trace(go.Surface(
                    x=X, y=Y, z=crit_plane,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=0.4,
                    name=f'Threshold {th:.2f}'
                ))

        if len(self.space.selected_points) == 0 and show_samples:
            warnings.warn(f"No selected points. Therefore no points will be printed")
            show_samples = False

        if show_samples:
            adjusted_points = []
            for pt in self.space.selected_points:
                new_pt = list(pt)
                for dim in range(len(new_pt)):
                    if dim != dim1 and dim != dim2:
                        new_pt[dim] = fixed_values[dim] if dim < len(fixed_values) else 0
                adjusted_points.append(new_pt)

            sample_points_x = np.array([p[dim1] for p in adjusted_points])
            sample_points_y = np.array([p[dim2] for p in adjusted_points])
            sample_values = self.space.get_values_for_points(adjusted_points, save_points=False)

            # ==========================================
            # NEU: BIAS AUF DIE SAMPLE PUNKTE ANWENDEN
            # OOP-Version
            # ==========================================
            if simulate_bias and getattr(self.space, "bias", None) is not None:
                for i, pt in enumerate(adjusted_points):
                    # Hier ist der saubere Aufruf:
                    sample_values[i] = self.space.bias.apply(sample_values[i], pt)
            # ==========================================

            fig.add_trace(go.Scatter3d(
                x=sample_points_x, y=sample_points_y, z=sample_values,
                mode='markers',
                marker=dict(color='red', size=5),
                name='Sample Points'
            ))

        # Init View
        elev = np.deg2rad(45)   # 45°
        azim = np.deg2rad(-45)  # -45°
        r = 2.0                 # Distance Camera

        eye = dict(
            x=r * np.cos(elev) * np.cos(azim),
            y=r * np.cos(elev) * np.sin(azim),
            z=r * np.sin(elev)
        )

        fig.update_layout(
            scene_camera=dict(eye=eye)
        )

        if show_samples:
            title = "3D Criticality Surface: 2D Variation and Sample Points"
        else:
            title = "3D Criticality Surface: 2D Variation"

        fig.update_layout(
    title=title, 
    autosize=True,
    scene=dict(
        xaxis_title=f'Dimension {dim1+1}', 
        yaxis_title=f'Dimension {dim2+1}', 
        zaxis_title='Criticality',
        zaxis=dict(range=[0, 1], autorange=False)  # <--- Hier ist der Käfig eingebaut
    ),
    legend_title="Legend"   
        )

#        fig.update_layout(title=title, autosize=True,
#                        scene=dict(xaxis_title=f'Dimension {dim1+1}', yaxis_title=f'Dimension {dim2+1}', zaxis_title='Criticality'),
#                        legend_title="Legend")
        
        if save_path is not None:
            if format == 'png':
                fig.write_image(save_path + '.png')
            elif format == 'html':
                fig.write_html(save_path + '.html')
            print(f"Plot saved as {format.upper()} to {save_path}.{format}")

        fig.show()
        return


    def plot_3d_three_varied(self, axis1=0, axis2=1, axis3=2):
        """
        3-D scatter plot with three free dimensions, coloured by criticality value.
        All other dimensions are fixed to their first grid index.

        Parameters
        ----------
        axis1, axis2, axis3 : int
            Indices of the three dimensions to plot on the X, Y, Z axes.
        """
        if len(self.space.dimensions) < 3:
            raise ValueError("At least three dimensions are required for 3D plotting.")
        if max(axis1, axis2, axis3) >= len(self.space.dimensions):
            raise ValueError("Axis index out of range of the available dimensions")

        # Create generalized indices for slicing the high-dimensional arrays
        indices = [0] * len(self.space.dimensions)
        indices[axis1], indices[axis2], indices[axis3] = slice(None), slice(None), slice(None)
        indices = tuple(indices)
        
        X = self.space.meshgrid[axis1][indices]
        Y = self.space.meshgrid[axis2][indices]
        Z = self.space.meshgrid[axis3][indices]
        criticality = self.space.values[indices]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X, Y, Z, c=criticality, cmap='viridis', marker='o')
        ax.set_xlabel(f'Dimension {axis1+1}')
        ax.set_ylabel(f'Dimension {axis2+1}')
        ax.set_zlabel(f'Dimension {axis3+1}')
        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        plt.title('3D Criticality Visualization in Selected Axes')
        plt.show()
        return
    
    def plot_top_down_2d(self, dim1=0, dim2=1, fixed_values=None, show_samples=True, save_path=None, format='png'):
        """
        Plot a top-down 2D visualization of the criticality surface (heatmap) 
        with optional overlay of selected sample points.
        Uses the same colormap ('Viridis') as the 3D surface plot.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if len(self.space.dimensions) < 2:
            raise ValueError("At least two dimensions are required for 2D plotting.")
        if dim1 >= len(self.space.dimensions) or dim2 >= len(self.space.dimensions):
            raise ValueError("Chosen dimensions are higher than available dimensions")

        # Prepare slicing indices
        indices = [slice(None)] * len(self.space.dimensions)
        if fixed_values is None:
            fixed_values = [0] * len(self.space.dimensions)
        for dim in range(len(indices)):
            if dim != dim1 and dim != dim2:
                indices[dim] = fixed_values[dim] if dim < len(fixed_values) else 0

        Z = self.space.values[tuple(indices)]
        X = self.space.meshgrid[dim1][tuple(indices)]
        Y = self.space.meshgrid[dim2][tuple(indices)]

        fig, ax = plt.subplots(figsize=(7, 6))

        # Plot heatmap (criticality values in 2D)
        im = ax.pcolormesh(X, Y, Z, cmap="viridis", shading="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Criticality")

        # Overlay sample points
        if show_samples and len(self.space.selected_points) > 0:
            adjusted_points = []
            for pt in self.space.selected_points:
                new_pt = list(pt)
                for dim in range(len(new_pt)):
                    if dim != dim1 and dim != dim2:
                        new_pt[dim] = fixed_values[dim] if dim < len(fixed_values) else 0
                adjusted_points.append(new_pt)

            sample_points_x = np.array([p[dim1] for p in adjusted_points])
            sample_points_y = np.array([p[dim2] for p in adjusted_points])

            ax.scatter(
                sample_points_x,
                sample_points_y,
                c="red",
                edgecolor="k",
                s=40,
                label="Sample Points",
            )
            ax.legend()

        ax.set_xlabel(f"Dimension {dim1+1}")
        ax.set_ylabel(f"Dimension {dim2+1}")
        # ax.set_title("Top-Down 2D Criticality Map")
        plt.tight_layout()

        if save_path is not None:
            if format == "png":
                plt.savefig(save_path + ".png", dpi=300)
            elif format == "pdf":
                plt.savefig(save_path + ".pdf")
            print(f"Plot saved as {format.upper()} to {save_path}.{format}")

        plt.show()
    
    def plot_boundary_mask(self, projection: Tuple[int, int] = (0, 1), aggregation: str = "max"):
        """
        Visualize the boundary mask.

        Args:
            projection (Tuple[int, int]): Dimensions to project onto (only used for d > 2).
            aggregation (str): Aggregation method ('max', 'sum') for d > 2 projections.
        """
        if not hasattr(self.space, "boundary_mask") or self.space.boundary_mask is None:
            raise ValueError("Boundary mask not yet created. Run space.create_boundary_mask() first.")

        mask = self.space.boundary_mask
        dim = len(self.space.dimensions)

        if dim == 2:
            plt.figure(figsize=(6, 6))
            plt.imshow(mask.T, origin="lower", cmap="Greys", interpolation="nearest")
            plt.title("Boundary Region Mask (2D)")
            plt.xlabel("Dimension 0")
            plt.ylabel("Dimension 1")
            plt.colorbar(label="Boundary (True=1)")
            plt.tight_layout()
            plt.show()

        elif dim > 2:
            axes = list(range(dim))
            axes.remove(projection[0])
            axes.remove(projection[1])

            # Aggregate over remaining axes
            if aggregation == "max":
                proj_mask = np.max(mask, axis=tuple(axes))
            elif aggregation == "sum":
                proj_mask = np.sum(mask, axis=tuple(axes))
            else:
                raise ValueError(f"Unknown aggregation '{aggregation}'.")

            plt.figure(figsize=(6, 6))
            plt.imshow(proj_mask.T, origin="lower", cmap="Greys", interpolation="nearest")
            plt.title(f"Boundary Mask Projection (Dims {projection[0]} vs {projection[1]})")
            plt.xlabel(f"Dimension {projection[0]}")
            plt.ylabel(f"Dimension {projection[1]}")
            plt.colorbar(label="Boundary (aggregated)")
            plt.tight_layout()
            plt.show()

        else:
            raise ValueError("Visualization is only supported for 2D or higher dimensional spaces.")
        
    def plot_3d_two_varied_discrete(self, dim1=0, dim2=1, fixed_values=None, show_samples=True, criticality_thresholds: list = None, save_path=None, format='html'):
        """
        Plot ground-truth surface of discrete class labels (from meshgrid) and overlay labeled selected points.
        The Z-axis is the discrete class label (e.g., 0, 1, 2). Points are colored by class.

        Args:
            dim1 (int): First dimension (X axis)
            dim2 (int): Second dimension (Y axis)
            fixed_values (list, optional): Fixed values for other dims.
            show_samples (bool): Show selected points as scatter.
            criticality_thresholds (list, optional): Thresholds for discretization.
            save_path (str, optional): File output path.
            format (str): 'html' or 'png'.
        """

        if criticality_thresholds is None:
            crit_tres = self.space.criticality_thresholds
        else:
            crit_tres = sorted(criticality_thresholds)

        if len(self.space.dimensions) < 2:
            raise ValueError("At least two dimensions are required for 2D plotting.")
        if dim1 >= len(self.space.dimensions) or dim2 >= len(self.space.dimensions):
            raise ValueError("Chosen dimensions are higher than available dimensions")

        # --- Prepare meshgrid for discrete ground truth ---
        indices = [slice(None)] * len(self.space.dimensions)
        if fixed_values is None:
            fixed_values = [0] * len(self.space.dimensions)
        for dim in range(len(indices)):
            if dim != dim1 and dim != dim2:
                indices[dim] = fixed_values[dim] if dim < len(fixed_values) else 0

        X = self.space.meshgrid[dim1][tuple(indices)]
        Y = self.space.meshgrid[dim2][tuple(indices)]
        flat_points = np.stack([g.ravel() for g in [self.space.meshgrid[d][tuple(indices)] for d in range(len(self.space.dimensions))]], axis=-1)
        # Compute ground-truth discrete class for all meshgrid points
        Z_discrete = self.space.get_discrete_values_for_points(flat_points, thresholds=crit_tres, save_points=False)
        Z_discrete = Z_discrete.reshape(X.shape)

        # Colormap for classes
        unique_labels = np.unique(Z_discrete)
        n_classes = len(unique_labels)
        palette = matplotlib.cm.get_cmap('tab10', n_classes)
        color_map = {lbl: matplotlib.colors.rgb2hex(palette(i)) for i, lbl in enumerate(unique_labels)}

        # --- Plot the surface of class labels ---
        # For visual clarity, map each class to a fixed color
        surface_colors = np.empty(Z_discrete.shape, dtype=object)
        for lbl in unique_labels:
            surface_colors[Z_discrete == lbl] = color_map[lbl]
        surface_colors = np.vectorize(lambda x: x)(surface_colors)

        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_discrete,
            surfacecolor=Z_discrete,  # color by class
            colorscale=[ [i/(n_classes-1 if n_classes>1 else 1), color_map[lbl]] for i, lbl in enumerate(unique_labels)],
            cmin=np.min(unique_labels), cmax=np.max(unique_labels),
            showscale=False,
            opacity=0.7,
            name='Class Surface'
        ))

        # --- Overlay the selected points, colored by their class ---
        if show_samples and len(self.space.selected_points) > 0:
            # Adapt selected points to current slice
            adjusted_points = []
            for pt in self.space.selected_points:
                new_pt = list(pt)
                for dim in range(len(new_pt)):
                    if dim != dim1 and dim != dim2:
                        new_pt[dim] = fixed_values[dim] if dim < len(fixed_values) else 0
                adjusted_points.append(new_pt)
            adjusted_points = np.array(adjusted_points)

            # Compute class labels for selected points
            sample_labels = self.space.get_discrete_values_for_points(
                adjusted_points, thresholds=crit_tres, save_points=False
            )

            # One trace per class for legend
            for lbl in unique_labels:
                idx = (sample_labels == lbl)
                if np.any(idx):
                    fig.add_trace(go.Scatter3d(
                        x=adjusted_points[idx, dim1],
                        y=adjusted_points[idx, dim2],
                        z=sample_labels[idx],
                        mode='markers',
                        marker=dict(size=7, color=color_map[lbl], line=dict(width=1, color='black')),
                        name=f"Class {lbl} samples"
                    ))

        fig.update_layout(
            title="3D Discrete Class Surface with Selected Points",
            scene=dict(
                xaxis_title=f'Dimension {dim1+1}',
                yaxis_title=f'Dimension {dim2+1}',
                zaxis_title="Class label",
                zaxis=dict(range=[0, 1], autorange=False)
            ),
            legend_title="Legend",
            autosize=True
        )

        if save_path is not None:
            if format == 'png':
                fig.write_image(save_path + '.png')
            elif format == 'html':
                fig.write_html(save_path + '.html')
            print(f"Plot saved as {format.upper()} to {save_path}.{format}")

        fig.show()
        return
    
    def plot_2d_decision_boundary(self, resolution=300, show_samples=True, point_size=6):
        """
        Plot 2D class boundaries and sampled points for a 2D space.
        - Blue dots: All points in meshgrid.
        - Red lines: Boundary contours (class changes).
        - Optionally overlays selected points (black).
        """

        # Assume 2D space (use first two dimensions)
        d1_range = self.space.dimensions[0]
        d2_range = self.space.dimensions[1]
        xx, yy = np.meshgrid(
            np.linspace(d1_range[0], d1_range[1], resolution),
            np.linspace(d2_range[0], d2_range[1], resolution)
        )
        flat_grid = np.c_[xx.ravel(), yy.ravel()]
        # Compute discrete class for each grid point
        zz = self.space.get_discrete_values_for_points(flat_grid, thresholds=self.space.criticality_thresholds, save_points=False)
        zz = zz.reshape(xx.shape)

        plt.figure(figsize=(7, 6))

        # Draw decision boundary (contour where class changes)
        levels = np.unique(zz)
        if len(levels) > 1:
            plt.contour(xx, yy, zz, levels=np.arange(min(levels)+0.5, max(levels)), colors='red', linewidths=2, alpha=0.9)
        else:
            print("Only one class present, cannot plot boundaries.")

        # Overlay selected points (if any)
        if show_samples and len(self.space.selected_points) > 0:
            sel = np.array(list(self.space.selected_points))
            plt.scatter(sel[:, 0], sel[:, 1], c='black', s=25, label="Selected", linewidths=0.5)

        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        # plt.title("2D Class Boundaries and Points")
        plt.legend()
        plt.tight_layout()
        plt.show()
