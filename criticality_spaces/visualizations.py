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
import math
from sklearn.base import clone
from mapie.regression import CrossConformalRegressor
from sklearn.model_selection import LeaveOneOut
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

    # =========================================================================
    # NEUE METHODEN FÜR DIE MASTERARBEIT (ACTIVE LEARNING AUSWERTUNGEN)
    # =========================================================================

    def plot_atslg_landscape(self, selector, resolution=50):
        """
        Erstellt ein interaktives 2x5 Gitter (10 Panels) im Browser via Plotly.
        Inklusive aller Samples (Initial & Aktiv).
        """
        from plotly.subplots import make_subplots # Lokal importieren, falls noch nicht oben
        print("\n[Visualisierung] Generiere interaktive HTML-Datei inkl. Samples (10 Panels)...")
        
        bounds = self.space.dimensions
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.c_[X.ravel(), Y.ravel()]
        
        # --- 1. Echte Daten berechnen ---
        y_true = np.array(self.space.get_values_for_points(grid_points, save_points=False))
        y_illus = np.array([self.space.bias.apply(v, p) for v, p in zip(y_true, grid_points)])
        Z_true = (y_true - y_illus).reshape(X.shape)
        
        # --- 2. Modelle abfragen ---
        try:
            # Versuch 1: Normaler Aufruf, falls das Modell trainiert wurde
            p1 = selector.gpc.predict_proba(grid_points)[:, 1]
        except:
            # Fallback: Das Modell wurde nicht trainiert (nur eine Klasse vorhanden).
            # Wir prüfen die letzten Trainingsdaten, um herauszufinden, welche Klasse das war.
            if len(selector.sample_history) > 0:
                last_X = selector.sample_history[-1]
                # Wir berechnen die Labels für die Trainingsdaten neu
                _, labels, _ = selector._get_diff_data(last_X)
                # Da es nur eine Klasse gibt, reicht es, das erste Label anzuschauen
                dominant_class = labels[0] 
                prob = 1.0 if dominant_class == 1 else 0.0
            else:
                prob = 0.0 # Last Fallback
                
            p1 = np.full(len(grid_points), prob)
            
        p2 = 1.0 - p1
        gpc_uncertainty_raw = p1 * p2
        gpc_uncertainty_plot = gpc_uncertainty_raw * 4 
            
        if hasattr(selector, "gpr1") and hasattr(selector.gpr1, "X_train_"):
            mean1, std1 = selector.gpr1.predict(grid_points, return_std=True)
        else:
            mean1, std1 = np.zeros(len(grid_points)), np.ones(len(grid_points))
            
        if hasattr(selector, "gpr2") and hasattr(selector.gpr2, "X_train_"):
            mean2, std2 = selector.gpr2.predict(grid_points, return_std=True)
        else:
            mean2, std2 = np.zeros(len(grid_points)), np.ones(len(grid_points))

        # --- 3. Mathematik ---
        pred_diff = (p1 * mean1) + (p2 * mean2) 
        E1 = (mean1 ** 2) + (std1 ** 2)
        E2 = (mean2 ** 2) + (std2 ** 2)
        EI = (p1 * E1) + (p2 * E2)
        
        U_E = np.max(EI) if np.max(EI) > 0 else 1.0
        U_C = np.max(gpc_uncertainty_raw) if np.max(gpc_uncertainty_raw) > 0 else 1.0
        
        w = 0.5 
        af_scores = w * (EI / U_E) + (1 - w) * (gpc_uncertainty_raw / U_C)

        # --- 4. Plotly Interaktive Figure erstellen ---
        fig = make_subplots(
            rows=2, cols=5,
            specs=[[{'is_3d': True}] * 5, [{'is_3d': True}] * 5],
            subplot_titles=(
                "1. Realität", "2. Finales Modell", "3. P1 (Fehler)", "4. P2 (kein Fehler)", "5. GPC Unsicherheit",
                "6. Acquisition Function", "7. GPR 1 Mean", "8. GPR 1 Var", "9. GPR 2 Mean", "10. GPR 2 Var"
            ),
            horizontal_spacing=0.02, vertical_spacing=0.05
        )

        def add_surface(z_data, colorscale, row, col):
            fig.add_trace(go.Surface(x=x, y=y, z=z_data.reshape(X.shape), colorscale=colorscale, showscale=False), row=row, col=col)

        add_surface(Z_true, 'reds', 1, 1)
        add_surface(pred_diff, 'blues', 1, 2)
        add_surface(p1, 'purples', 1, 3)
        add_surface(p2, 'ylgn', 1, 4)
        add_surface(gpc_uncertainty_plot, 'solar', 1, 5) 
        add_surface(af_scores, 'plasma', 2, 1)
        add_surface(mean1, 'oranges', 2, 2)
        add_surface(std1**2, 'inferno', 2, 3)
        add_surface(mean2, 'greens', 2, 4)
        add_surface(std2**2, 'viridis', 2, 5)

        # --- 5. SAMPLES HINZUFÜGEN ---
        if len(self.space.selected_points) > 0:
            selected_X = np.array(self.space.selected_points)
            pt_y_true = np.array(self.space.get_values_for_points(selected_X, save_points=False))
            pt_diff = pt_y_true - np.array([self.space.bias.apply(v, p) for v, p in zip(pt_y_true, selected_X)])
            
            n_init = getattr(selector, 'n_initial', 10)
            
            # Panel 2: Finales Modell
            fig.add_trace(go.Scatter3d(
                x=selected_X[:n_init, 0], y=selected_X[:n_init, 1], z=pt_diff[:n_init],
                mode='markers', marker=dict(size=6, color='blue', symbol='diamond'), name="Initiale Samples"
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter3d(
                x=selected_X[n_init:, 0], y=selected_X[n_init:, 1], z=pt_diff[n_init:],
                mode='markers', marker=dict(size=4, color='black', symbol='circle'), name="Aktive Samples"
            ), row=1, col=2)

            # Panel 6: Acquisition Function
            max_af = np.max(af_scores) if np.max(af_scores) > 0 else 1.0
            fig.add_trace(go.Scatter3d(
                x=selected_X[n_init:, 0], y=selected_X[n_init:, 1], z=np.full(len(selected_X)-n_init, max_af),
                mode='markers', marker=dict(size=4, color='black', symbol='circle'), name="Gezogene Punkte"
            ), row=2, col=1)

        # --- 6. Layout & Export ---
        fig.update_layout(
            title_text="Vollständige ATSLG-Architektur (Interaktiv)",
            title_x=0.5, height=900, width=2200, margin=dict(l=0, r=0, b=0, t=50), showlegend=False
        )

        camera = dict(eye=dict(x=1.5, y=1.5, z=0.5))
        for i in range(1, 11):
            fig.update_layout(**{f'scene{i}_camera': camera})

        html_file = "atslg_interaktiv.html"
        fig.write_html(html_file)
        print(f"[Visualisierung] Fertig! Öffne {html_file} im Browser...")
        
        import webbrowser
        import os
        webbrowser.open('file://' + os.path.realpath(html_file))

    def plot_nn_landscape(self, selector, resolution=50):
        """Zeigt, wie gut das Neuronale Netz die Realität gelernt hat."""
        print("\n[Visualisierung] Generiere NN-Fehler-Landschaft (2 Panels)...")
        
        bounds = self.space.dimensions
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.c_[X.ravel(), Y.ravel()]
        
        y_true = np.array(self.space.get_values_for_points(grid_points, save_points=False))
        y_illus = np.array([self.space.bias.apply(v, p) for v, p in zip(y_true, grid_points)])
        Z_true = (y_true - y_illus).reshape(X.shape)
        
        if hasattr(selector, "nn_model"):
            Z_pred = selector.nn_model.predict(grid_points).reshape(X.shape)
        else:
            Z_pred = np.zeros(X.shape)
        
        fig = plt.figure(figsize=(14, 6))
        fig.suptitle("Evaluation: Neuronales Netz (Random Sampling)", fontsize=14, fontweight='bold')
        
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_true, cmap='Reds', alpha=0.8, edgecolor='none')
        ax1.set_title("Echte Realität (Wahrer Fehler)")
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
        
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_pred, cmap='Purples', alpha=0.8, edgecolor='none')
        ax2.set_title("Gelerntes NN-Modell (MLPRegressor)")
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)

        if len(self.space.selected_points) > 0:
            selected_X = np.array(self.space.selected_points)
            pt_y_true = np.array(self.space.get_values_for_points(selected_X, save_points=False))
            pt_y_illus = np.array([self.space.bias.apply(v, p) for v, p in zip(pt_y_true, selected_X)])
            pt_diff = pt_y_true - pt_y_illus
            ax2.scatter(selected_X[:, 0], selected_X[:, 1], pt_diff, color='black', s=30, label="Gezogene Samples")
            ax2.legend()
            
        plt.tight_layout()
        plt.show()

    def plot_standard_gpr_landscape(self, selector, resolution=50):
        """Ablationsstudie: 2-Panel Plot für das Standard-GPR."""
        print("\n[Visualisierung] Generiere Fehler-Landschaft für Standard-GPR (2 Panels)...")
        
        bounds = self.space.dimensions
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.c_[X.ravel(), Y.ravel()]
        
        y_true = np.array(self.space.get_values_for_points(grid_points, save_points=False))
        y_illus = np.array([self.space.bias.apply(v, p) for v, p in zip(y_true, grid_points)])
        Z_true = (y_true - y_illus).reshape(X.shape)
        
        if hasattr(selector, "gpr") and hasattr(selector.gpr, "X_train_"):
            mean, _ = selector.gpr.predict(grid_points, return_std=True)
        else:
            mean = np.zeros(len(grid_points))
        Z_pred = mean.reshape(X.shape)
        
        fig = plt.figure(figsize=(14, 6))
        fig.suptitle("Ablationsstudie: Standard GPR-Verhalten ohne Klassifikator", fontsize=14, fontweight='bold')
        
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_true, cmap='Reds', alpha=0.8, edgecolor='none')
        ax1.set_title("Echte Realität (Wahrer Fehler)")
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
        
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_pred, cmap='Oranges', alpha=0.8, edgecolor='none')
        ax2.set_title("Gelerntes Modell (Einziger GPR)")
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)

        if len(self.space.selected_points) > 0:
            selected_X = np.array(self.space.selected_points)
            pt_y_true = np.array(self.space.get_values_for_points(selected_X, save_points=False))
            pt_y_illus = np.array([self.space.bias.apply(v, p) for v, p in zip(pt_y_true, selected_X)])
            ax2.scatter(selected_X[:, 0], selected_X[:, 1], (pt_y_true - pt_y_illus), color='black', s=20, label="Trainingsdaten")
            ax2.legend()
            
        plt.tight_layout()
        plt.show()

    def plot_dinn_landscape(self, selector, resolution=50):
        """Zeigt Realität, NN-Vorhersage und die MAPIE-Unsicherheitskarte in 3 Panels."""
        print("\n[Visualisierung] Generiere DINN-Fehler-Landschaft inkl. Unsicherheit (3 Panels)...")
        
        bounds = self.space.dimensions
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.c_[X.ravel(), Y.ravel()]
        
        # 1. Echte Realität
        y_true = np.array(self.space.get_values_for_points(grid_points, save_points=False))
        y_illus = np.array([self.space.bias.apply(v, p) for v, p in zip(y_true, grid_points)])
        Z_true = (y_true - y_illus).reshape(X.shape)
        
        # 2. Modell & Unsicherheit (MAPIE) neu berechnen auf dem finalen Stand
        if hasattr(selector, "nn_model") and len(selector.sample_history) > 0:
            from mapie.regression import CrossConformalRegressor
            from sklearn.model_selection import LeaveOneOut
            
            # Letzter Stand der Trainingsdaten
            current_X = selector.sample_history[-1]
            
            # Ground Truth der bekannten Punkte für MAPIE berechnen
            y_true_train = np.array(self.space.get_values_for_points(current_X, save_points=False))
            y_illus_train = np.array([self.space.bias.apply(v, p) for v, p in zip(y_true_train, current_X)])
            current_diffs = y_true_train - y_illus_train
            
            # MAPIE ein letztes Mal auf alle Daten anwenden, um die Map zu malen
            mapie = CrossConformalRegressor(
                estimator=selector.nn_model, 
                cv=LeaveOneOut(), 
                method='plus', 
                random_state=42
            )
            mapie.fit_conformalize(current_X, current_diffs)
            
            try:
                pred_mean, pis = mapie.predict_interval(grid_points)
            except TypeError:
                pred_mean, pis = mapie.predict_interval(grid_points, alpha=0.1)
                
            Z_pred = pred_mean.reshape(X.shape)
            
            # Unsicherheit ausrechnen
            lower_bound = pis[:, 0, 0]
            upper_bound = pis[:, 1, 0]
            Z_uncert = (upper_bound - lower_bound).reshape(X.shape)
        else:
            Z_pred = np.zeros(X.shape)
            Z_uncert = np.zeros(X.shape)
        
        # --- 3 Panels zeichnen ---
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(18, 6))
        fig.suptitle("Evaluation: Neuronales Netz mit MAPIE (Jackknife+)", fontsize=14, fontweight='bold')
        
        # Panel 1: Realität
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_true, cmap='Reds', alpha=0.8, edgecolor='none')
        ax1.set_title("Echte Realität (Bias)")
        fig.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.1)
        
        # Panel 2: Vorhersage
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_pred, cmap='Purples', alpha=0.8, edgecolor='none')
        ax2.set_title("Gelerntes NN (Vorhersage)")
        fig.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.1)

        # Panel 3: Unsicherheit
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(X, Y, Z_uncert, cmap='YlGnBu', alpha=0.8, edgecolor='none')
        ax3.set_title("Unsicherheitskarte (Intervallbreite)")
        fig.colorbar(surf3, ax=ax3, shrink=0.5, pad=0.1)

        # Gemessene Samples als schwarze Punkte einzeichnen
        # Gemessene Samples differenziert einzeichnen (Initial vs. Aktiv)
        if len(self.space.selected_points) > 0:
            selected_X = np.array(self.space.selected_points)
            pt_y_true = np.array(self.space.get_values_for_points(selected_X, save_points=False))
            pt_y_illus = np.array([self.space.bias.apply(v, p) for v, p in zip(pt_y_true, selected_X)])
            pt_diff = pt_y_true - pt_y_illus
            
            # Anzahl der initialen Punkte abrufen (Fallback auf 10)
            n_init = getattr(selector, 'n_initial', 10)
            
            # --- Panel 2: Vorhersage ---
            # 1. Initiale Samples (Blaue Diamanten)
            ax2.scatter(
                selected_X[:n_init, 0], 
                selected_X[:n_init, 1], 
                pt_diff[:n_init], 
                color='blue', 
                marker='D', 
                s=40, 
                label="Initiale Samples"
            )
            
            # 2. Aktive Samples (Schwarze Kreise)
            if len(selected_X) > n_init:
                ax2.scatter(
                    selected_X[n_init:, 0], 
                    selected_X[n_init:, 1], 
                    pt_diff[n_init:], 
                    color='black', 
                    marker='o', 
                    s=30, 
                    label="Aktive Samples"
                )
            ax2.legend()
            
            # --- Panel 3: Unsicherheit ---
            # Hier zeichnen wir die Punkte am Boden (Höhe 0) ein, um die Verteilung zu sehen
            ax3.scatter(
                selected_X[:n_init, 0], 
                selected_X[:n_init, 1], 
                np.zeros(n_init), 
                color='blue', 
                marker='D', 
                s=40
            )
            if len(selected_X) > n_init:
                ax3.scatter(
                    selected_X[n_init:, 0], 
                    selected_X[n_init:, 1], 
                    np.zeros(len(selected_X) - n_init), 
                    color='black', 
                    marker='o', 
                    s=30
                )
            
        plt.tight_layout()
        plt.show()

    def plot_comparison_landscape(self, selector, resolution=50):
        """Erstellt einen 3-Panel-Vergleich: Illusion, Realität und NN-Korrektur."""
        print("\n[Visualisierung] Generiere Vergleichs-Landschaft (3 Panels)...")
        
        bounds = self.space.dimensions
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution) # Korrigiert auf bounds[1]
        X, Y = np.meshgrid(x, y)
        grid_points = np.c_[X.ravel(), Y.ravel()]
        
        # 1. Realität (Ground Truth)
        y_true = np.array(self.space.get_values_for_points(grid_points, save_points=False))
        Z_real = y_true.reshape(X.shape)
        
        # 2. Illusion (Simulation)
        # Exakt so berechnen, wie es dein NN-Selektor in _get_diff_data tut!
        y_illus = np.array([self.space.bias.apply(v, p) for v, p in zip(y_true, grid_points)])
        Z_illusion = y_illus.reshape(X.shape)
        
        # 3. Korrektur (Illusion + NN Vorhersage)
        Z_nn_pred = selector.nn_model.predict(grid_points).reshape(X.shape)
        Z_corrected = Z_illusion + Z_nn_pred
        
        # =====================================================================
        # DER TRICK: GEMEINSAME SKALIERUNG FÜR ALLE PLOTS
        # =====================================================================
        # Wir suchen den absolut niedrigsten und höchsten Wert aller drei Welten
        z_min = min(Z_real.min(), Z_illusion.min(), Z_corrected.min())
        z_max = max(Z_real.max(), Z_illusion.max(), Z_corrected.max())
        
        # Plotting
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(20, 6))
        fig.suptitle("Performance-Vergleich: Simulation vs. Realität vs. NN-Korrektur", fontsize=16, fontweight='bold')
        
        # Panel 1: Illusion
        ax1 = fig.add_subplot(131, projection='3d')
        # vmin und vmax zwingen die Farbe in unser Skalierungs-Korsett
        surf1 = ax1.plot_surface(X, Y, Z_illusion, cmap='Blues', alpha=0.8, edgecolor='none', vmin=z_min, vmax=z_max)
        ax1.set_title("1. Illusion (Unkorrigiert)")
        ax1.set_zlim(z_min, z_max) # Zwingt die Z-Achse, gleich hoch zu sein
        fig.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.1)
        
        # Panel 2: Realität
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_real, cmap='Reds', alpha=0.8, edgecolor='none', vmin=z_min, vmax=z_max)
        ax2.set_title("2. Realität (Ground Truth)")
        ax2.set_zlim(z_min, z_max)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.1)
        
        # Panel 3: Korrektur
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(X, Y, Z_corrected, cmap='Greens', alpha=0.8, edgecolor='none', vmin=z_min, vmax=z_max)
        ax3.set_title("3. NN-Korrektur (Illusion + NN)")
        ax3.set_zlim(z_min, z_max)
        fig.colorbar(surf3, ax=ax3, shrink=0.5, pad=0.1)
        
        plt.tight_layout()
        plt.show()

    def plot_dinn_uncertainty_evolution(self, selector, resolution=40):
        """
        Plottet die Entwicklung der MAPIE-Unsicherheitskarte über alle Iterationen.
        Erstellt dynamisch ein Grid von 3D-Subplots.
        """
        import math
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.base import clone
        from mapie.regression import CrossConformalRegressor
        from sklearn.model_selection import LeaveOneOut

        print("\n[Visualisierung] Generiere Evolution der Unsicherheit...")
        print("Hinweis: Dies kann je nach Anzahl der Iterationen einen Moment dauern, da MAPIE nachgebaut wird.")

        history = selector.sample_history
        n_iters = len(history)

        if n_iters == 0:
            print("Keine Historie gefunden!")
            return

        # --- 1. Layout berechnen ---
        # Maximal 5 Spalten, Reihen werden automatisch aufgefüllt
        cols = min(5, n_iters)
        rows = math.ceil(n_iters / cols)

        fig = plt.figure(figsize=(4 * cols, 4 * rows))
        fig.suptitle("Evolution der MAPIE-Unsicherheit (Active Learning)", fontsize=16, fontweight='bold')

        # --- 2. Grid vorbereiten ---
        # Resolution etwas runtergesetzt (40), damit es bei vielen Iterationen schneller rechnet
        bounds = self.space.dimensions
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.c_[X.ravel(), Y.ravel()]

        # Wir klonen das initiale Modell, um saubere Trainingsdurchläufe zu garantieren
        base_model = clone(selector.nn_model)

        # --- 3. Schleife über alle Iterationen ---
        for i, current_X in enumerate(history):
            print(f" -> Berechne Landschaft für Iteration {i+1}/{n_iters} (Samples: {len(current_X)})...")
            
            # Ground Truth und Diff für den aktuellen Stand berechnen
            y_true_train = np.array(self.space.get_values_for_points(current_X, save_points=False))
            y_illus_train = np.array([self.space.bias.apply(v, p) for v, p in zip(y_true_train, current_X)])
            current_diffs = y_true_train - y_illus_train

            # Modell und MAPIE exakt wie in der Schleife trainieren
            base_model.fit(current_X, current_diffs)
            mapie = CrossConformalRegressor(
                estimator=base_model, 
                cv=LeaveOneOut(), 
                method='plus', 
                random_state=42
            )
            mapie.fit_conformalize(current_X, current_diffs)

            # Unsicherheit für das gesamte Grid vorhersagen
            try:
                _, pis = mapie.predict_interval(grid_points)
            except TypeError:
                _, pis = mapie.predict_interval(grid_points, alpha=0.1)

            lower_bound = pis[:, 0, 0]
            upper_bound = pis[:, 1, 0]
            Z_uncert = (upper_bound - lower_bound).reshape(X.shape)

            # --- 4. Subplot zeichnen ---
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            surf = ax.plot_surface(X, Y, Z_uncert, cmap='YlGnBu', alpha=0.9, edgecolor='none')
            
            ax.set_title(f"Iter {i+1} (N={len(current_X)})", fontsize=10)
            # Achsen-Labels ausblenden, damit das Grid nicht zu unordentlich wird
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # Die Samples, die bis zu dieser Iteration bekannt waren, auf den Boden (0) malen
            ax.scatter(current_X[:, 0], current_X[:, 1], np.zeros(len(current_X)), color='black', marker='o', s=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9) # Platz für den Suptitle lassen
        plt.show()