import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def pauli_to_numeric(pauli_str, N):
    """
    Convert a Pauli string (length N²) into a 2D array (N×N) of numeric codes:
      I -> 0, X -> 1, Z -> 2, Y -> 3.
    """
    mapping = {"I": 0, "X": 1, "Z": 2, "Y": 3}
    numeric = [mapping[p] for p in pauli_str]
    return np.array(numeric).reshape((N, N))

def simulate_and_plot_fractal_2D(N, T_steps, initial_operator, evolution_matrix):
    """
    Simulate the fractal QCA evolution on a 2D grid and create a colored visualization
    for each time step.
    Creates a plot with multiple subplots, each showing the state at a specific time step.
    """
    from qca.core import simulate_fractal_QCA_2D
    evolution = simulate_fractal_QCA_2D(N, T_steps, initial_operator, evolution_matrix)
    
    # Define a discrete colormap: I: white, X: red, Z: blue, Y: green
    cmap = ListedColormap(["white", "red", "blue", "green"])
    
    # Create a subplot grid (e.g., 4 columns)
    ncols = 4
    nrows = (T_steps + 1 + ncols - 1) // ncols  # round up
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()
    
    for i, op_str in enumerate(evolution):
        data = pauli_to_numeric(op_str, N)
        ax = axes[i]
        ax.imshow(data, cmap=cmap, interpolation="none", aspect="equal")
        ax.set_title(f"t = {i}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide empty subplots
    for j in range(len(evolution), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    return evolution
