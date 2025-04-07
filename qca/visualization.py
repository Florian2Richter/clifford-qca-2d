import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def pauli_to_numeric(pauli_str, N):
    """
    Konvertiert einen Pauli-String (Länge N²) in ein 2D-Array (N×N) numerischer Codes:
      I -> 0, X -> 1, Z -> 2, Y -> 3.
    """
    mapping = {"I": 0, "X": 1, "Z": 2, "Y": 3}
    numeric = [mapping[p] for p in pauli_str]
    return np.array(numeric).reshape((N, N))

def simulate_and_plot_fractal_2D(N, T_steps, initial_operator, evolution_matrix):
    """
    Simuliert die fractale QCA-Evolution auf einem 2D-Gitter und erstellt für jeden
    Zeitschritt eine farbige Darstellung. 
    Es wird ein Plot mit mehreren Subplots erzeugt, die jeweils den Zustand zu einem
    bestimmten Zeitschritt zeigen.
    """
    from qca.core import simulate_fractal_QCA_2D
    evolution = simulate_fractal_QCA_2D(N, T_steps, initial_operator, evolution_matrix)
    
    # Definiere eine diskrete Colormap: I: weiß, X: rot, Z: blau, Y: grün
    cmap = ListedColormap(["white", "red", "blue", "green"])
    
    # Erstelle ein Subplot-Gitter (z. B. 4 Spalten)
    ncols = 4
    nrows = (T_steps + 1 + ncols - 1) // ncols  # aufrunden
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()
    
    for i, op_str in enumerate(evolution):
        data = pauli_to_numeric(op_str, N)
        ax = axes[i]
        ax.imshow(data, cmap=cmap, interpolation="none", aspect="equal")
        ax.set_title(f"t = {i}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Blende leere Subplots aus
    for j in range(len(evolution), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    return evolution
