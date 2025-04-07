import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from qca.core import build_T_2D, vector_to_pauli_string, mod2_matmul
from qca.visualization import pauli_to_numeric

def update(frame, img, state, T_matrix, N):
    # Berechne den nächsten Zustand und aktualisiere den Zustand (in einer Liste, damit er veränderbar ist)
    new_state = mod2_matmul(T_matrix, state[0]) % 2
    state[0] = new_state.copy()
    pauli_str = vector_to_pauli_string(new_state)
    data = pauli_to_numeric(pauli_str, N)
    img.set_data(data)
    return [img]

if __name__ == '__main__':
    N = 50  # Gittergröße (8x8)
    total_cells = 2 * (N * N)  # Länge des Zustandsvektors (X- und Z-Anteile)
    
    # Initialer Zustand: zentrale Zelle (X) aktiv, Rest 0
    initial_operator = np.zeros(total_cells, dtype=int)
    center = (N * N) // 2
    initial_operator[center] = 1  # Setzt an der zentralen Zelle den X-Anteil auf 1
    
    # Erstelle die 2D-Evolutionsmatrix
    T_matrix = build_T_2D(N)
    
    # Speichere den Zustand in einer Liste, um ihn in der update-Funktion veränderbar zu halten
    state = [initial_operator]
    
    # Initiale Visualisierung: Wir erzeugen ein Bild mit den Pauli-Zuständen als Zahlenwerte
    fig, ax = plt.subplots()
    pauli_str = vector_to_pauli_string(state[0])
    data = pauli_to_numeric(pauli_str, N)
    # Definiere eine diskrete Colormap (I: weiß, X: rot, Z: blau, Y: grün)
    cmap = plt.matplotlib.colors.ListedColormap(["white", "red", "blue", "green"])
    im = ax.imshow(data, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="none", aspect="equal")
    ax.set_title("2D Clifford QCA Evolution (Simulation)")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Erstelle die Animation (alle 500ms ein Frame)
    ani = animation.FuncAnimation(fig, update, fargs=(im, state, T_matrix, N),
                                  interval=500, blit=True)
    plt.show()
