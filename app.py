import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from qca.core import build_T_2D, vector_to_pauli_string, mod2_matmul
from qca.visualization import pauli_to_numeric

def update(frame, img, state, T_matrix, N):
    # Calculate the next state and update the state (in a list to make it mutable)
    new_state = mod2_matmul(T_matrix, state[0]) % 2
    state[0] = new_state.copy()
    pauli_str = vector_to_pauli_string(new_state)
    data = pauli_to_numeric(pauli_str, N)
    img.set_data(data)
    return [img]

if __name__ == '__main__':
    N = 50  # Grid size (50x50)
    total_cells = 2 * (N * N)  # Length of state vector (X and Z components)
    
    # Initial state: center cell (X) active, rest 0
    initial_operator = np.zeros(total_cells, dtype=int)
    center = (N * N) // 2
    initial_operator[center] = 1  # Set X component to 1 at the center cell
    
    # Create the 2D evolution matrix
    T_matrix = build_T_2D(N)
    
    # Store the state in a list to keep it mutable in the update function
    state = [initial_operator]
    
    # Initial visualization: Create an image with Pauli states as numeric values
    fig, ax = plt.subplots()
    pauli_str = vector_to_pauli_string(state[0])
    data = pauli_to_numeric(pauli_str, N)
    # Define a discrete colormap (I: white, X: red, Z: blue, Y: green)
    cmap = plt.matplotlib.colors.ListedColormap(["white", "red", "blue", "green"])
    im = ax.imshow(data, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="none", aspect="equal")
    ax.set_title("2D Clifford QCA Evolution (Simulation)")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create the animation (one frame every 500ms)
    ani = animation.FuncAnimation(fig, update, fargs=(im, state, T_matrix, N),
                                  interval=500, blit=True)
    plt.show()
