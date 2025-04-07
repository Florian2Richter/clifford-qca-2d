import numpy as np

def mod2_matmul(A, v):
    """Multiply matrix A and vector v over F2 (Mod2 arithmetic)."""
    return (A.dot(v)) % 2

def left_shift(N):
    """Create the left shift matrix L (N×N) with periodic boundary conditions."""
    L = np.zeros((N, N), dtype=int)
    for i in range(N):
        L[i, (i - 1) % N] = 1
    return L

def right_shift(N):
    """Create the right shift matrix R (N×N) with periodic boundary conditions."""
    R = np.zeros((N, N), dtype=int)
    for i in range(N):
        R[i, (i + 1) % N] = 1
    return R

def build_T_2D(N):
    """
    Build the evolution matrix T for a 2D-Clifford QCA on an N×N grid.
    Each cell is assigned two bits (X and Z part), so the state vector
    has a total length of 2*N².
    
    The update rules are generalized as follows:
      new_x = (H_left + H_right + V_up + V_down + I) * old_x + old_z   (mod 2)
      new_z = old_x.
      
    This results in block form:
      T = [ A   I ]
          [ I   0 ]
    where A = (H_left + H_right + V_up + V_down + I) mod 2.
    """
    N2 = N * N
    # Identity matrix for the flat grid (size: N²×N²)
    I_grid = np.eye(N2, dtype=int)
    
    # Generate the 1D shift matrices (N×N)
    I_N = np.eye(N, dtype=int)
    L_1d = left_shift(N)
    R_1d = right_shift(N)
    
    # Horizontal shifts: 
    # H_left = I_N ⊗ L_1d, H_right = I_N ⊗ R_1d.
    H_left = np.kron(I_N, L_1d)
    H_right = np.kron(I_N, R_1d)
    
    # Vertical shifts:
    # V_up = L_1d ⊗ I_N, V_down = R_1d ⊗ I_N.
    V_up = np.kron(L_1d, I_N)
    V_down = np.kron(R_1d, I_N)
    
    # Define A as sum of shifts plus identity (mod 2)
    A = (H_left + H_right + V_up + V_down + I_grid) % 2
    
    # Create the block matrix T
    top = np.hstack((A, I_grid))
    bottom = np.hstack((I_grid, np.zeros((N2, N2), dtype=int)))
    T = np.vstack((top, bottom)) % 2
    return T

def vector_to_pauli_string(v):
    """
    Convert a 2M-vector over F2 (first M entries = X part, 
    next M entries = Z part) into a string of Pauli operators.
    
    (0,0) -> I, (1,0) -> X, (0,1) -> Z, (1,1) -> Y.
    """
    M = len(v) // 2
    paulis = []
    for i in range(M):
        x = v[i] % 2
        z = v[M + i] % 2
        if x == 0 and z == 0:
            paulis.append("I")
        elif x == 1 and z == 0:
            paulis.append("X")
        elif x == 0 and z == 1:
            paulis.append("Z")
        elif x == 1 and z == 1:
            paulis.append("Y")
    return "".join(paulis)

def simulate_fractal_QCA_2D(N, T_steps, initial_operator, evolution_matrix):
    """
    Simulate the QCA evolution on a 2D grid of size N×N for T_steps time steps
    using the specified evolution matrix.
    Returns a list of Pauli strings (one per time step),
    where each string has length N² (corresponding to the cells in the grid).
    """
    state = initial_operator.copy() % 2
    evolution = [vector_to_pauli_string(state)]
    for t in range(1, T_steps + 1):
        state = mod2_matmul(evolution_matrix, state) % 2
        evolution.append(vector_to_pauli_string(state))
    return evolution
