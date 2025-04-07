import numpy as np

def mod2_matmul(A, v):
    """Multipliziere Matrix A und Vektor v über F2 (Mod2-Arithmetik)."""
    return (A.dot(v)) % 2

def left_shift(N):
    """Erstellt die Linksverschiebungsmatrix L (N×N) mit periodischen Randbedingungen."""
    L = np.zeros((N, N), dtype=int)
    for i in range(N):
        L[i, (i - 1) % N] = 1
    return L

def right_shift(N):
    """Erstellt die Rechtsverschiebungsmatrix R (N×N) mit periodischen Randbedingungen."""
    R = np.zeros((N, N), dtype=int)
    for i in range(N):
        R[i, (i + 1) % N] = 1
    return R

def build_T_2D(N):
    """
    Baut die Evolutionsmatrix T für ein 2D-Clifford QCA auf einem N×N Gitter.
    Jeder Zelle werden zwei Bits zugeordnet (X- und Z-Teil), sodass der Zustandsvektor
    insgesamt Länge 2*N² hat.
    
    Die Update-Regeln werden wie folgt generalisiert:
      new_x = (H_left + H_right + V_up + V_down + I) * old_x + old_z   (mod 2)
      new_z = old_x.
      
    Daraus folgt in Blockform:
      T = [ A   I ]
          [ I   0 ]
    wobei A = (H_left + H_right + V_up + V_down + I) mod 2 ist.
    """
    N2 = N * N
    # Identitätsmatrix für das flache Gitter (Größe: N²×N²)
    I_grid = np.eye(N2, dtype=int)
    
    # Erzeuge die 1D-Shift-Matrizen (N×N)
    I_N = np.eye(N, dtype=int)
    L_1d = left_shift(N)
    R_1d = right_shift(N)
    
    # Horizontale Shifts: 
    # H_left = I_N ⊗ L_1d, H_right = I_N ⊗ R_1d.
    H_left = np.kron(I_N, L_1d)
    H_right = np.kron(I_N, R_1d)
    
    # Vertikale Shifts:
    # V_up = L_1d ⊗ I_N, V_down = R_1d ⊗ I_N.
    V_up = np.kron(L_1d, I_N)
    V_down = np.kron(R_1d, I_N)
    
    # Definiere A als Summe der Shifts plus Identität (mod 2)
    A = (H_left + H_right + V_up + V_down + I_grid) % 2
    
    # Erstelle die Blockmatrix T
    top = np.hstack((A, I_grid))
    bottom = np.hstack((I_grid, np.zeros((N2, N2), dtype=int)))
    T = np.vstack((top, bottom)) % 2
    return T

def vector_to_pauli_string(v):
    """
    Konvertiert einen 2M-Vektor über F2 (erste M Einträge = X-Teil, 
    nächste M Einträge = Z-Teil) in einen String von Pauli-Operatoren.
    
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
    Simuliert die QCA-Evolution auf einem 2D-Gitter der Größe N×N für T_steps Zeitschritte
    unter Verwendung der angegebenen Evolutionsmatrix. 
    Als Rückgabe erhält man eine Liste von Pauli-Strings (je einer pro Zeitschritt),
    wobei jeder String Länge N² hat (entsprechend den Zellen im Gitter).
    """
    state = initial_operator.copy() % 2
    evolution = [vector_to_pauli_string(state)]
    for t in range(1, T_steps + 1):
        state = mod2_matmul(evolution_matrix, state) % 2
        evolution.append(vector_to_pauli_string(state))
    return evolution
