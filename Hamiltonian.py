#things to note
#.venv/Scripts/Activate.ps1 if venv is not activated
#the qubits q0q1q2q3 in qiskit are ordered |q3q2q1q0> so it will flip the values compared to top to bottom of the diagram

# H = - sum (X i tensor Xi+1 + Yi tensor Yi+1 + Jz Zi tensor Zi+1) 
# compares each value with the next ro make the hamiltonian 

#consider Jz > 1 and Jz < -1
#Jz > 1: initial state is |0>^tensor L or |1>^tensor L
#Jz < -1: initial state is |10>^tensor L 

#apply rotation operator to lattice sites (start w the middle) 
#evolve state over time and observe
#measure <Xi(t)>, <Yi(t)>, <Zi(t)> on each lattice site

import qiskit 
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

def hamiltonian1(L, Jz):
    """
    Docstring for hamiltonian
    
    :param X: X pauli op
    (X, 0, t)
    :param Y: Y pauli op
    :param Z: Z pauli op
    """
    ham_list = []

    for i in range (int(L)-1):
        ham_list.append(("XX", [i, i+1], -1)) # XX bc Xi and X i+1
        ham_list.append(("YY", [i, i+1], -1))
        ham_list.append(("ZZ", [i, i+1], -Jz))

    H =  SparsePauliOp.from_sparse_list(ham_list, num_qubits=L)

    return H


Jz = 1.1 #can vary

a = 4 #THIS IS L

def initialise(Jz, a):
    """ 
    decides L and the initial state to use depending on the value of Jz
    Returns: L, initial states
    """
    if  Jz > 1:
        L = a
        # all |0> ^tensor L
        psi_0 = Statevector.from_label("0"*L)
        # all |1> ^tensor L
        psi_1 = Statevector.from_label("1"*L)

        return L, [psi_0, psi_1]
    
    elif Jz < -1:
        L = a
        
        alternating = "".join(["10"[(i % 2)] for i in range(L)])
        psi_10 = Statevector.from_label(alternating)
        
        return L, [psi_10]
    
    else:
        raise ValueError("Jz must be >1 or < -1")

L, initial_states = initialise(Jz, a) #L HERE IS WHAT THE TENSOR NUMBER IS

H = hamiltonian1(L, Jz)

print(H)
