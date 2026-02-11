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

def hamiltonian1(L, J):
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
        ham_list.append(("ZZ", [i, i+1], -J))

    H =  SparsePauliOp.from_sparse_list(ham_list, num_qubits=L)

    return H


J = 1.1 #can vary

a = 4

if J > 1:
    L = a
elif J < -1:
    L = a/2

H = hamiltonian1(L, J)

print(H)

evolution_gate = PauliEvolutionGate(H, time=1.0)
qc = QuantumCircuit(int(L))
qc.append(evolution_gate, range(int(L)))

print(qc.decompose().draw()) 
