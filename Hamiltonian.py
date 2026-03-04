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
from qiskit.quantum_info import SparsePauliOp, Statevector
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import functools as ft

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
    Returns: L, list of initial states as Statevectors
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








#### Classical Comparison function, does the calculation with matrix multiplication

def ClassicalComparison(L,Jz,qubit_measured,direction_measured,steps=100,t_tot=1,allzeros=True):
    '''
    performs an expectation value measurement for a spin proj. in a certain direction on a certain qubit. Does this classically with matrix multiplication.
    (still need to make the initial state rotation classical) (imports needed: scipy as sp, numpy as np, functools as ft)
    
    returns two np.arrays: ([time_values],[measurements])

    :param L: (int) number of qubits
    :param Jz: (float) J_z value 
    
    :param qubit_measured: (int) which qubit (1 to L) is measured
    :param direction_measured: (str) "X" "Y" or "Z", which projection operator is applied to the measured qubit
    :param steps: (int) number of time steps (auto set to 100)
    :param t_tot: (float) total time (in seconds? not seconds cause there are no hbars or cs) (auto set to 1)
    :param allzeros: (bool) if Jz>1, True chooses the all |0> state, False chooses the all |1> state
    '''
    #defining the Pauli matrices
    pauli_x = ((0,1),(1,0))
    pauli_y = ((0,-1j),(1j,0))
    pauli_z = ((1,0),(0,-1))

    #Hamiltonian as an np.array matrix
    matrix_h = np.array(hamiltonian1(L,Jz))

    #getting initial state, according to Jz value
    if Jz > 1:
        all0, all1 = initialise(Jz,L)[1]
        if allzeros==True:
            initial = all0
        else:
            initial = all1
    elif Jz < -1:
        initial = initialise(Jz,L)[1][0]
    else:
        print('Please pick J_z > 1 or J_z < -1')
        return

    #rotating middle qubit's spin to x-y plane
    ### This is the quantum bit that i should figure out classically ###
    qc = QuantumCircuit(int(L))
    qc.initialize(initial) #initialize state
    #apply rotation
    middle = L //2 #picks L/2 int value
    qc.ry(np.pi/2, middle) #rotation here is into x-y plane
    initial = Statevector(qc)  #initial state, now rotated
    ###

    #getting the vector for our state as an np.array
    psi0 = np.array(initial)
    
    #choosing correct pauli matrix
    if direction_measured == "X" or direction_measured == "x":
        pauli = pauli_x
    elif direction_measured == "Y" or direction_measured == "y":
        pauli = pauli_y
    elif direction_measured == "Z" or direction_measured == "z":
        pauli = pauli_z
    else:
        print("Please choose strings 'X', 'Y', or 'Z' for direction_measured")
        return
    
    #creating our observable operator matrix, for our expectation value
    product = [np.eye(2) for i in range(L)]
    product[qubit_measured-1] = pauli         #list of matrices to kronecker product to build the obervable matrix
    observable_m = ft.reduce(np.kron,product) #does multiple kronecker products in a row (with matrices in list)

    # Initialize an empty list to store the observables we compute
    measurements = []

    #timestep
    dt = t_tot/steps

    # Run a loop over timesteps
    for t in range(steps):
        # Compute the time evolution operator that evolves the system to time t * 0.01
        Ut = sp.linalg.expm(- 1j * matrix_h * dt * t)

        # Apply the time evolution operator to the initial state in order to construct |psi(t)>
        psi_t = Ut @ psi0

        # Compute the expectation value of the Z projection of one of the qubits

        expval = psi_t.T @ observable_m @ psi_t

        # Save the above expectation value to the list we created at the start
        # (It's safe to only keep the real part because all observables are Hermitian, meaning they have real expectation values)
        measurements += [expval.real]
    
    t_values = np.array([dt*t for t in range(steps)])

    return (t_values,np.array(measurements))
