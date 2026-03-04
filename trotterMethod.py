import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit.library import PauliEvolutionGate

# defining a single trotter_step
def trotter_step_1st_order(circuit: QuantumCircuit, H: SparsePauliOp, dt: float):
    """
    Apply a single 1st order trotter step onto a given QuantumCircuit

    Parameters:
    - circuit (QuantumCircuit): The Quantum Circuit on which the trotter step will be applied
    - H (SparsePauliOp): The hamiltonian represented as a sparse Pauli operator
    - dt (float): The step length in time each trotter step will take

    Returns:
    -circuit (QuantumCircuit): The quatum circuit after applying one trotter step
    """
    for pauli_str, coeff in H.to_list():
        # convert string to Pauli object
        pauli_op = Pauli(pauli_str)
        evolution_gate = PauliEvolutionGate(pauli_op, time=dt * coeff.real)
        circuit.append(evolution_gate, range(circuit.num_qubits))
    return circuit

# defining a function that applies the single trotter step n times
def first_order_trotter(H: SparsePauliOp, t: float, n_steps: int, num_qubits: int = None) -> QuantumCircuit:
    """
    Apply trotter_step_1st_order onto a quantum circuit n qubits for n_steps to evolve for time t.

    Parameters:
    - H (SparsePauliOp): The hamiltonian represented as a sparse Pauli operator
    - t (float): length of time to evolve qubits for
    - n_steps (int): Number of steps applied to circuit
    - num_qubits (int): Number of qubits in the system

    Return:
    - qc (QuantumCircuit): circuit implemeting time evolution
    """
    if num_qubits is None:
        num_qubits = H.num_qubits
    
    # create circuit with width = num_qubits
    qc = QuantumCircuit(num_qubits)
    # find the length of each individual trotter step
    dt = t/n_steps
    # loop
    for _ in range(n_steps):
        trotter_step_1st_order(qc, H, dt)
    return qc