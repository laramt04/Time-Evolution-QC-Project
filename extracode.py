#Our trotter
num_qubits = H.num_qubits

# start from |0000>
# rotate the middle qubit into the x-y plane
initial_state = initialise(Jz, L, allzeros=True, rotate_middle=True).data

# time parameters
t_max = 5.0
n_time_points = 10
times = np.linspace(0, t_max, n_time_points)

# trotter parameters
n_steps_per_unit_time = 20

# build observables: X, Y, Z operators on each qubit

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

I = np.eye(2, dtype = complex)

def build_local_observable(single_qubit_op: np.ndarray, qubit_idx: int, num_qubits: int) -> np.ndarray:
    """
    Build N-qubit observable with single_qubit_op on qubit_idx and identity elsewhere.
    
    Parameters
    -----------
    - single_qubi_op: np.ndarray -> 2x2 matrix representing the operator to b applied to qubit (Pauli X, Y, Z)
    - qubit_idx: int -> Index of qubit being used
    - num_qubits: int -> total num of qubits in the system

    Returns
    -----------
    - A matrix (2^N x 2^N) representing the full N-qubit observable

    """
    # create list of ID operators, then replace one with desired operator
    ops = [I] * num_qubits
    ops[qubit_idx] = single_qubit_op

    # build the full matrix
    result = ops[0]
    # apply kronecker products
    for op in ops[1:]:
        result = np.kron(result, op)
    # return final operator
    return result

X_observables = [build_local_observable(X, i, num_qubits) for i in range(num_qubits)]
Y_observables = [build_local_observable(Y, i, num_qubits) for i in range(num_qubits)]
Z_observables = [build_local_observable(Z, i, num_qubits) for i in range(num_qubits)]

observable_dict = {
    'X': X_observables,
    'Y': Y_observables,
    'Z': Z_observables
}

# calculate the exact and trotter expectations
exact_expectations = {
    f'{obs}_{i}': [] for obs in ['X', 'Y', 'Z'] for i in range(num_qubits)
}

trotter_expectations = {
    f'{obs}_{i}': [] for obs in ['X', 'Y', 'Z'] for i in range(num_qubits)
}

# calculate time evolution and expectation values
print("Calculating time evolution")
print("-" * 50)
for idx, t in enumerate(times):
    # print the progress of the evolution
    if idx % 20 == 0:
        print(f"Progress: {idx}/{n_time_points} (t = {t:.2f})")
    
    # exact evolution
    exact_state = exact_time_evolution(H, initial_state, t)

    # trotter evo
    n_steps = max(1, int(n_steps_per_unit_time * t))
    trotter_state = trotter_time_evo(H, initial_state, t, n_steps)

    # find expectation values for X, Y, Z on every qubit
    for obs_label, obs_list in observable_dict.items():
        # loop over qubits
        for i in range(num_qubits):
            exact_exp = compute_expectation_value(exact_state, obs_list[i])
            trotter_exp = compute_expectation_value(trotter_state, obs_list[i])

            exact_expectations[f'{obs_label}_{i}'].append(exact_exp)
            trotter_expectations[f'{obs_label}_{i}'].append(trotter_exp)

print("Done!")

# plotting X, Y, Z for each qubit

fig, axes = plt.subplots(3, num_qubits, figsize=(4 * num_qubits, 11), sharex=True)

if num_qubits == 1:
    axes = np.array(axes).reshape(3, 1)

row_labels = ['X', 'Y', 'Z']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for row, obs_label in enumerate(row_labels):
    for i in range(num_qubits):
        ax = axes[row, i]

        ax.plot(
            times,
            exact_expectations[f'{obs_label}_{i}'],
            label='Exact',
            color=colors[i % len(colors)],
            linewidth=2
        )

        ax.plot(
            times,
            trotter_expectations[f'{obs_label}_{i}'],
            label='Trotter',
            color=colors[i % len(colors)],
            linestyle='--',
            linewidth=2,
            alpha=0.8
        )

        ax.set_title(f'Qubit {i}: ⟨{obs_label}_{i}⟩ vs Time', fontsize=12)
        ax.set_ylabel(f'⟨{obs_label}_{i}⟩', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, t_max])
        ax.set_ylim([-1.1, 1.1])

        if row == 2:
            ax.set_xlabel('Time', fontsize=11)

        if row == 0 and i == 0:
            ax.legend(fontsize=10)

plt.suptitle(
    'Heisenberg Chain: Exact vs Trotter Time Evolution\n'
    f'Local X, Y, Z expectation values (n_steps = {n_steps_per_unit_time} per unit time)',
    fontsize=15
)
plt.tight_layout()
plt.savefig('trotter_vs_exact_xyz.png', dpi=150, bbox_inches='tight')
plt.show()




#our trotter with noise 
L, Jz, steps, t_tot = 5, -1.1, 50, 5.0
t_values = np.linspace(0, t_tot, steps + 1)

H = hamiltonian1(L, Jz)

psi0 = np.array(initialise(Jz, L, allzeros=True, rotate_middle=True))
qc_step = QuantumCircuit(L)
trotter_step_1st_order(qc_step, H, t_tot / steps)
sv = Statevector(psi0)
noiseless_states = [psi0]
for _ in range(steps):
    sv = sv.evolve(qc_step)
    noiseless_states.append(np.array(sv))

noise_configs = [
    ('X',           'X-only (bit flip)'),
    ('Z',           'Z-only (phase flip)'),
    ('depolarizing','Depolarizing'),
]

pauli_configs = [
    (pauli_x, 'X'),
    (pauli_y, 'Y'),
    (pauli_z, 'Z'),
]

colors = ['orangered', 'gold', 'steelblue', 'm', 'limegreen']

for rate in [0.001, 0.005]:
    for obs_idx, (pauli_mat, obs_label) in enumerate(pauli_configs):

        fig, axes = plt.subplots(
            3, L,
            figsize=(4 * L, 4 * 3),
            sharex=True, sharey=True
        )
        fig.suptitle(
            rf'Observable $\langle {obs_label} \rangle$,  $p = {rate}$',
            fontsize=15, y=1.01
        )

        for row, (noise_type, row_label) in enumerate(noise_configs):

            noisy_states = trotter_with_noise_model(
                L, Jz, steps, t_tot, rate, noise_type=noise_type
            )

            for q in range(L):
                ax = axes[row, q]

                t_ex, ex = ClassicalComparison(L, Jz, q + 1, obs_label, steps, t_tot)

                obs = observable_matrix(L, q, pauli_mat)
                noiseless_vals = [expectation_value(psi, obs) for psi in noiseless_states]
                noisy_vals     = [expectation_value(psi, obs) for psi in noisy_states]

                col = colors[q]
                ax.plot(t_ex,       ex,             color=col, linestyle='--', lw=1.2, label='Exact')
                ax.plot(t_values,   noiseless_vals, color=col, linestyle=':',  lw=1.2, label='Trotter')
                ax.plot(t_values,   noisy_vals,     color='k', linestyle='-',  lw=0.9, alpha=0.75, label='Noisy')

                ax.set_ylim(-1.1, 1.1)
                ax.set_title(rf'$q_{q}$ — {row_label}', fontsize=8)

                if row == 2:
                    ax.set_xlabel(r'$t\ (\mathrm{eV}^{-1})$', fontsize=8)
                if q == 0:
                    ax.set_ylabel(rf'$\langle {obs_label} \rangle$', fontsize=8)

                ax.legend(fontsize=6, loc='upper right')

        plt.tight_layout()
        fname = f'obs_{obs_label}_rate_{str(rate).replace(".", "_")}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'Saved: {fname}')




#Mchlachan data

L       = 45
Jz      = -1.1
steps   = 300
t_total = 5.0
dt      = t_total / steps
allzeros = True
psi_target = np.array(initialise(Jz, L, allzeros, rotate_middle=True)) #gets initial qubits
H_mat      = hamiltonian1(L, Jz).to_matrix() #hamiltonian

n_params = 3 * L * n_layers #depth of ansatz
theta0   = fit_theta_to_state(psi_target, L, n_params, n_restarts=10) #velocity papameter

z_obs = observable_matrix(L, L//2, pauli_z) 
print(f"⟨Z⟩ middle qubit at t=0: {expectation_value(psi_target, z_obs):.4f}  (expect ~0)")

print("Running McLachlan evolution...")
mc_states = mclachlan_evolve(theta0, H_mat, L, dt, steps)
t_values  = np.arange(len(mc_states)) * dt

fig, axes = plt.subplots(1, 3, figsize=(18, 5)) #plots expectation values over time
for ax, pauli_mat, label in zip(axes, [pauli_x, pauli_y, pauli_z], ['X','Y','Z']):
    for q in range(L):
        obs  = observable_matrix(L, q, pauli_mat)
        vals = [expectation_value(psi, obs) for psi in mc_states]
        ax.plot(t_values, vals, '.-', label=rf'$q_{q}$')
    ax.set_title(rf'$\langle {label} \rangle$ over time — McLachlan')
    ax.set_xlabel(r'$t$')
    ax.set_ylim(-1, 1)
    ax.legend()
plt.tight_layout()
plt.show()



# theoreticaly vs exact
# THIS CODE BLOCK TAKES ~5 mins TO RUN
# --- 5.1 TheoreticalST vs Exact (benchmark, not hardware-compatible): Jz = 1.1 ---
# Shows that the Qiskit built-in Trotter matches almost exactly when circuit depth grows freely
L, Jz, steps, t_tot = 5, 1.1, 150, 5

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, direction, label in zip(axes, ['X','Y','Z'], ['X','Y','Z']):
    for q in range(1, L + 1):
        t_ex, ex   = ClassicalComparison(L, Jz, q, direction, steps, t_tot)
        t_th, th_  = TheoreticalST(L, Jz, q, direction, steps, t_tot, allzeros, order=1, reps=1)
        ax.plot(t_ex, ex,       label=rf'Exact $\langle {label}_{q}\rangle$')
        ax.plot(t_th, th_, '--', label=rf'TheoreticalST $\langle {label}_{q}\rangle$')
    ax.set_xlabel(r'$t\ (eV^{-1})$')
    ax.set_ylabel(rf'$\langle {label}(t) \rangle$')
    ax.set_title(rf'TheoreticalST vs Exact: $\langle {label} \rangle$, $J_z={Jz}$')
    ax.set_ylim([-1.1, 1.1])
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

L, Jz, steps, t_tot, reps = 5, 1.1, 150, 6, 50  
fig, axes = plt.subplots(L, 3, figsize=(18, L * 3))

for q in range(1, L + 1):
    for col, (direction, label) in enumerate(zip(['X', 'Y', 'Z'], ['X', 'Y', 'Z'])):
        ax = axes[q - 1, col]

        t_ex, ex  = ClassicalComparison(L, Jz, q, direction, steps, t_tot)
        t_th, th_ = PhysicalST(L, Jz, q, direction, steps, t_tot,
                                allzeros=True, order=1, reps=reps)  # ← fixed

        ax.plot(t_ex, ex,        label='Exact')
        ax.plot(t_th, th_, '--', label='PhysicalST')
        ax.set_xlabel(r'$t\ (eV^{-1})$')
        ax.set_ylabel(rf'$\langle {label}_{q}(t) \rangle$')
        ax.set_title(rf'$\langle {label}_{q} \rangle$, $J_z={Jz}$')
        ax.set_ylim([-1.1, 1.1])
        ax.legend(fontsize=7)
plt.tight_layout()
plt.show()


# physical vs exact
# --- 5.1 PhysicalST vs Exact: Jz = 1.1 ---
# here, choosing t_tot = 6 to see where they start to diverge, even without noise
L, Jz, steps, t_tot, reps = 5, 1.1, 150, 6, 50

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, direction, label in zip(axes, ['X','Y','Z'], ['X','Y','Z']):
    for q in range(1, L + 1):
        t_ex, ex   = ClassicalComparison(L, Jz, q, direction, steps, t_tot)
        t_st, st_  = PhysicalST(L, Jz, q, direction, steps, t_tot, allzeros=True, order=1, reps=reps)
        ax.plot(t_ex, ex,       label=rf'Exact $\langle {label}_{q}\rangle$')
        ax.plot(t_st, st_, '--', label=rf'PhysicalST $\langle {label}_{q}\rangle$')
    ax.set_xlabel(r'$t\ (eV^{-1})$')
    ax.set_ylabel(rf'$\langle {label}(t) \rangle$')
    ax.set_title(rf'PhysicalST vs Exact: $\langle {label} \rangle$, $J_z={Jz}$')
    ax.set_ylim([-1.1, 1.1])
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

L, Jz, steps, t_tot, reps = 5, 1.1, 150, 6, 50  
fig, axes = plt.subplots(L, 3, figsize=(18, L * 3))
for q in range(1, L + 1):
    for col, (direction, label) in enumerate(zip(['X', 'Y', 'Z'], ['X', 'Y', 'Z'])):
        ax = axes[q - 1, col]

        t_ex, ex  = ClassicalComparison(L, Jz, q, direction, steps, t_tot)
        t_th, th_ = PhysicalST(L, Jz, q, direction, steps, t_tot,
                                allzeros=True, order=1, reps=reps)  # ← fixed

        ax.plot(t_ex, ex,        label='Exact')
        ax.plot(t_th, th_, '--', label='PhysicalST')
        ax.set_xlabel(r'$t\ (eV^{-1})$')
        ax.set_ylabel(rf'$\langle {label}_{q}(t) \rangle$')
        ax.set_title(rf'$\langle {label}_{q} \rangle$, $J_z={Jz}$')
        ax.set_ylim([-1.1, 1.1])
        ax.legend(fontsize=7)
plt.tight_layout()
plt.show()



#IBM FEZ data
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, direction, label in zip(axes, ['X','Y','Z'], ['X','Y','Z']):
    for q in range(1, L + 1):
        col = colors[q - 1]
        d = data[(direction, q)]
        ax.plot(d['t_ex'], d['ex'],  color=col, linestyle='-')
        ax.plot(d['t_st'], d['st'], color=col, linestyle='--')
        ax.plot(d['t_ns'], d['ns'], color=col, linestyle=':')

    qubit_handles = [mlines.Line2D([], [], color=colors[q], label=rf'$q_{q+1}$')
                     for q in range(L)]
    style_handles = [mlines.Line2D([], [], color='black', linestyle='-',  label='Exact'),
                     mlines.Line2D([], [], color='black', linestyle='--', label='PhysicalST'),
                     mlines.Line2D([], [], color='black', linestyle=':',  label='Noisy')]

    leg1 = ax.legend(handles=qubit_handles, fontsize=6, loc='upper left')
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, fontsize=6, loc='upper right')

    ax.set_xlabel(r'$t\ (eV^{-1})$')
    ax.set_ylabel(rf'$\langle {label}(t) \rangle$')
    ax.set_title(rf'PhysicalST + IBM Fez Noise: $\langle {label} \rangle$, {reps} ST reps')
    ax.set_ylim([-1.1, 1.1])

plt.tight_layout()
plt.show()



#IBM quantum computer data and plots 
# Load the Qiskit Runtime service
from qiskit_ibm_runtime import QiskitRuntimeService, Session
try:
    with open('apikey3.json') as file:
        api_key = json.load(file)
except:
    print("No API key found: make sure to modify this code and enter your API key manually in the below command.")
QiskitRuntimeService.save_account(channel='ibm_quantum_platform', instance="open-instance", token=api_key["apikey"], overwrite=True, set_as_default=True)

# # Load saved credentials
service = QiskitRuntimeService()
# # Use the least busy backend, or uncomment the loading of a specific backend like "ibm_brisbane".
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
L=5
Jz=1.1
reps=10
qubit_measured = 3
direction_measured = 'X'
#creating hamiltonian
hamiltonian = hamiltonian1(L,Jz)
#Setting order and reps of SuzukiTrotter method
st = SuzukiTrotter(order=1, reps=reps)
psi0 = initialise(Jz, L, allzeros=True, rotate_middle=True)
# Define the observable we wish to estimate
paulistring=""
for i in range(L):
     if i == qubit_measured - 1:
         paulistring+=direction_measured
     else:
         paulistring+="I"
obs = SparsePauliOp([paulistring],coeffs=[1])
target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)
estimator = Estimator(mode=backend)
#t_values = np.log10(np.array(range(1,100,10)))
t_values = np.logspace(-2, np.log10(5), 10)
xmeasurements=[]
for timeval in t_values:
        gate = PauliEvolutionGate(operator=hamiltonian, time=timeval)
        circ = st.synthesize(gate)
        qc   = QuantumCircuit(L)
        qc.initialize(psi0.data)
        qc.append(circ, range(L))
        isa_qc = pm.run(qc)
        isa_obs = obs.apply_layout(isa_qc.layout)
        result = estimator.run(pubs=[(isa_qc, [isa_obs])]).result()
        xmeasurements.append(result[0].data.evs[0]) 
tclass,classic = ClassicalComparison(L,Jz,qubit_measured,direction_measured,t_tot=t_values[-1])
tphys, phys = PhysicalST(L, Jz, qubit_measured, direction_measured,t_tot=t_values[-1])

plt.figure(figsize=(18,5))

# Data from quantum hardware
plt.plot(t_values, xmeasurements, 'o', label="Real Quantum Computer Chip")
# Suzuki–Trotter physical simulation
plt.plot(tphys, phys, label="Physical Suzuki–Trotter")
# Classical model
plt.plot(tclass, classic, label="Classical Model")

plt.xlabel(r'$t\ (eV^{-1})$')
plt.ylabel(rf'$\langle {direction_measured}(t) \rangle$')
plt.title(rf'Time evolution of $\langle {direction_measured}(t) \rangle$ (qubit N°{qubit_measured} of 5 qubits), $J_z={Jz}$')

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show() #was then changed for each expectation value