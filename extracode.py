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
L, Jz, steps, t_tot = 4, 1.1, 150, 5

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


# physical vs exact
# --- 5.1 PhysicalST vs Exact: Jz = 1.1 ---
# here, choosing t_tot = 6 to see where they start to diverge, even without noise
L, Jz, steps, t_tot, reps = 4, 1.1, 150, 6, 50

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