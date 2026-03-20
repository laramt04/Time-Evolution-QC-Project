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