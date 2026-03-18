#Mchlachan time evolution and printing
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

