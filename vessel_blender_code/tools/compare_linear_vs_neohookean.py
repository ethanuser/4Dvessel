import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import vessel_utils as vu

# Parameters
E = vu.YOUNG_MODULUS_SILICON  # Young's modulus (Pa)

# Strain range
eps = np.linspace(0.0, 0.5, 501)   # 0–50% engineering strain
lam = 1.0 + eps

# Stress models
sigma_linear = E * eps
sigma_neo = mu * (lam**2 - 1.0 / lam)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(eps * 100, sigma_linear / 1e6, label="Linear elastic: σ = E ε")
plt.plot(eps * 100, sigma_neo / 1e6, label="Neo-Hookean (incompressible)")

# Mark 20% strain
eps_mark = 0.2
lam_mark = 1.0 + eps_mark
plt.scatter([eps_mark * 100], [E * eps_mark / 1e6], marker="o")
plt.scatter([eps_mark * 100], [mu * (lam_mark**2 - 1.0 / lam_mark) / 1e6], marker="o")

plt.xlabel("Engineering strain ε (%)")
plt.ylabel("Stress σ (MPa)")
plt.title(f"Stress–strain comparison (E = {E/1e6:.2f} MPa)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Print stresses at 20% strain
sigma_linear_20 = E * eps_mark
sigma_neo_20 = mu * (lam_mark**2 - 1.0 / lam_mark)

sigma_linear_20 / 1e6, sigma_neo_20 / 1e6