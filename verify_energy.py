"""Energy-conservation check for the membrane-ball dynamics.

The dynamics live in exactly one place, ``membrane-rl/membrane_rl/physics.py``.
This script drives that implementation (no private copy) and checks that total
mechanical energy, sampled at the end of each rendered frame, stays within 5 %
of its initial value.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "membrane-rl"))
from membrane_rl.physics import MembraneSim, Params, State  # noqa: E402

FRAMES = 5000
THRESHOLD_PCT = 5.0


def main():
    sim = MembraneSim(Params())
    sim.reset(State(pos=np.array([0.2, 0.1, 4.0]), vel=np.array([1.5, 1.0, 0.0])))

    initial_energy = sim.energy()["total"]
    max_dev = 0.0

    for _ in range(FRAMES):
        sim.step()
        max_dev = max(max_dev, abs(sim.energy()["total"] - initial_energy))

    rel_leak = (max_dev / initial_energy) * 100.0
    print(f"Initial Energy: {initial_energy:.6f} J")
    print(f"Max Deviation: {max_dev:.6f} J")
    print(f"Relative Leak: {rel_leak:.6f}%")
    print(f"Cylinder Bounces: {sim.n_cyl_bounces}, Ring Bounces: {sim.n_ring_bounces}")

    if rel_leak <= THRESHOLD_PCT:
        print(f"Verification SUCCESS: energy drift within {THRESHOLD_PCT:.1f}% tolerance.")
        sys.exit(0)
    print("Verification FAILURE: Energy leak exceeds threshold!")
    sys.exit(1)


if __name__ == "__main__":
    main()
