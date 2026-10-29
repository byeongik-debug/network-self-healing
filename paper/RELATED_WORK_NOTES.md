# Related-work positioning notes

Updated 2026-10-27 from publisher or primary-paper pages. These notes support
Section 1.1 of `DRAFT.md`; they are not a systematic review.

| Theme | Prior work | Connection | Distinction to preserve |
|---|---|---|---|
| Safe RL shielding | Alshiekh et al., AAAI 2018, DOI 10.1609/AAAI.V32I1.11797 | Model-based shield mediates learner actions | Their central property is safety; our empirical counterexample and lexicographic refinement concern progress under a damage-optimal shield |
| Runtime assurance | Slagel et al., NFM 2024, NASA record 20230017350 | Trusted runtime component constrains an unverified controller | Our setting is a finite discrete network configuration graph, not hybrid-system Simplex switching |
| Cellular self-healing | Mismar and Evans, VTC 2014, DOI 10.1109/VTCFall.2014.6965842 | RL selects compensating actions after outage | Cellular power/tilt optimization differs from configuration-policy damage |
| Cellular fault management | Mismar et al., Asilomar 2018, DOI 10.1109/ACSSC.2018.8645083 | Deep Q-learning selects alarm corrective actions | Focuses radio performance; does not supply our damage/progress certificate |
| Intent-based SDN recovery | Makiyah et al., Discover Networks 2026, DOI 10.1007/s44354-026-00028-z | Closest application scope: heterogeneous emulated SDN faults and RL remediation | Compare emulator breadth versus our formal finite-state safety/liveness analysis |
| GNN network modeling | Rusek et al., RouteNet, arXiv:1910.01508 | Message passing supports variable topology representation | Predictive network modeling, not repair action selection |
| GNN–DRL transfer | Ding et al., JNCA 2024, DOI 10.1016/j.jnca.2024.103927 | Targets routing on unseen topologies | Traffic engineering objective; our T4/T6 study targets fault recovery and shielding |

## Writing cautions

- Do not claim shielding itself is novel; the contribution is the demonstrated
  safety–liveness failure mode and lexicographic progress refinement in this
  network-remediation model.
- Do not claim the first use of RL for network self-healing.
- Do not compare recovery percentages across papers with different faults,
  topology sizes, information access, or success definitions.
- The 2026 intent-based SDN paper raises the expected empirical bar for Mininet
  or equivalent emulation; until our harness is executed, retain the
  simulator-only limitation.
