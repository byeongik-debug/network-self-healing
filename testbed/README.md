# Mininet validation

This directory contains the emulator validation for the paper. It must be run
on Linux with root privileges, Mininet, and Open vSwitch installed.

```bash
sudo -E bash testbed/run_mininet_validation.sh
```

The wrapper creates a new timestamped directory under `results/testbed/runs/`,
runs the parser preflight, records Mininet/Open vSwitch versions, rejects result
overwrites, performs all 51 replays, and exits nonzero unless both recovery and
damage match the simulator for all 51 scenarios. Set `MININET_RUN_ID` to a
unique archival identifier when a stable directory name is required.

The harness builds the same two-switch/four-host topology as the simulator,
configures access VLANs and an 802.1Q trunk through `ovs-vsctl`, injects every
held-out fault, replays the frozen Progress-Safe trajectory, and probes all six
unordered host pairs after each action. A run is publishable evidence only when
the output reports 51/51 recovery matches and 51/51 damage matches. The script
must not be described as executed merely because it imports or passes unit
tests; the generated `results/testbed/*.json` file is the evidence.

For a smoke test, add `--limit 3`. Always run `sudo mn -c` after an interrupted
testbed process.
