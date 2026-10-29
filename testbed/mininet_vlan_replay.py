"""Replay Progress-Safe trajectories on a real Open vSwitch Mininet topology.

Run on Linux as root after installing Mininet and Open vSwitch.  The script
recreates the simulator's two-switch/four-host VLAN topology, injects each fault,
replays a frozen action trajectory, and measures policy health with real ICMP
probes after every state-changing action.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from config.topology import HOSTS, REFERENCE_PORTS
from env.faults import FaultDataset, FaultMutation


HOST_IPS = {"H1": "10.0.0.1", "H2": "10.0.0.2", "H3": "10.0.0.3", "H4": "10.0.0.4"}
ACTION_PATTERN = re.compile(
    r"(?P<kind>set_access_vlan|set_trunk_vlan)\("
    r"(?P<switch>SW[12]),(?P<port>eth[123]),vlan=(?P<vlan>10|20)"
    r"(?:,(?P<operation>allow|remove))?\)"
)


@dataclass(frozen=True)
class ReplayResult:
    scenario_id: str
    expected_simulator_recovery: bool
    testbed_recovery: bool
    declared_success: bool
    expected_simulator_damage: int
    testbed_damage: int
    policy_mismatch_steps: int
    action_count: int
    actions: tuple[str, ...]


def ovs(*arguments: str) -> None:
    subprocess.run(["ovs-vsctl", *arguments], check=True, capture_output=True, text=True)


def interface(switch: str, port: str) -> str:
    return f"{switch.lower()}-{port}"


class MininetVlanTestbed:
    def __init__(self, settle_seconds: float = 0.1) -> None:
        try:
            from mininet.link import TCLink
            from mininet.net import Mininet
            from mininet.node import OVSSwitch
        except ImportError as error:
            raise RuntimeError("Mininet is required; install it on Linux") from error
        self.settle_seconds = settle_seconds
        self.net = Mininet(controller=None, switch=OVSSwitch, link=TCLink, build=False)
        self.switches = {
            "SW1": self.net.addSwitch("s1", failMode="standalone"),
            "SW2": self.net.addSwitch("s2", failMode="standalone"),
        }
        self.hosts = {
            name: self.net.addHost(name.lower(), ip=f"{HOST_IPS[name]}/24")
            for name in HOSTS
        }
        self.net.addLink(self.hosts["H1"], self.switches["SW1"], port2=1)
        self.net.addLink(self.hosts["H2"], self.switches["SW1"], port2=2)
        self.net.addLink(self.hosts["H3"], self.switches["SW2"], port2=1)
        self.net.addLink(self.hosts["H4"], self.switches["SW2"], port2=2)
        self.net.addLink(self.switches["SW1"], self.switches["SW2"], port1=3, port2=3)
        self.net.build()
        self.net.start()
        self.configuration: dict[str, dict[str, dict[str, Any]]] = {}
        self.reset_reference()

    def stop(self) -> None:
        self.net.stop()

    def reset_reference(self) -> None:
        self.configuration = json.loads(json.dumps(REFERENCE_PORTS))
        self._apply_full_configuration()

    def _apply_full_configuration(self) -> None:
        for switch, ports in self.configuration.items():
            for port, settings in ports.items():
                name = interface(switch, port)
                ovs("clear", "Port", name, "tag")
                ovs("clear", "Port", name, "trunks")
                if settings["mode"] == "access":
                    ovs("set", "Port", name, f"tag={settings['access_vlan']}")
                else:
                    vlans = ",".join(str(vlan) for vlan in settings["allowed_vlans"])
                    ovs("set", "Port", name, f"trunks={vlans}")
        self._settle()

    def set_access_vlan(self, switch: str, port: str, vlan: int) -> None:
        self.configuration[switch][port]["access_vlan"] = vlan
        ovs("set", "Port", interface(switch, port), f"tag={vlan}")
        self._settle()

    def set_trunk_vlan(self, switch: str, port: str, vlan: int, allowed: bool) -> None:
        current = set(self.configuration[switch][port]["allowed_vlans"])
        current.add(vlan) if allowed else current.discard(vlan)
        self.configuration[switch][port]["allowed_vlans"] = sorted(current)
        value = ",".join(str(item) for item in sorted(current))
        assignment = f"trunks={value}" if value else "trunks=[]"
        ovs("set", "Port", interface(switch, port), assignment)
        self._settle()

    def apply_fault(self, mutation: FaultMutation) -> None:
        if mutation.kind == "access_vlan":
            self.set_access_vlan(mutation.switch, mutation.port, mutation.vlan)
        elif mutation.kind == "trunk_remove":
            self.set_trunk_vlan(mutation.switch, mutation.port, mutation.vlan, False)
        else:
            raise ValueError(f"unsupported fault mutation: {mutation.kind}")

    def policy_results(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        names = sorted(self.hosts)
        for index, source in enumerate(names):
            for destination in names[index + 1:]:
                command = f"ping -n -q -c 1 -W 1 {HOST_IPS[destination]}"
                reachable = self.hosts[source].cmd(command).find("1 received") >= 0
                expected = HOSTS[source]["vlan"] == HOSTS[destination]["vlan"]
                results[f"{source}-{destination}"] = reachable == expected
        return results

    def _settle(self) -> None:
        for host in self.hosts.values():
            host.cmd("ip neigh flush all")
        time.sleep(self.settle_seconds)


def parse_action(label: str) -> dict[str, Any]:
    if label in ("diagnose", "rollback", "declare_done"):
        return {"kind": label}
    match = ACTION_PATTERN.fullmatch(label)
    if not match:
        raise ValueError(f"unrecognized trajectory action: {label}")
    result: dict[str, Any] = match.groupdict()
    result["vlan"] = int(result["vlan"])
    if result["kind"] == "set_trunk_vlan" and result["operation"] is None:
        raise ValueError(f"trunk action requires allow/remove: {label}")
    if result["kind"] == "set_access_vlan" and result["operation"] is not None:
        raise ValueError(f"access action cannot include allow/remove: {label}")
    return result


def validate_trajectories(
    trajectories: list[dict[str, Any]], scenarios: list[Any]
) -> dict[str, dict[str, Any]]:
    """Fail before changing the testbed if the frozen replay set is incomplete."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in trajectories:
        scenario_id = str(row.get("scenario_id", ""))
        if not scenario_id or scenario_id in by_id:
            raise ValueError(f"missing or duplicate trajectory scenario_id: {scenario_id!r}")
        actions = row.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"trajectory has no actions: {scenario_id}")
        for action in actions:
            parse_action(str(action))
        if "recovery_success" not in row or "new_policy_damage_events" not in row:
            raise ValueError(f"trajectory lacks expected outcomes: {scenario_id}")
        by_id[scenario_id] = row
    expected = {scenario.scenario_id for scenario in scenarios}
    missing = sorted(expected - set(by_id))
    if missing:
        raise ValueError(f"missing trajectories for {len(missing)} scenarios: {missing}")
    return by_id


def command_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as error:
        return f"unavailable: {error}"
    text = (completed.stdout or completed.stderr).strip()
    return text.splitlines()[0] if text else f"exit={completed.returncode}"


def replay(
    testbed: MininetVlanTestbed,
    scenario: Any,
    trajectory: dict[str, Any],
) -> ReplayResult:
    testbed.reset_reference()
    for mutation in scenario.mutations:
        testbed.apply_fault(mutation)
    previous = testbed.policy_results()
    damage = mismatch_steps = 0
    history: list[dict[str, dict[str, dict[str, Any]]]] = []
    declared_success = False
    labels = tuple(str(label) for label in trajectory["actions"])
    for label in labels:
        action = parse_action(label)
        kind = action["kind"]
        if kind == "set_access_vlan":
            history.append(json.loads(json.dumps(testbed.configuration)))
            testbed.set_access_vlan(action["switch"], action["port"], action["vlan"])
        elif kind == "set_trunk_vlan":
            history.append(json.loads(json.dumps(testbed.configuration)))
            testbed.set_trunk_vlan(
                action["switch"], action["port"], action["vlan"],
                action["operation"] == "allow",
            )
        elif kind == "rollback":
            if not history:
                raise RuntimeError("trajectory requested rollback without history")
            testbed.configuration = history.pop()
            testbed._apply_full_configuration()
        current = testbed.policy_results()
        damage += sum(previous[key] and not current[key] for key in previous)
        mismatch_steps += int(not all(current.values()))
        previous = current
        if kind == "declare_done":
            declared_success = all(current.values())
            break
    return ReplayResult(
        scenario_id=scenario.scenario_id,
        expected_simulator_recovery=bool(trajectory["recovery_success"]),
        testbed_recovery=all(previous.values()),
        declared_success=declared_success,
        expected_simulator_damage=int(trajectory["new_policy_damage_events"]),
        testbed_damage=damage,
        policy_mismatch_steps=mismatch_steps,
        action_count=len(labels),
        actions=labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        default="results/multiseed_progress/progress_safe_seed_2026_trajectories.json",
    )
    parser.add_argument("--output", default="results/testbed/mininet_replay_seed_2026.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--settle-seconds", type=float, default=0.1)
    args = parser.parse_args()
    if os.name != "posix" or os.geteuid() != 0:
        raise SystemExit("run this Mininet testbed as root on Linux")

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing result: {output}")
    trajectories_path = Path(args.trajectories)
    trajectories = json.loads(trajectories_path.read_text(encoding="utf-8"))
    scenarios = list(FaultDataset().stratified_split("eval", 2026))
    if args.limit is not None:
        scenarios = scenarios[:args.limit]
    by_id = validate_trajectories(trajectories, scenarios)
    testbed = MininetVlanTestbed(args.settle_seconds)
    try:
        results = [replay(testbed, scenario, by_id[scenario.scenario_id]) for scenario in scenarios]
    finally:
        testbed.stop()
        subprocess.run(["mn", "-c"], check=False)

    report = {
        "environment": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "platform": sys.platform,
            "python": sys.version,
            "mininet": command_version(["mn", "--version"]),
            "open_vswitch": command_version(["ovs-vsctl", "--version"]),
            "settle_seconds": args.settle_seconds,
            "trajectory_source": str(trajectories_path),
        },
        "summary": {
            "scenario_count": len(results),
            "testbed_recovered": sum(row.testbed_recovery for row in results),
            "declared_success": sum(row.declared_success for row in results),
            "simulator_testbed_recovery_matches": sum(
                row.expected_simulator_recovery == row.testbed_recovery for row in results
            ),
            "simulator_testbed_damage_matches": sum(
                row.expected_simulator_damage == row.testbed_damage for row in results
            ),
            "testbed_total_damage": sum(row.testbed_damage for row in results),
        },
        "results": [asdict(row) for row in results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
