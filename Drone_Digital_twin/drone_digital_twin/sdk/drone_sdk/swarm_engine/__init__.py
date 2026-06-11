"""
drone_sdk.swarm_engine
======================
Multi-Drone Swarm Engine — Module 21 of the UAV Digital Twin Platform.

Implements:

1. **SwarmVehicle** — per-vehicle state + control wrapper
2. **FormationController** — leader-follower, virtual structure, and
   consensus-based formation controllers
3. **SwarmCommunication** — range-limited message passing with packet loss
   and latency simulation
4. **SwarmManager** — orchestrates N vehicles, handles joining/leaving,
   applies collision avoidance
5. **DefenseSim** — GPS denial, RF jamming, adversarial drone scenarios

Python version: 3.9+
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Formation types
# ─────────────────────────────────────────────────────────────────────────────

class FormationType(str, Enum):
    LINE         = "line"
    V_SHAPE      = "v_shape"
    WEDGE        = "wedge"
    DIAMOND      = "diamond"
    CIRCLE       = "circle"
    GRID         = "grid"
    LEADER_FOLLOW = "leader_follow"


# ─────────────────────────────────────────────────────────────────────────────
#  Per-vehicle state wrapper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SwarmVehicle:
    """State and metadata for one drone in the swarm."""
    vehicle_id:    str
    position:      np.ndarray       # (3,) NED, metres
    velocity:      np.ndarray       # (3,) NED, m/s
    heading:       float            # radians
    is_leader:     bool = False
    is_active:     bool = True
    battery_soc:   float = 1.0
    last_seen:     float = field(default_factory=time.monotonic)

    def distance_to(self, other: "SwarmVehicle") -> float:
        return float(np.linalg.norm(self.position - other.position))

    def bearing_to(self, other: "SwarmVehicle") -> float:
        delta = other.position[:2] - self.position[:2]
        return float(math.atan2(delta[1], delta[0]))

    @property
    def is_stale(self) -> bool:
        return (time.monotonic() - self.last_seen) > 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Formation geometry builder
# ─────────────────────────────────────────────────────────────────────────────

class FormationGeometry:
    """Computes desired formation offsets for each follower relative to leader.

    All offsets are in the leader's body frame (x=forward, y=right, z=down).
    They are converted to NED at runtime using the leader's heading.

    Usage::

        offsets = FormationGeometry.compute(FormationType.V_SHAPE, n_drones=5)
        # Returns list of (3,) NED offset arrays
    """

    @staticmethod
    def compute(
        formation:  FormationType,
        n_drones:   int,
        spacing:    float = 5.0,   # metres between drones
    ) -> List[np.ndarray]:
        """Return per-drone offset from swarm reference point."""
        n = n_drones
        if formation == FormationType.LINE:
            return [np.array([0.0, i * spacing, 0.0]) for i in range(n)]

        elif formation == FormationType.V_SHAPE:
            offsets = [np.zeros(3)]   # lead
            for i in range(1, n):
                side  = 1 if i % 2 == 1 else -1
                depth = math.ceil(i / 2)
                offsets.append(np.array([depth * spacing, side * depth * spacing * 0.7, 0.0]))
            return offsets

        elif formation == FormationType.WEDGE:
            offsets = [np.zeros(3)]
            for i in range(1, n):
                side = 1 if i % 2 == 0 else -1
                row  = (i + 1) // 2
                offsets.append(np.array([-row * spacing * 0.8, side * row * spacing * 0.5, 0.0]))
            return offsets

        elif formation == FormationType.DIAMOND:
            if n < 4:
                return FormationGeometry.compute(FormationType.LINE, n, spacing)
            return [
                np.array([spacing, 0.0,     0.0]),   # front
                np.array([0.0,     spacing,  0.0]),   # right
                np.array([0.0,    -spacing,  0.0]),   # left
                np.array([-spacing, 0.0,     0.0]),   # rear
            ] + [np.array([0.0, 0.0, 0.0])] * max(0, n - 4)

        elif formation == FormationType.CIRCLE:
            angles  = np.linspace(0, 2 * math.pi, n, endpoint=False)
            radius  = spacing * n / (2 * math.pi)
            return [np.array([radius * math.cos(a), radius * math.sin(a), 0.0]) for a in angles]

        elif formation == FormationType.GRID:
            cols  = math.ceil(math.sqrt(n))
            return [
                np.array([(i // cols) * spacing, (i % cols) * spacing, 0.0])
                for i in range(n)
            ]

        else:  # LEADER_FOLLOW: single file behind leader
            return [np.array([-i * spacing, 0.0, 0.0]) for i in range(n)]

    @staticmethod
    def rotate_to_heading(offset: np.ndarray, heading: float) -> np.ndarray:
        """Rotate a formation offset vector to align with leader heading."""
        c, s = math.cos(heading), math.sin(heading)
        R    = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return R @ offset


# ─────────────────────────────────────────────────────────────────────────────
#  Formation controller
# ─────────────────────────────────────────────────────────────────────────────

class FormationController:
    """Computes velocity commands to maintain formation.

    Uses a combination of:
    - Formation error term (attraction to desired position)
    - Velocity matching term (consensus)
    - Collision avoidance term (repulsion from nearby drones)

    Usage::

        ctrl    = FormationController(spacing=5.0)
        leader  = swarm.get_leader()
        for follower in swarm.followers:
            offset  = formation_offsets[follower.vehicle_id]
            cmd_vel = ctrl.compute_velocity_command(leader, follower, offset)
    """

    def __init__(
        self,
        k_formation:   float = 1.5,    # Formation error gain
        k_velocity:    float = 0.8,    # Velocity matching gain
        k_repulsion:   float = 2.0,    # Collision avoidance gain
        safe_radius_m: float = 2.0,    # Minimum inter-drone distance
        max_speed_ms:  float = 8.0,    # Maximum command speed
    ) -> None:
        self._kf   = k_formation
        self._kv   = k_velocity
        self._kr   = k_repulsion
        self._safe = safe_radius_m
        self._vmax = max_speed_ms

    def compute_velocity_command(
        self,
        leader:       SwarmVehicle,
        follower:     SwarmVehicle,
        desired_offset: np.ndarray,
        neighbors:    Optional[List[SwarmVehicle]] = None,
    ) -> np.ndarray:
        """Compute 3-D velocity command for a follower.

        Args:
            leader:          Leader vehicle state.
            follower:        This follower's current state.
            desired_offset:  Desired NED offset from leader (already rotated).
            neighbors:       Other nearby drones for collision avoidance.

        Returns:
            (3,) velocity command in NED frame (m/s).
        """
        # Desired position
        desired_pos  = leader.position + desired_offset
        pos_error    = desired_pos - follower.position
        formation_cmd = self._kf * pos_error

        # Velocity matching
        vel_cmd       = self._kv * (leader.velocity - follower.velocity)

        # Collision avoidance
        repulsion_cmd = np.zeros(3)
        if neighbors:
            for n in neighbors:
                if n.vehicle_id == follower.vehicle_id:
                    continue
                dist = follower.distance_to(n)
                if 0 < dist < self._safe:
                    direction = follower.position - n.position
                    if np.linalg.norm(direction) > 0:
                        direction /= np.linalg.norm(direction)
                    repulsion_cmd += self._kr * (self._safe / dist - 1.0) * direction

        cmd = formation_cmd + vel_cmd + repulsion_cmd
        # Speed limiting
        speed = np.linalg.norm(cmd)
        if speed > self._vmax:
            cmd = cmd * self._vmax / speed
        return cmd


# ─────────────────────────────────────────────────────────────────────────────
#  Swarm communication model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CommConfig:
    """Inter-drone communication parameters."""
    range_m:       float = 500.0   # Maximum communication range
    packet_loss:   float = 0.02    # Probability of packet drop per message
    latency_ms:    Tuple[float,float] = (5.0, 30.0)  # Min/max latency
    bandwidth_kbps: float = 250.0  # Available bandwidth


@dataclass
class SwarmMessage:
    """One inter-drone message."""
    sender_id:    str
    receiver_id:  str    # "" = broadcast
    payload:      dict
    timestamp:    float = field(default_factory=time.monotonic)
    latency_ms:   float = 0.0
    delivered:    bool  = False


class SwarmCommunication:
    """Simulates range-limited inter-drone communication with packet loss.

    Usage::

        comm = SwarmCommunication(CommConfig())
        ok   = comm.send("drone_0", "drone_1", {"pos": [1,2,3]}, vehicles)
        msgs = comm.receive("drone_1")
    """

    def __init__(self, config: Optional[CommConfig] = None, seed: int = 0) -> None:
        self._cfg  = config or CommConfig()
        self._rng  = np.random.default_rng(seed)
        self._inbox: Dict[str, List[SwarmMessage]] = {}
        self._total_sent:     int = 0
        self._total_dropped:  int = 0

    def send(
        self,
        sender_id:   str,
        receiver_id: str,
        payload:     dict,
        vehicles:    List[SwarmVehicle],
    ) -> bool:
        """Send a message from sender to receiver (or broadcast if receiver_id='').

        Returns True if message was delivered (not dropped by packet loss).
        """
        self._total_sent += 1

        # Check range
        sender   = next((v for v in vehicles if v.vehicle_id == sender_id), None)
        if receiver_id:
            receiver = next((v for v in vehicles if v.vehicle_id == receiver_id), None)
            if sender and receiver:
                dist = sender.distance_to(receiver)
                if dist > self._cfg.range_m:
                    self._total_dropped += 1
                    return False

        # Packet loss
        if self._rng.random() < self._cfg.packet_loss:
            self._total_dropped += 1
            return False

        # Latency
        lo, hi    = self._cfg.latency_ms
        latency   = float(self._rng.uniform(lo, hi))

        msg = SwarmMessage(
            sender_id   = sender_id,
            receiver_id = receiver_id,
            payload     = payload,
            latency_ms  = latency,
            delivered   = True,
        )

        if receiver_id:
            self._inbox.setdefault(receiver_id, []).append(msg)
        else:
            # Broadcast to all active vehicles
            for v in vehicles:
                if v.vehicle_id != sender_id and v.is_active:
                    self._inbox.setdefault(v.vehicle_id, []).append(msg)

        return True

    def receive(self, vehicle_id: str) -> List[SwarmMessage]:
        """Return all pending messages for a vehicle (clears inbox)."""
        msgs = self._inbox.pop(vehicle_id, [])
        return msgs

    @property
    def drop_rate(self) -> float:
        if self._total_sent == 0:
            return 0.0
        return self._total_dropped / self._total_sent

    def get_stats(self) -> dict:
        return {
            "total_sent":    self._total_sent,
            "total_dropped": self._total_dropped,
            "drop_rate":     round(self.drop_rate, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Swarm Manager
# ─────────────────────────────────────────────────────────────────────────────

class SwarmManager:
    """Orchestrates an N-drone swarm with formation control and communication.

    Usage::

        manager = SwarmManager(n_drones=5)
        manager.set_formation(FormationType.V_SHAPE)
        manager.set_leader("drone_0")

        for step in sim_loop:
            manager.tick(dt=0.02)
            cmds = manager.get_velocity_commands()
    """

    def __init__(
        self,
        n_drones:        int             = 4,
        formation:       FormationType   = FormationType.V_SHAPE,
        spacing:         float           = 5.0,
        comm_config:     Optional[CommConfig] = None,
    ) -> None:
        self._n       = n_drones
        self._formation = formation
        self._spacing   = spacing
        self._comm      = SwarmCommunication(comm_config)
        self._ctrl      = FormationController(safe_radius_m=spacing * 0.4)
        self._vehicles: Dict[str, SwarmVehicle] = {}
        self._leader_id: Optional[str] = None

        # Initialise vehicles in a line
        for i in range(n_drones):
            vid = f"drone_{i}"
            self._vehicles[vid] = SwarmVehicle(
                vehicle_id = vid,
                position   = np.array([float(i * spacing), 0.0, -5.0]),
                velocity   = np.zeros(3),
                heading    = 0.0,
                is_leader  = (i == 0),
                is_active  = True,
            )
        if n_drones > 0:
            self._leader_id = "drone_0"

    def set_formation(self, formation: FormationType) -> None:
        self._formation = formation

    def set_leader(self, vehicle_id: str) -> None:
        if vehicle_id in self._vehicles:
            for v in self._vehicles.values():
                v.is_leader = (v.vehicle_id == vehicle_id)
            self._leader_id = vehicle_id

    def add_vehicle(self, vehicle: SwarmVehicle) -> None:
        self._vehicles[vehicle.vehicle_id] = vehicle
        self._n = len(self._vehicles)

    def remove_vehicle(self, vehicle_id: str) -> bool:
        if vehicle_id in self._vehicles:
            del self._vehicles[vehicle_id]
            self._n = len(self._vehicles)
            if self._leader_id == vehicle_id:
                active = [v for v in self._vehicles.values() if v.is_active]
                self._leader_id = active[0].vehicle_id if active else None
            return True
        return False

    def tick(self, dt: float = 0.02) -> None:
        """Advance swarm state by one timestep."""
        if not self._leader_id:
            return
        leader = self._vehicles.get(self._leader_id)
        if leader is None:
            return

        # Compute formation offsets for current heading
        offsets = FormationGeometry.compute(self._formation, self._n, self._spacing)
        rotated = [FormationGeometry.rotate_to_heading(o, leader.heading) for o in offsets]

        vehicles_list = list(self._vehicles.values())
        for i, (vid, vehicle) in enumerate(self._vehicles.items()):
            if not vehicle.is_active or vehicle.is_leader:
                continue
            idx = min(i, len(rotated) - 1)
            cmd = self._ctrl.compute_velocity_command(
                leader, vehicle, rotated[idx], vehicles_list
            )
            # Integrate
            vehicle.velocity  = cmd
            vehicle.position += cmd * dt
            vehicle.last_seen = time.monotonic()

    def get_velocity_commands(self) -> Dict[str, np.ndarray]:
        return {vid: v.velocity.copy() for vid, v in self._vehicles.items()}

    def get_positions(self) -> Dict[str, np.ndarray]:
        return {vid: v.position.copy() for vid, v in self._vehicles.items()}

    def inter_drone_distances(self) -> np.ndarray:
        """Return (N, N) distance matrix."""
        vs = list(self._vehicles.values())
        n  = len(vs)
        D  = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = vs[i].distance_to(vs[j])
        return D

    @property
    def n_active(self) -> int:
        return sum(1 for v in self._vehicles.values() if v.is_active)

    @property
    def leader(self) -> Optional[SwarmVehicle]:
        return self._vehicles.get(self._leader_id) if self._leader_id else None

    def get_swarm_centroid(self) -> np.ndarray:
        active = [v.position for v in self._vehicles.values() if v.is_active]
        if not active:
            return np.zeros(3)
        return np.mean(active, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
#  DefenseSim — GPS denial, electronic warfare scenarios
# ─────────────────────────────────────────────────────────────────────────────

class DefenseScenario(str, Enum):
    GPS_DENIAL    = "gps_denial"
    RF_JAMMING    = "rf_jamming"
    SPOOFING      = "spoofing"
    ADVERSARIAL   = "adversarial_drone"


@dataclass
class JammingSource:
    """Electronic warfare threat source."""
    position:      np.ndarray   # NED metres
    power_dbm:     float = 30.0
    frequency_mhz: float = 1575.42   # GPS L1
    radius_m:      float = 500.0


class DefenseSim:
    """Simulates GPS denial, RF jamming, and adversarial drone scenarios.

    Critical for DRDO/defense applications and counter-UAV research.

    Usage::

        sim = DefenseSim()
        sim.add_jammer(JammingSource(np.array([500, 0, -50]), radius_m=300))
        gps_ok = sim.gps_available(vehicle_pos=np.array([100, 0, -50]))
        comm_ok = sim.comm_available("drone_0", "drone_1", vehicles)
    """

    def __init__(self) -> None:
        self._jammers:    List[JammingSource]  = []
        self._spoofers:   List[JammingSource]  = []
        self._adversaries: List[SwarmVehicle]  = []

    def add_jammer(self, jammer: JammingSource) -> None:
        self._jammers.append(jammer)

    def add_spoofer(self, spoofer: JammingSource) -> None:
        self._spoofers.append(spoofer)

    def add_adversary(self, vehicle: SwarmVehicle) -> None:
        self._adversaries.append(vehicle)

    def gps_available(self, vehicle_pos: np.ndarray) -> bool:
        """True if GPS is available at vehicle_pos (not jammed)."""
        for j in self._jammers:
            dist = float(np.linalg.norm(vehicle_pos - j.position))
            if dist < j.radius_m:
                return False
        return True

    def gps_spoofed(self, vehicle_pos: np.ndarray) -> bool:
        """True if vehicle receives spoofed GPS signal."""
        for s in self._spoofers:
            dist = float(np.linalg.norm(vehicle_pos - s.position))
            if dist < s.radius_m:
                return True
        return False

    def spoof_position(
        self,
        true_pos: np.ndarray,
        spoofer:  Optional[JammingSource] = None,
    ) -> np.ndarray:
        """Return a spoofed GPS position (offset from true position)."""
        if spoofer is None and self._spoofers:
            spoofer = self._spoofers[0]
        if spoofer is None:
            return true_pos.copy()
        # Spoof toward the spoofer's location
        offset = (spoofer.position - true_pos) * 0.3
        return true_pos + offset

    def comm_jammed(
        self,
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        frequency_mhz: float = 915.0,
    ) -> bool:
        """True if communication link is jammed."""
        midpoint = (pos_a + pos_b) / 2
        for j in self._jammers:
            # Check if jammer frequency overlaps and midpoint is in range
            freq_match = abs(j.frequency_mhz - frequency_mhz) < 50
            in_range   = float(np.linalg.norm(midpoint - j.position)) < j.radius_m
            if freq_match and in_range:
                return True
        return False

    def get_nearest_adversary(
        self,
        vehicle_pos: np.ndarray,
    ) -> Optional[Tuple[SwarmVehicle, float]]:
        """Return nearest adversarial drone and its distance."""
        if not self._adversaries:
            return None
        closest  = min(self._adversaries,
                       key=lambda a: np.linalg.norm(vehicle_pos - a.position))
        distance = float(np.linalg.norm(vehicle_pos - closest.position))
        return closest, distance

    def threat_level(self, vehicle_pos: np.ndarray) -> float:
        """Aggregate threat level [0=none, 1=critical]."""
        level = 0.0
        if not self.gps_available(vehicle_pos):
            level += 0.4
        if self.gps_spoofed(vehicle_pos):
            level += 0.3
        adversary = self.get_nearest_adversary(vehicle_pos)
        if adversary:
            _, dist = adversary
            if dist < 100:
                level += 0.5
            elif dist < 300:
                level += 0.2
        return float(min(1.0, level))

    def get_summary(self) -> dict:
        return {
            "n_jammers":    len(self._jammers),
            "n_spoofers":   len(self._spoofers),
            "n_adversaries": len(self._adversaries),
        }
