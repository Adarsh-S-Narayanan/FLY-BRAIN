"""3D Continuous Simulation Environment for Drosophila Foraging.

Features:
- 3D continuous space (x, y, z) with ground terrain and trees.
- Trees with trunks, branch clusters, foliage canopies, and hanging fruits (apples, oranges).
- 3D Olfactory odor plume model (Gaussian diffusion from fruits + wind vector).
- Aerodynamic fly flight physics (position, velocity, pitch, yaw, roll, thrust, lift, drag).
- Fruit harvesting, ripening dynamics, collision detection.
"""
import math
import hashlib
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class Fruit3D:
    id: str
    fruit_type: str            # "apple", "orange", "berry"
    position: np.ndarray       # shape (3,) [x, y, z]
    radius: float = 0.35
    value: float = 1.0         # Energy / reward value
    ripe: bool = True
    odor_intensity: float = 1.0
    tree_id: str = "tree_0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.fruit_type,
            "position": [round(float(c), 3) for c in self.position],
            "radius": round(self.radius, 3),
            "value": round(self.value, 3),
            "ripe": self.ripe,
            "odor_intensity": round(self.odor_intensity, 3),
            "tree_id": self.tree_id,
        }


@dataclass
class Tree3D:
    id: str
    position: np.ndarray       # Base position (x, y, 0)
    trunk_height: float = 4.0
    trunk_radius: float = 0.6
    canopy_radius: float = 3.5
    canopy_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fruit_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "position": [round(float(c), 3) for c in self.position],
            "trunk_height": round(self.trunk_height, 3),
            "trunk_radius": round(self.trunk_radius, 3),
            "canopy_radius": round(self.canopy_radius, 3),
            "canopy_center": [round(float(c), 3) for c in self.canopy_center],
            "fruit_ids": self.fruit_ids,
        }


@dataclass
class FlyState3D:
    id: str = "fly_0"
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, -10.0, 1.5]))  # x, y, z
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.5, 0.0]))   # vx, vy, vz
    heading: float = 0.5 * math.pi      # yaw angle (radians, 0 = +X, pi/2 = +Y)
    pitch: float = 0.0                  # pitch angle (radians, positive = look up)
    roll: float = 0.0                   # roll angle
    antenna_span: float = 0.15          # distance between left and right antennae
    energy: float = 1.0
    health: float = 1.0
    wing_phase: float = 0.0
    fruits_eaten: int = 0

    def get_antenna_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns 3D coordinates of (left_antenna, right_antenna)."""
        # Forward vector
        fwd = np.array([
            math.cos(self.pitch) * math.cos(self.heading),
            math.cos(self.pitch) * math.sin(self.heading),
            math.sin(self.pitch)
        ])
        # Right vector
        right = np.array([
            math.sin(self.heading),
            -math.cos(self.heading),
            0.0
        ])
        half_span = self.antenna_span / 2.0
        left_pos = self.position + fwd * 0.1 - right * half_span
        right_pos = self.position + fwd * 0.1 + right * half_span
        return left_pos, right_pos

    def to_dict(self) -> Dict[str, Any]:
        l_ant, r_ant = self.get_antenna_positions()
        return {
            "id": self.id,
            "position": [round(float(c), 3) for c in self.position],
            "velocity": [round(float(c), 3) for c in self.velocity],
            "heading": round(float(self.heading), 4),
            "pitch": round(float(self.pitch), 4),
            "roll": round(float(self.roll), 4),
            "energy": round(float(self.energy), 4),
            "health": round(float(self.health), 4),
            "fruits_eaten": self.fruits_eaten,
            "left_antenna": [round(float(c), 3) for c in l_ant],
            "right_antenna": [round(float(c), 3) for c in r_ant],
        }


@dataclass
class World3DConfig:
    bounds: Tuple[float, float, float] = (30.0, 30.0, 15.0)  # half-width X, half-width Y, max Z
    n_trees: int = 4
    fruits_per_tree: int = 5
    wind_vector: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.1, 0.0]))
    odor_diffusion_sigma: float = 4.0
    dt: float = 0.1
    seed: int = 42


class World3D:
    def __init__(self, config: Optional[World3DConfig] = None):
        self.config = config or World3DConfig()
        self.tick = 0
        self.trees: Dict[str, Tree3D] = {}
        self.fruits: Dict[str, Fruit3D] = {}
        self.fly = FlyState3D()
        self.energy_consumed = 0.0
        self.reset(self.config.seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.config.seed = seed
        rng = np.random.RandomState(int(self.config.seed) % (2 ** 31))
        self.tick = 0
        self.trees.clear()
        self.fruits.clear()
        self.energy_consumed = 0.0

        # Generate Trees around center
        tree_locations = [
            (0.0, 4.0),     # Main central tree ahead of fly
            (-7.0, 6.0),    # Left tree
            (7.0, 8.0),     # Right tree
            (-2.0, 14.0),   # Back tree
        ]

        for i in range(min(self.config.n_trees, len(tree_locations))):
            tx, ty = tree_locations[i]
            t_id = f"tree_{i}"
            trunk_h = float(rng.uniform(3.5, 5.0))
            canopy_r = float(rng.uniform(3.0, 4.2))
            canopy_c = np.array([tx, ty, trunk_h + canopy_r * 0.6])

            tree = Tree3D(
                id=t_id,
                position=np.array([tx, ty, 0.0]),
                trunk_height=trunk_h,
                trunk_radius=0.6,
                canopy_radius=canopy_r,
                canopy_center=canopy_c,
                fruit_ids=[]
            )

            # Spawn Fruits hanging inside/around canopy
            for j in range(self.config.fruits_per_tree):
                f_id = f"fruit_{i}_{j}"
                # Spherical offset within canopy
                phi = float(rng.uniform(0, 2 * math.pi))
                theta = float(rng.uniform(0.2 * math.pi, 0.8 * math.pi))
                r = float(rng.uniform(0.5, canopy_r * 0.85))

                fx = canopy_c[0] + r * math.sin(theta) * math.cos(phi)
                fy = canopy_c[1] + r * math.sin(theta) * math.sin(phi)
                fz = canopy_c[2] + r * math.cos(theta)

                f_type = "apple" if (i + j) % 2 == 0 else "orange"
                fruit = Fruit3D(
                    id=f_id,
                    fruit_type=f_type,
                    position=np.array([fx, fy, fz]),
                    radius=0.35 if f_type == "apple" else 0.4,
                    value=1.0 if f_type == "apple" else 1.2,
                    ripe=True,
                    odor_intensity=1.0,
                    tree_id=t_id
                )
                self.fruits[f_id] = fruit
                tree.fruit_ids.append(f_id)

            self.trees[t_id] = tree

        # Reset Fly position at starting approach point
        self.fly = FlyState3D(
            id="fly_0",
            position=np.array([0.0, -10.0, 1.5]),
            velocity=np.array([0.0, 0.2, 0.0]),
            heading=0.5 * math.pi, # facing towards +Y (where trees are)
            pitch=0.0,
            roll=0.0,
            energy=1.0,
            health=1.0,
            fruits_eaten=0
        )

    def sample_odor_concentration(self, point: np.ndarray) -> float:
        """Calculates 3D odor concentration at a point (Gaussian plume + wind drift)."""
        total_odor = 0.0
        sigma = self.config.odor_diffusion_sigma
        wind = self.config.wind_vector

        for fruit in self.fruits.values():
            if not fruit.ripe:
                continue
            # Plume center shifts with wind based on distance from fruit source
            diff = point - fruit.position
            # Effective distance accounting for wind advection
            effective_diff = diff - wind * 0.5
            dist_sq = float(np.sum(effective_diff ** 2))
            conc = fruit.odor_intensity * math.exp(-dist_sq / (2.0 * sigma ** 2))
            total_odor += conc

        return total_odor

    def get_sensory_input(self) -> Dict[str, Any]:
        """Calculates sensory perception for the fly (olfactory stereo + visual target vector)."""
        left_pos, right_pos = self.fly.get_antenna_positions()
        left_odor = self.sample_odor_concentration(left_pos)
        right_odor = self.sample_odor_concentration(right_pos)
        center_odor = self.sample_odor_concentration(self.fly.position)

        # Find nearest ripe fruit
        nearest_fruit: Optional[Fruit3D] = None
        min_dist = float("inf")
        for fruit in self.fruits.values():
            if not fruit.ripe:
                continue
            d = float(np.linalg.norm(self.fly.position - fruit.position))
            if d < min_dist:
                min_dist = d
                nearest_fruit = fruit

        if nearest_fruit is not None:
            vec_to_fruit = nearest_fruit.position - self.fly.position
            dist_to_fruit = min_dist
            dir_to_fruit = vec_to_fruit / max(1e-6, dist_to_fruit)
        else:
            vec_to_fruit = np.zeros(3)
            dist_to_fruit = 100.0
            dir_to_fruit = np.zeros(3)

        # Body frame directional angles to nearest fruit
        fwd = np.array([
            math.cos(self.fly.pitch) * math.cos(self.fly.heading),
            math.cos(self.fly.pitch) * math.sin(self.fly.heading),
            math.sin(self.fly.pitch)
        ])
        up = np.array([0.0, 0.0, 1.0])

        dot_fwd = float(np.dot(dir_to_fruit, fwd))
        bearing_error = math.acos(max(-1.0, min(1.0, dot_fwd)))

        # Signed yaw bearing error
        cross_z = fwd[0] * dir_to_fruit[1] - fwd[1] * dir_to_fruit[0]
        signed_bearing = bearing_error if cross_z >= 0 else -bearing_error

        # Elevation error to fruit
        elevation_error = math.atan2(dir_to_fruit[2], max(1e-3, np.linalg.norm(dir_to_fruit[:2]))) - self.fly.pitch

        return {
            "left_odor": round(left_odor, 4),
            "right_odor": round(right_odor, 4),
            "center_odor": round(center_odor, 4),
            "odor_gradient": round(left_odor - right_odor, 4),
            "nearest_fruit_dist": round(dist_to_fruit, 3),
            "nearest_fruit_dir": [round(float(c), 3) for c in dir_to_fruit],
            "bearing_error": round(signed_bearing, 4),
            "elevation_error": round(elevation_error, 4),
            "fly_altitude": round(float(self.fly.position[2]), 3),
            "fly_speed": round(float(np.linalg.norm(self.fly.velocity)), 3),
        }

    def apply_action(self, action: np.ndarray):
        """Applies 3D continuous flight control actions:
        action = [thrust, pitch_rate, yaw_rate, roll_rate] in range [-1.0, 1.0].
        """
        thrust = float(np.clip(action[0], 0.0, 1.0)) * 6.0      # Forward acceleration
        pitch_rate = float(np.clip(action[1], -1.0, 1.0)) * 1.5 # rad/s
        yaw_rate = float(np.clip(action[2], -1.0, 1.0)) * 2.0   # rad/s
        roll_rate = float(np.clip(action[3], -1.0, 1.0)) * 1.5  # rad/s

        dt = self.config.dt

        # Update angles
        self.fly.heading = (self.fly.heading + yaw_rate * dt) % (2 * math.pi)
        self.fly.pitch = float(np.clip(self.fly.pitch + pitch_rate * dt, -1.2, 1.2))
        self.fly.roll = float(np.clip(self.fly.roll + roll_rate * dt, -1.0, 1.0))

        # Forward vector
        fwd = np.array([
            math.cos(self.fly.pitch) * math.cos(self.fly.heading),
            math.cos(self.fly.pitch) * math.sin(self.fly.heading),
            math.sin(self.fly.pitch)
        ])

        # Aerodynamics: Thrust + Lift - Drag - Gravity
        gravity = np.array([0.0, 0.0, -1.8]) # scaled fly gravity
        lift = np.array([0.0, 0.0, 1.8 * (0.5 + 0.5 * action[0])]) # Lift tied to thrust
        drag = -0.4 * self.fly.velocity

        accel = fwd * thrust + drag + gravity + lift
        self.fly.velocity += accel * dt
        # Clamp velocity
        speed = np.linalg.norm(self.fly.velocity)
        max_speed = 5.0
        if speed > max_speed:
            self.fly.velocity = (self.fly.velocity / speed) * max_speed

        # Update position
        self.fly.position += self.fly.velocity * dt

        # Boundary checks
        bx, by, bz = self.config.bounds
        self.fly.position[0] = float(np.clip(self.fly.position[0], -bx, bx))
        self.fly.position[1] = float(np.clip(self.fly.position[1], -by, by))
        # Ground floor Z=0.2, ceiling Z=bz
        if self.fly.position[2] < 0.2:
            self.fly.position[2] = 0.2
            self.fly.velocity[2] = max(0.0, self.fly.velocity[2]) # bounce off ground
        elif self.fly.position[2] > bz:
            self.fly.position[2] = bz
            self.fly.velocity[2] = min(0.0, self.fly.velocity[2])

        # Energy consumption
        cost = 0.002 * (1.0 + thrust * 0.5)
        self.fly.energy = max(0.0, self.fly.energy - cost)
        self.fly.wing_phase = (self.fly.wing_phase + 0.8) % (2 * math.pi)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Steps 3D simulation by 1 tick. Returns (obs, reward, done, info)."""
        self.tick += 1
        prev_sensory = self.get_sensory_input()
        prev_dist = prev_sensory["nearest_fruit_dist"]

        self.apply_action(action)

        sensory = self.get_sensory_input()
        curr_dist = sensory["nearest_fruit_dist"]

        # Calculate Reward
        reward = 0.0
        done = False
        info = {"harvested": False, "collision": False}

        # 1. Distance progress reward
        dist_delta = prev_dist - curr_dist
        reward += dist_delta * 2.0

        # 2. Odor concentration reward
        reward += sensory["center_odor"] * 0.5

        # 3. Fruit Harvest Check
        harvest_radius = 1.2
        for f_id, fruit in list(self.fruits.items()):
            if fruit.ripe:
                d = float(np.linalg.norm(self.fly.position - fruit.position))
                if d <= harvest_radius:
                    fruit.ripe = False
                    reward += 15.0
                    self.fly.fruits_eaten += 1
                    self.fly.energy = min(1.5, self.fly.energy + fruit.value * 0.5)
                    self.energy_consumed += fruit.value
                    info["harvested"] = True
                    info["fruit_id"] = f_id
                    break

        # 4. Trunk Collision Penalty
        for tree in self.trees.values():
            # Distance to trunk cylinder (Z between 0 and trunk_height)
            if 0.0 <= self.fly.position[2] <= tree.trunk_height:
                dist_2d = float(np.linalg.norm(self.fly.position[:2] - tree.position[:2]))
                if dist_2d < tree.trunk_radius:
                    reward -= 5.0
                    info["collision"] = True
                    # Push out of trunk
                    push_dir = self.fly.position[:2] - tree.position[:2]
                    norm = np.linalg.norm(push_dir)
                    if norm > 1e-3:
                        push_dir /= norm
                    else:
                        push_dir = np.array([1.0, 0.0])
                    self.fly.position[:2] = tree.position[:2] + push_dir * (tree.trunk_radius + 0.1)

        # Check termination (energy depleted or all fruits eaten or max steps)
        if self.fly.energy <= 0.0 or self.tick >= 1000:
            done = True
        elif all(not f.ripe for f in self.fruits.values()):
            done = True
            reward += 20.0 # Clear bonus

        return sensory, reward, done, info

    def snapshot(self) -> Dict[str, Any]:
        """Serializes current 3D world state for REST/WebSocket UI."""
        return {
            "tick": self.tick,
            "fly": self.fly.to_dict(),
            "trees": [t.to_dict() for t in self.trees.values()],
            "fruits": [f.to_dict() for f in self.fruits.values()],
            "wind": [round(float(c), 3) for c in self.config.wind_vector],
            "sensory": self.get_sensory_input(),
            "energy_consumed": round(self.energy_consumed, 3),
        }
