import sys
import numpy as np
from ursina import *

# Ensure we can import from src
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.connectome.loader import get_or_create_circuit
from src.connectome.types import GraphMode
from src.brain.runtime import BrainRuntime

def create_flight_simulation():
    app = Ursina()
    
    # Setup environment
    window.title = 'FlyBrain 3D Flight Simulation'
    window.borderless = False
    window.fullscreen = False
    
    # Create the world
    sky = Sky()
    ground = Entity(model='plane', scale=(100, 1, 100), color=color.lime, collider='box')
    
    # Create the fly body
    fly_body = Entity(
        model='cube', 
        color=color.black, 
        scale=(0.5, 0.2, 0.5),
        position=(0, 2, 0)
    )
    # Add simple wings for visual effect
    left_wing = Entity(parent=fly_body, model='cube', scale=(1.2, 0.05, 0.4), position=(-0.6, 0.1, 0), color=color.white50)
    right_wing = Entity(parent=fly_body, model='cube', scale=(1.2, 0.05, 0.4), position=(0.6, 0.1, 0), color=color.white50)
    
    # Target to fly towards
    target = Entity(model='sphere', color=color.red, scale=(1, 1, 1), position=(5, 5, 5))
    
    # Set camera to follow the fly
    camera.parent = fly_body
    camera.position = (0, 3, -10)
    camera.look_at(fly_body)
    
    # --- Connectome Setup ---
    print("Initializing FlyBrain Connectome...")
    # Using a smaller synthetic graph for fast real-time CPU execution
    graph = get_or_create_circuit(max_neurons=256, mode=GraphMode.SYNTHETIC_TEST)
    brain = BrainRuntime(graph, use_gpu=False, enable_plasticity=True)
    
    # Get motor output indices (using arbitrary indices if specific ones aren't mapped)
    motor_thrust = brain.motor_act_indices[0] if len(brain.motor_act_indices) > 0 else 0
    motor_yaw = brain.motor_act_indices[1] if len(brain.motor_act_indices) > 1 else 1
    motor_pitch = brain.motor_act_indices[2] if len(brain.motor_act_indices) > 2 else 2
    
    def update():
        # 1. Sensory Input: Raycast/Distance to target
        dist_to_target = distance(fly_body.position, target.position)
        
        # Feed sensory data into visual cortex
        vis_stimulus = np.zeros(16, dtype=np.float32)
        vis_stimulus[0] = max(0.0, 1.0 - (dist_to_target / 50.0))  # Proximity
        
        # Step the biological brain
        brain.step(sensory_inputs={"visual": vis_stimulus}, reward=0.0)
        
        # 2. Read Motor Outputs (firing rates)
        thrust_activation = brain.state.activations[motor_thrust]
        yaw_activation = brain.state.activations[motor_yaw]
        pitch_activation = brain.state.activations[motor_pitch]
        
        # Map neural output to aerodynamics
        forward_thrust = 2.0 + (thrust_activation * 15.0)
        yaw_steer = (yaw_activation - 0.5) * 150.0 * time.dt
        pitch_steer = (pitch_activation - 0.5) * 100.0 * time.dt
        
        # Flap wings based on thrust
        left_wing.rotation_z = np.sin(time.time() * 20 * forward_thrust) * 20
        right_wing.rotation_z = -np.sin(time.time() * 20 * forward_thrust) * 20
        
        # 3. Apply physical forces
        fly_body.rotation_y += yaw_steer
        fly_body.rotation_x += pitch_steer
        
        fly_body.position += fly_body.forward * forward_thrust * time.dt
        
        # Simple gravity / ground collision
        if fly_body.y < 0.2:
            fly_body.y = 0.2
            
        # Foraging reward mechanic
        if dist_to_target < 2.0:
            target.position = (
                np.random.uniform(-20, 20),
                np.random.uniform(2, 10),
                np.random.uniform(-20, 20)
            )
            # Potentiate synapses (Reward!)
            brain.step(sensory_inputs={"visual": np.ones(16, dtype=np.float32)}, reward=1.0)
    
    app.update = update
    app.run()

if __name__ == '__main__':
    create_flight_simulation()
