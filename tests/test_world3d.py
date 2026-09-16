"""Unit tests for 3D World continuous environment and RL Fly Trainer."""
import unittest
import numpy as np
from src.world.world3d import World3D, World3DConfig, FlyState3D
from src.trainer.fly_3d_trainer import Fly3DTrainer, extract_state_vector


class TestWorld3D(unittest.TestCase):
    def test_world3d_initialization(self):
        world = World3D(World3DConfig(n_trees=4, fruits_per_tree=5, seed=42))
        self.assertEqual(len(world.trees), 4)
        self.assertEqual(len(world.fruits), 20)
        self.assertEqual(world.fly.position[1], -10.0)
        self.assertEqual(world.fly.position[2], 1.5)

    def test_odor_sampling_and_antennas(self):
        world = World3D()
        left_pos, right_pos = world.fly.get_antenna_positions()
        self.assertEqual(left_pos.shape, (3,))
        self.assertEqual(right_pos.shape, (3,))

        # Odor concentration near fruit tree should be positive
        fruit_pos = list(world.fruits.values())[0].position
        odor_at_fruit = world.sample_odor_concentration(fruit_pos)
        self.assertGreater(odor_at_fruit, 0.5)

    def test_sensory_input(self):
        world = World3D()
        sensory = world.get_sensory_input()
        self.assertIn("left_odor", sensory)
        self.assertIn("right_odor", sensory)
        self.assertIn("nearest_fruit_dist", sensory)
        self.assertGreater(sensory["nearest_fruit_dist"], 0.0)

    def test_step_physics_and_rewards(self):
        world = World3D()
        action = np.array([0.8, 0.1, -0.2, 0.0], dtype=np.float32)
        sensory, reward, done, info = world.step(action)
        self.assertIsNotNone(sensory)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("harvested", info)

    def test_rl_trainer_step(self):
        world = World3D()
        trainer = Fly3DTrainer(world=world)
        stats = trainer.train_step(max_ep_steps=50)
        self.assertEqual(stats["episode"], 1)
        self.assertIn("reward", stats)
        self.assertGreater(stats["steps"], 0)
        self.assertEqual(len(trainer.reward_history), 1)


if __name__ == "__main__":
    unittest.main()
