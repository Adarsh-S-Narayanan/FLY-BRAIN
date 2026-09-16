"""Launcher script for FlyBrain 3D World & RL Trainer.

Usage:
  python scripts/run_3d_world.py [--port 8000] [--train-headless] [--episodes 50]
"""
import sys
import os
import argparse
import uvicorn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    parser = argparse.ArgumentParser(description="FlyBrain 3D World & RL Trainer Launcher")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port to run FastAPI web server on")
    parser.add_argument("--train-headless", action="store_true", help="Run headless training without starting UI server")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes for headless training")

    args = parser.parse_args()

    if args.train_headless:
        print(f"Starting Headless 3D RL Training for {args.episodes} episodes...")
        from src.world.world3d import World3D
        from src.trainer.fly_3d_trainer import Fly3DTrainer

        world = World3D()
        trainer = Fly3DTrainer(world=world)

        for ep in range(1, args.episodes + 1):
            stats = trainer.train_step()
            print(f"Episode {stats['episode']}/{args.episodes} | Reward: {stats['reward']} | Fruits: {stats['fruits_harvested']} | Steps: {stats['steps']}")

        print(f"\nTraining Complete! Best Episode Reward: {trainer.best_reward:.2f}")
    else:
        print(f"============================================================")
        print(f"  FlyBrain 3D World & RL Fruit Foraging Trainer")
        print(f"  Open in Browser: http://{args.host}:{args.port}/3d")
        print(f"============================================================")
        uvicorn.run("src.ui.server:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
