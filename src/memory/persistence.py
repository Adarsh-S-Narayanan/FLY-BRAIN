import os
import sqlite3
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional
from src.memory.working import WorkingMemory

DEFAULT_DB_PATH = os.path.join("diagnostics", "memory_store.db")

class PersistentMemoryManager:
    """
    Multi-store persistent memory framework surviving process restarts.
    Manages Working Memory, Episodic Memory, Semantic Associative Memory,
    Skill Memory, and Dream Replay Memory.
    """
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.working = WorkingMemory(capacity=7)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 1. Episodic memory table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                step INTEGER,
                observation TEXT,
                action TEXT,
                reward REAL,
                prediction_error REAL,
                outcome TEXT
            )
            """)

            # 2. Semantic associative memory table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept TEXT UNIQUE,
                description TEXT,
                embedding BLOB,
                associations TEXT
            )
            """)

            # 3. Skill memory table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_memory (
                skill_name TEXT PRIMARY KEY,
                tool_sequence TEXT,
                success_count INTEGER,
                attempt_count INTEGER,
                last_used REAL
            )
            """)

            # 4. Dream & replay memory table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS dream_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                seed INTEGER,
                base_episode_id INTEGER,
                simulated_action TEXT,
                counterfactual_reward REAL,
                insight TEXT
            )
            """)

            # 5. Experiment history table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_history (
                experiment_id TEXT PRIMARY KEY,
                timestamp REAL,
                seed INTEGER,
                config TEXT,
                results TEXT
            )
            """)
            conn.commit()

    # --- Episodic Operations ---
    def record_episode(
        self,
        step: int,
        observation: Any,
        action: str,
        reward: float,
        prediction_error: float,
        outcome: Any
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO episodic_memory (timestamp, step, observation, action, reward, prediction_error, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                step,
                json.dumps(observation) if not isinstance(observation, str) else observation,
                action,
                reward,
                prediction_error,
                json.dumps(outcome) if not isinstance(outcome, str) else outcome
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM episodic_memory ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # --- Semantic Associative Operations ---
    def store_concept(self, concept: str, description: str, embedding: np.ndarray, associations: Optional[Dict] = None):
        emb_blob = np.asarray(embedding, dtype=np.float32).tobytes()
        assoc_str = json.dumps(associations or {})
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO semantic_memory (concept, description, embedding, associations)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(concept) DO UPDATE SET
                description=excluded.description,
                embedding=excluded.embedding,
                associations=excluded.associations
            """, (concept, description, emb_blob, assoc_str))
            conn.commit()

    def query_semantic(self, query_emb: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vec = np.asarray(query_emb, dtype=np.float32).flatten()
        norm_q = np.linalg.norm(query_vec) + 1e-7
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, concept, description, embedding, associations FROM semantic_memory")
            rows = cursor.fetchall()
            
            scored = []
            for r in rows:
                emb = np.frombuffer(r["embedding"], dtype=np.float32)
                sim = float(np.dot(query_vec, emb) / (norm_q * (np.linalg.norm(emb) + 1e-7)))
                scored.append({
                    "concept": r["concept"],
                    "description": r["description"],
                    "similarity": sim,
                    "associations": json.loads(r["associations"])
                })
            scored.sort(key=lambda x: x["similarity"], reverse=True)
            return scored[:top_k]

    # --- Skill Memory Operations ---
    def record_skill(self, skill_name: str, tool_sequence: List[str], success: bool):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT success_count, attempt_count FROM skill_memory WHERE skill_name=?", (skill_name,))
            row = cursor.fetchone()
            if row:
                s_count = row[0] + (1 if success else 0)
                a_count = row[1] + 1
                cursor.execute("""
                UPDATE skill_memory SET
                    success_count=?, attempt_count=?, last_used=?, tool_sequence=?
                WHERE skill_name=?
                """, (s_count, a_count, time.time(), json.dumps(tool_sequence), skill_name))
            else:
                cursor.execute("""
                INSERT INTO skill_memory (skill_name, tool_sequence, success_count, attempt_count, last_used)
                VALUES (?, ?, ?, ?, ?)
                """, (skill_name, json.dumps(tool_sequence), 1 if success else 0, 1, time.time()))
            conn.commit()

    def get_skills(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM skill_memory ORDER BY success_count DESC")
            return [
                {
                    "skill_name": r["skill_name"],
                    "tool_sequence": json.loads(r["tool_sequence"]),
                    "success_count": r["success_count"],
                    "attempt_count": r["attempt_count"],
                    "success_rate": round(r["success_count"] / max(1, r["attempt_count"]), 3)
                }
                for r in cursor.fetchall()
            ]

    # --- Dream & Replay Operations ---
    def record_dream(
        self,
        seed: int = 42,
        base_episode_id: int = 1,
        simulated_action: str = "explore",
        counterfactual_reward: float = 0.5,
        insight: str = "",
        **kwargs
    ) -> int:
        ep_id = kwargs.get("episode_id", base_episode_id)
        reward_val = kwargs.get("hypothetical_reward", counterfactual_reward)
        insight_str = kwargs.get("consolidation_insight", insight)
        seed_val = kwargs.get("seed", seed)
        action_val = kwargs.get("simulated_action", simulated_action)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO dream_memory (timestamp, seed, base_episode_id, simulated_action, counterfactual_reward, insight)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (time.time(), seed_val, ep_id, action_val, reward_val, insight_str))
            conn.commit()
            return cursor.lastrowid

    def get_recent_dreams(self, limit: int = 10) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM dream_memory ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    # --- Experiment History Operations ---
    def record_experiment(self, experiment_id: str, seed: int, config: Dict, results: Dict):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO experiment_history (experiment_id, timestamp, seed, config, results)
            VALUES (?, ?, ?, ?, ?)
            """, (experiment_id, time.time(), seed, json.dumps(config), json.dumps(results)))
            conn.commit()
