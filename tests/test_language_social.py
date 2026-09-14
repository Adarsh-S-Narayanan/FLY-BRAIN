"""STAGE G/H tests: grounded language, emergent trust, social memory,
information transfer measured by receiver behavior change."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np

from src.language.grounded import GroundedLanguageSystem
from src.social.model import SocialMemory, TRUST_INIT


class TestGrounding(unittest.TestCase):
    def test_concept_requires_features(self):
        ls = GroundedLanguageSystem(seed=1)
        c = ls.ground_concept("food", [0.8, 0.1, 0.3], "sensory",
                              {"food_gradient": 0.8})
        self.assertEqual(len(c.feature_vector), ls.feature_dim)
        with self.assertRaises(ValueError):
            ls.ground_concept("bad", [float("nan")], "sensory")
        with self.assertRaises(ValueError):
            ls.ground_concept("bad", [0.1], "spiritual")

    def test_symbol_binds_only_to_grounded_concept(self):
        ls = GroundedLanguageSystem(seed=2)
        with self.assertRaises(ValueError):
            ls.learn_symbol("kel", "nonexistent")
        c = ls.ground_concept("hazard", [0.9, 0.0, 0.2], "sensory")
        self.assertTrue(ls.learn_symbol("kel", c.id if hasattr(c, 'id') else c.concept_id))

    def test_production_from_internal_state_only(self):
        ls = GroundedLanguageSystem(seed=3)
        food = ls.ground_concept("food", [0.9, 0, 0, 0, 0, 0, 0, 0], "sensory")
        danger = ls.ground_concept("danger", [0.0, 0.9, 0, 0, 0, 0, 0, 0], "sensory")
        ls.learn_symbol("mibu", food.concept_id)
        ls.learn_symbol("kara", danger.concept_id)
        # salient concepts drive production
        syms = ls.produce({"salient_concepts": [food.concept_id, danger.concept_id]})
        self.assertEqual(syms, ["mibu", "kara"])
        # nothing salient -> nothing produced (never invented text)
        self.assertEqual(ls.produce({"salient_concepts": []}), [])
        # unknown concept id -> skipped, not fabricated
        self.assertEqual(ls.produce({"salient_concepts": ["nope"]}), [])

    def test_comprehension_roundtrip(self):
        ls = GroundedLanguageSystem(seed=4)
        c = ls.ground_concept("rest", [0, 0.5, 0, 0, 0, 0, 0, 0], "internal")
        ls.learn_symbol("zuzu", c.concept_id)
        self.assertEqual(ls.comprehend(["zuzu"]), [c.concept_id])
        self.assertEqual(ls.comprehend(["unknown-word"]), [])

    def test_grounding_strength_grows_with_use(self):
        ls = GroundedLanguageSystem(seed=5)
        c = ls.ground_concept("water", [0.1, 0.9, 0, 0, 0, 0, 0, 0], "sensory")
        ls.learn_symbol("wawa", c.concept_id)
        s0 = ls.grounding_strength("wawa")
        for _ in range(10):
            ls.comprehend(["wawa"])
        self.assertGreater(ls.grounding_strength("wawa"), s0)
        self.assertEqual(ls.grounding_strength("never-seen"), 0.0)
        self.assertEqual(ls.ungrounded_symbols(), [])

    def test_language_persists_across_snapshot(self):
        ls = GroundedLanguageSystem(seed=6)
        c = ls.ground_concept("kin", [0.2, 0.2, 0.9, 0, 0, 0, 0, 0], "social")
        ls.learn_symbol("tribe", c.concept_id)
        ls.produce({"salient_concepts": [c.concept_id]})
        snap = ls.snapshot()
        ls2 = GroundedLanguageSystem.restore(snap)
        self.assertEqual(ls2.comprehend(["tribe"]), [c.concept_id])
        self.assertEqual(ls2.vocabulary_size(), ls.vocabulary_size())


class TestSocialTrust(unittest.TestCase):
    def test_no_hardcoded_friendship(self):
        mem = SocialMemory("org-a")
        self.assertFalse(mem.knows("org-b"))
        self.assertEqual(mem.trust_of("org-b"), TRUST_INIT)
        self.assertEqual(mem.relationship_strength("org-b"), 0.0)

    def test_trust_emerges_from_outcomes(self):
        mem = SocialMemory("org-a")
        # positive teaching outcomes raise trust (delta rule -> converges to outcome)
        for t in range(10):
            mem.record_interaction("teacher-1", t, "taught_by", outcome=+0.8)
        # negative competitive outcomes lower trust
        for t in range(10):
            mem.record_interaction("rival-1", t, "competed", outcome=-0.8)
        self.assertGreater(mem.trust_of("teacher-1"), 0.6)
        self.assertLess(mem.trust_of("rival-1"), 0.1)
        self.assertEqual(mem.top_partners(1)[0], "teacher-1")

    def test_relationship_persistence_and_strength(self):
        mem = SocialMemory("org-a")
        for t in range(12):
            mem.record_interaction("ally", t, "cooperated", outcome=0.6)
        self.assertGreaterEqual(mem.relationship_strength("ally"), 0.5)
        snap = mem.snapshot()
        mem2 = SocialMemory.restore(snap)
        self.assertEqual(mem2.trust_of("ally"), mem.trust_of("ally"))
        self.assertEqual(mem2.knows("ally"), True)
        self.assertEqual(mem2.summary()["total_interactions"], 12)

    def test_outcome_bounds_and_self_rejection(self):
        mem = SocialMemory("org-a")
        rec = mem.record_interaction("x", 1, "communicated", outcome=99.0)
        self.assertEqual(rec.history[0].outcome, 1.0)
        with self.assertRaises(ValueError):
            mem.record_interaction("org-a", 2, "cooperated", 0.5)
        with self.assertRaises(ValueError):
            mem.record_interaction("", 3, "cooperated", 0.5)


class TestInformationTransfer(unittest.TestCase):
    def test_message_changes_receiver_knowledge(self):
        """Measured transfer: receiver gains a grounded concept it can use."""
        sender = GroundedLanguageSystem(seed=11)
        receiver = GroundedLanguageSystem(seed=12)
        c = sender.ground_concept("food-north", [0.9, 0, 0, 0, 0, 0, 0, 0], "sensory",
                                  {"direction": "north", "food_gradient": 0.9})
        sender.learn_symbol("nolo", c.concept_id)
        msg = sender.produce({"salient_concepts": [c.concept_id]})
        self.assertEqual(msg, ["nolo"])
        # receiver has no grounding for the symbol -> transfer fails honestly
        self.assertEqual(receiver.comprehend(msg), [])
        # after TEACHING the grounding (social learning), transfer succeeds
        rc = receiver.ground_concept("food-north", [0.9, 0, 0, 0, 0, 0, 0, 0], "sensory",
                                     {"taught_by": "sender", "tick": 1})
        receiver.learn_symbol("nolo", rc.concept_id, teacher="sender")
        self.assertEqual(receiver.comprehend(msg), [rc.concept_id])
        self.assertGreater(receiver.grounding_strength("nolo"), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
