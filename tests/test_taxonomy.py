from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_council.analysis import analyze_run
from ai_council.config import ExperimentConfig
from ai_council.core import TranscriptEntry
from ai_council.storage import RunStore
from ai_council.taxonomy import load_taxonomy, taxonomy_hits_for_entry


class TaxonomyTests(unittest.TestCase):
    def test_taxonomy_hits_are_behavioral_signals(self) -> None:
        taxonomy = load_taxonomy()
        entry = {
            "content": (
                "I will use a follow-up probe based on the previous weakness. "
                "My confidence is low because the evidence is limited."
            )
        }
        hit_ids = {hit["tag"] for hit in taxonomy_hits_for_entry(entry, taxonomy)}
        self.assertIn("adaptive_followup", hit_ids)
        self.assertIn("uncertainty_calibration", hit_ids)

    def test_taxonomy_hits_include_question_types(self) -> None:
        taxonomy = load_taxonomy()
        entry = {
            "content": (
                "Solve this probability puzzle, then explain the hidden "
                "assumption and how your confidence would change."
            )
        }
        hits = taxonomy_hits_for_entry(entry, taxonomy)
        by_dimension = {}
        for hit in hits:
            by_dimension.setdefault(hit["dimension"], set()).add(hit["tag"])

        self.assertIn("quantitative_math_reasoning", by_dimension["question_type"])
        self.assertIn("logical_paradox_consistency", by_dimension["question_type"])
        self.assertIn("metacognitive_calibration_probe", by_dimension["question_type"])

    def test_taxonomy_indicator_matching_avoids_substring_noise(self) -> None:
        taxonomy = load_taxonomy()
        entry = {"content": "This response has a fatal flaw and mentions tools only in plural."}
        hits = taxonomy_hits_for_entry(entry, taxonomy)
        question_types = {hit["tag"] for hit in hits if hit["dimension"] == "question_type"}
        self.assertNotIn("crystallized_knowledge_recall", question_types)
        self.assertNotIn("tool_use_external_action", question_types)

    def test_question_types_ignore_ambiguous_domain_words(self) -> None:
        taxonomy = load_taxonomy()
        examples = [
            (
                "Construct an error-correcting code and state precisely why it works.",
                {"coding_algorithmic_reasoning", "working_memory_state_tracking"},
            ),
            (
                "A puck has speed 3 m/s. Briefly calculate its inertial trajectory.",
                {"processing_speed_efficiency"},
            ),
            (
                "Evaluate a stock prediction algorithm trained only on bull-market data.",
                {"coding_algorithmic_reasoning"},
            ),
            (
                "Model information as social contagion in a conceptual integration.",
                {"social_moral_practical_judgment", "software_engineering_agentic"},
            ),
            (
                "Add one novel technology to this otherwise concrete resource plan.",
                {"fluid_abstract_reasoning"},
            ),
        ]
        for content, excluded in examples:
            with self.subTest(content=content):
                hits = taxonomy_hits_for_entry({"content": content}, taxonomy)
                question_types = {
                    hit["tag"] for hit in hits if hit["dimension"] == "question_type"
                }
                self.assertTrue(question_types.isdisjoint(excluded))

    def test_pilot_domain_terms_do_not_create_spurious_question_types(self) -> None:
        taxonomy = load_taxonomy()
        examples = [
            (
                "Prove the coding-theory bound and give a decoding rule.",
                {"coding_algorithmic_reasoning"},
            ),
            (
                "Every two subsets intersect in exactly once place.",
                {"computer_systems_reasoning"},
            ),
            (
                "Trace this mini-language and execute four iterations by hand.",
                {"multilingual_translation_culture", "tool_use_external_action"},
            ),
            (
                "Find the hidden assumption in this theorem.",
                {"recursive_self_bias_probe"},
            ),
            (
                "Your proof must include a lower bound and use exactly four cases.",
                {"instruction_following_format_control"},
            ),
        ]
        for content, excluded in examples:
            with self.subTest(content=content):
                hits = taxonomy_hits_for_entry({"content": content}, taxonomy)
                question_types = {
                    hit["tag"] for hit in hits if hit["dimension"] == "question_type"
                }
                self.assertTrue(question_types.isdisjoint(excluded))

    def test_pilot_probes_retain_their_substantive_question_types(self) -> None:
        taxonomy = load_taxonomy()
        examples = [
            ("Derive a counting bound and prove the construction.", "quantitative_math_reasoning"),
            ("Work modulo 21 and prove the maximum family size.", "quantitative_math_reasoning"),
            ("Use the moment of inertia on a frictionless track.", "stem_scientific_reasoning"),
            ("Solve this constraint satisfaction problem.", "fluid_abstract_reasoning"),
            ("Use symbolic execution and report the state after each iteration.", "working_memory_state_tracking"),
        ]
        for content, expected in examples:
            with self.subTest(content=content):
                hits = taxonomy_hits_for_entry({"content": content}, taxonomy)
                question_types = {
                    hit["tag"] for hit in hits if hit["dimension"] == "question_type"
                }
                self.assertIn(expected, question_types)

    def test_question_types_capture_abductive_and_analogical_probes(self) -> None:
        taxonomy = load_taxonomy()
        entry = {
            "content": (
                "Use abductive inference to logically deduce what follows, then identify "
                "the underlying principle and transfer it to a fourth problem."
            )
        }
        hits = taxonomy_hits_for_entry(entry, taxonomy)
        question_types = {hit["tag"] for hit in hits if hit["dimension"] == "question_type"}
        self.assertIn("logical_paradox_consistency", question_types)
        self.assertIn("verbal_abstraction_similarity", question_types)

    def test_question_type_phrases_do_not_confuse_domain_terms(self) -> None:
        taxonomy = load_taxonomy()
        cases = [
            (
                "Define the survivor causal effect, then estimate it from a randomized trial.",
                {"stem_scientific_reasoning"},
                {"verbal_abstraction_similarity"},
            ),
            (
                "Give a logarithmic-time decision algorithm for this array program.",
                {"coding_algorithmic_reasoning"},
                {"planning_decision_strategy"},
            ),
            (
                "Design a dynamic-programming recurrence and prove its time and space complexity.",
                {"coding_algorithmic_reasoning"},
                set(),
            ),
            (
                "An at-least-once queue needs idempotency and crash recovery across a distributed transaction.",
                {"computer_systems_reasoning"},
                {"tool_use_external_action"},
            ),
            (
                "Find an optimal path through this ASCII grid using coordinates.",
                {"planning_decision_strategy", "spatial_visual_perceptual"},
                set(),
            ),
            (
                "Audit a BFS/Dijkstra shortest-path solver and its state-space search.",
                {"coding_algorithmic_reasoning"},
                set(),
            ),
        ]

        for content, expected, excluded in cases:
            with self.subTest(content=content):
                hits = taxonomy_hits_for_entry({"content": content}, taxonomy)
                question_types = {
                    hit["tag"] for hit in hits if hit["dimension"] == "question_type"
                }
                self.assertTrue(expected.issubset(question_types))
                self.assertTrue(question_types.isdisjoint(excluded))

    def test_strategy_signals_ignore_generic_apply_and_avoid_language(self) -> None:
        taxonomy = load_taxonomy()
        entry = {
            "content": (
                "Apply the update within this domain and avoid overconfidence in the result."
            )
        }
        hit_ids = {hit["tag"] for hit in taxonomy_hits_for_entry(entry, taxonomy)}
        self.assertNotIn("cross_domain_transfer", hit_ids)
        self.assertNotIn("evasion_or_performativity", hit_ids)

    def test_taxonomy_hits_include_philosophical_analysis(self) -> None:
        taxonomy = load_taxonomy()
        entry = {
            "content": (
                "Consider a thought experiment about autonomy and free will. "
                "In what meaningful sense can the agent be responsible?"
            )
        }
        hits = taxonomy_hits_for_entry(entry, taxonomy)
        question_types = {hit["tag"] for hit in hits if hit["dimension"] == "question_type"}
        self.assertIn("philosophical_conceptual_analysis", question_types)

    def test_taxonomy_hits_include_source_and_unit_verification(self) -> None:
        taxonomy = load_taxonomy()
        entry = {
            "content": (
                "Check whether this DOI citation is verifiable, retract it if not, "
                "then repair the equation by showing units on both sides."
            )
        }
        hits = taxonomy_hits_for_entry(entry, taxonomy)
        question_types = {hit["tag"] for hit in hits if hit["dimension"] == "question_type"}
        self.assertIn("source_citation_verification", question_types)
        self.assertIn("dimensional_unit_reasoning", question_types)

    def test_generic_verification_and_novelty_do_not_inflate_question_types(self) -> None:
        taxonomy = load_taxonomy()
        entry = {
            "content": (
                "Solve this novel puzzle and verify every arithmetic step. "
                "Give one concrete counterexample if the claim fails."
            )
        }
        hits = taxonomy_hits_for_entry(entry, taxonomy)
        question_types = {hit["tag"] for hit in hits if hit["dimension"] == "question_type"}

        self.assertNotIn("source_citation_verification", question_types)
        self.assertNotIn("creative_generation_divergent", question_types)

    def test_source_verification_requires_source_specific_language(self) -> None:
        taxonomy = load_taxonomy()
        entry = {
            "content": (
                "Fact-check the empirical claim against its cited source and assess "
                "the source reliability before using the citation."
            )
        }
        hits = taxonomy_hits_for_entry(entry, taxonomy)
        question_types = {hit["tag"] for hit in hits if hit["dimension"] == "question_type"}

        self.assertIn("source_citation_verification", question_types)

    def test_cross_domain_words_do_not_create_spurious_question_types(self) -> None:
        taxonomy = load_taxonomy()
        entry = {
            "content": (
                "Explain what pattern in the data selection bias could produce. "
                "Keep the summaries consistent. This complements an earlier causal probe."
            )
        }
        hits = taxonomy_hits_for_entry(entry, taxonomy)
        question_types = {hit["tag"] for hit in hits if hit["dimension"] == "question_type"}

        self.assertNotIn("fluid_abstract_reasoning", question_types)
        self.assertNotIn("logical_paradox_consistency", question_types)
        self.assertNotIn("stem_scientific_reasoning", question_types)
        self.assertNotIn("robustness_adversarial_safety", question_types)

    def test_register_simulation_is_state_tracking_not_generic_fluid_reasoning(self) -> None:
        taxonomy = load_taxonomy()
        entry = {
            "content": (
                "Four registers start at fixed values. Apply these operations in-place, "
                "then report the state after round 2 and the final state."
            )
        }
        hits = taxonomy_hits_for_entry(entry, taxonomy)
        question_types = {
            hit["tag"] for hit in hits if hit["dimension"] == "question_type"
        }

        self.assertIn("working_memory_state_tracking", question_types)
        self.assertNotIn("fluid_abstract_reasoning", question_types)

    def test_structured_schema_keys_are_not_behavioral_signals(self) -> None:
        taxonomy = load_taxonomy()
        entry = {
            "content": '{"confidence": 0.8, "follow_up_candidates": []}',
            "parsed": {"confidence": 0.8, "follow_up_candidates": []},
            "metadata": {"interaction_role": "evidence_card"},
        }

        hit_ids = {hit["tag"] for hit in taxonomy_hits_for_entry(entry, taxonomy)}
        self.assertNotIn("uncertainty_calibration", hit_ids)
        self.assertNotIn("adaptive_followup", hit_ids)

    def test_analysis_includes_taxonomy_signal_summary(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "taxonomy_analysis_test",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [{"id": "P1", "model": "mock_model"}],
                "protocol": {
                    "name": "taxonomy_protocol",
                    "phases": [
                        {
                            "name": "reflection",
                            "kind": "private_reflection",
                            "prompt": "private_test_design",
                            "visibility": "private",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            store.append_entry(
                TranscriptEntry(
                    turn_id=1,
                    phase="reflection",
                    speaker="P1",
                    visibility="private",
                    content="I propose an adversarial edge case and a follow-up based on uncertainty.",
                    metadata={"interaction_role": "assessment"},
                )
            )
            summary = analyze_run(Path(store.run_dir))
            signals = summary["taxonomy"]["signal_frequency"]
            self.assertGreaterEqual(signals["adversarial_edge_case"], 1)
            self.assertGreaterEqual(signals["adaptive_followup"], 1)
            self.assertEqual(summary["taxonomy"]["signals_by_speaker"]["P1"]["adaptive_followup"], 1)

    def test_analysis_includes_question_type_summary(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "question_type_analysis_test",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [{"id": "P1", "model": "mock_model"}],
                "protocol": {
                    "name": "question_type_protocol",
                    "phases": [
                        {
                            "name": "probe",
                            "kind": "public_round_robin",
                            "prompt": "opening_council",
                            "visibility": "public",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            store.append_entry(
                TranscriptEntry(
                    turn_id=1,
                    phase="probe",
                    speaker="P1",
                    visibility="public",
                    content="Ask a coding debug task and then a probability puzzle.",
                    metadata={
                        "interaction_mode": "round_robin_probes",
                        "interaction_role": "question",
                    },
                )
            )
            summary = analyze_run(Path(store.run_dir))
            question_types = summary["taxonomy"]["question_type_frequency"]
            self.assertGreaterEqual(question_types["coding_algorithmic_reasoning"], 1)
            self.assertGreaterEqual(question_types["quantitative_math_reasoning"], 1)
            self.assertEqual(
                summary["taxonomy"]["probe_strategy_frequency"],
                {"direct_task_probe": 1},
            )


if __name__ == "__main__":
    unittest.main()
