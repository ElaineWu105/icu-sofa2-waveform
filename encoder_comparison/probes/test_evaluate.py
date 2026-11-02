import unittest

import numpy as np
import pandas as pd

from probes.evaluate import (TASKS, Task, bootstrap, cap_windows, decide, gain, macro_f1, primary, resp_verdict, retention_verdict,
                            score)


def row(part, task, input_name, reference, candidate, verdict, gate=True):
    return {"part": part, "task": task, "input": input_name, "metric": "auroc", "reference": reference, "candidate": candidate,
            "gain": 0., "ci_low": 0., "ci_high": 0., "margin": .02, "gate": gate, "verdict": verdict, "draws": 10}


def table(a_lost=False, b_lost=False, a_rr="better", b_rr="better", b_icd="same", b_over_a="same"):
    rows = [row(1, "age", "ecg", "original", "epoch17", "retained"),
            row(1, "age", "ecg", "original", "arm_a", "lost" if a_lost else "retained"),
            row(1, "age", "ecg", "original", "arm_b", "lost" if b_lost else "retained")]
    for candidate, rr, icd in (("arm_a", a_rr, "same"), ("arm_b", b_rr, b_icd)):
        rows += [row(2, "rr_monitor", "all3", "original", candidate, rr), row(2, "support", "all3", "original", candidate, "same"),
                 row(2, "icd_respiratory", "all3", "original", candidate, icd)]
    rows += [row(2, "rr_monitor", "all3", "arm_a", "arm_b", b_over_a), row(2, "support", "all3", "arm_a", "arm_b", "same")]
    return pd.DataFrame(rows)


class EvaluateTests(unittest.TestCase):
    def test_every_part_one_task_has_a_margin(self):
        self.assertTrue(all(task.margin > 0 for task in TASKS if task.part == 1))
        self.assertEqual(len({task.name for task in TASKS}), len(TASKS))

    def test_cap_windows_keeps_all_patients_and_limit(self):
        subjects = np.array([1] * 10 + [2] * 3)
        kept = cap_windows(subjects, 6)
        self.assertEqual((subjects[kept] == 1).sum(), 6)
        self.assertEqual((subjects[kept] == 2).sum(), 3)
        self.assertEqual(kept[0], 0)
        self.assertEqual(kept[5], 9)

    def test_macro_f1_matches_hand_value(self):
        self.assertAlmostEqual(macro_f1(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), (0, 1)), (2 / 3 + .8) / 2)

    def test_score_binary_and_regression(self):
        y = np.array([0., 0., 1., 1.])
        self.assertEqual(score("binary", y, np.array([.1, .2, .8, .9]), np.array([.5]))["auroc"], 1.)
        self.assertIsNone(score("binary", np.zeros(4), np.array([.1, .2, .8, .9]), np.array([.5]))["auroc"])
        self.assertEqual(score("regression", y, y + 2)["mae"], 2.)

    def test_score_multilabel_uses_each_threshold(self):
        y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=float)
        prediction = np.array([[.1, .4], [.9, .1], [.2, .4], [.8, .2]])
        self.assertEqual(score("multilabel", y, prediction, np.array([.5, .3]))["macro_f1"], 1.)
        self.assertEqual(score("multilabel", y, prediction, np.array([.5, .9]))["macro_f1"], .5)

    def test_primary_agrees_with_score(self):
        rng = np.random.default_rng(0)
        single = rng.integers(0, 2, 300).astype(float)
        many = rng.integers(0, 2, (300, 4)).astype(float)
        classes = rng.integers(0, 5, 300)
        cases = [("binary", "auroc", single, rng.random(300), np.array([.5])),
                 ("regression", "mae", single, rng.random(300), None),
                 ("multiclass", "macro_f1", classes, rng.random((300, 5)), None),
                 ("multilabel", "macro_f1", many, rng.random((300, 4)), np.array([.3, .4, .5, .6])),
                 ("multilabel", "macro_auroc", many, rng.random((300, 4)), np.array([.3, .4, .5, .6]))]
        for kind, metric, y, prediction, thresholds in cases:
            task = Task("t", 1, "window", kind, ("t",), ("ecg",), metric, 1.)
            self.assertAlmostEqual(primary(task, y, prediction, thresholds), score(kind, y, prediction, thresholds)[metric])

    def test_gain_sign(self):
        self.assertEqual(gain("mae", 5., 4.), 1.)
        self.assertEqual(gain("auroc", .7, .8) > 0, True)

    def test_retention_verdict(self):
        self.assertEqual(retention_verdict(-.01, -.015, .01, .02), "retained")
        self.assertEqual(retention_verdict(-.053, -.097, -.01, .02), "lost")
        self.assertEqual(retention_verdict(-.01, -.05, .03, .02), "inconclusive")
        self.assertEqual(retention_verdict(-.03, -.06, .01, .02), "inconclusive")

    def test_resp_verdict(self):
        self.assertEqual(resp_verdict(.1, .2), "better")
        self.assertEqual(resp_verdict(-.2, -.1), "worse")
        self.assertEqual(resp_verdict(-.1, .1), "same")

    def test_bootstrap_resamples_patients_not_rows(self):
        task = Task("t", 1, "window", "regression", ("t",), ("ecg",), "mae", 1.)
        # Patient 1 has error 0 in all rows and patient 2 has error 1, so draws land on 0, 1/2 or 2/3 style means.
        data = {"y": np.zeros(6), "subjects": np.array([1, 1, 1, 1, 2, 2]), "pred_a": np.array([0, 0, 0, 0, 1, 1.]), "thr_a": np.array([.5])}
        values = bootstrap(task, data, ["a"], 200)["a"]
        self.assertEqual(set(np.round(values, 6)) - {0., round(1 / 3, 6), 1.}, set())

    def test_decide_both_pass_keeps_simpler_arm(self):
        self.assertEqual(decide(table())[1], "arm_a")

    def test_decide_prefers_arm_b_when_resp_is_clearly_better(self):
        self.assertEqual(decide(table(b_over_a="better"))[1], "arm_b")

    def test_decide_fails_arm_on_lost_ability(self):
        self.assertEqual(decide(table(a_lost=True))[1], "arm_b")

    def test_decide_fails_arm_b_on_fair_resp_task(self):
        self.assertEqual(decide(table(b_icd="worse"))[1], "arm_a")

    def test_decide_no_arm_needs_a_person(self):
        self.assertIsNone(decide(table(a_lost=True, b_lost=True))[1])
        self.assertIsNone(decide(table(a_rr="same", b_rr="same"))[1])


if __name__ == "__main__":
    unittest.main()
