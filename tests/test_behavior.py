import doctest
import unittest

import pandas as pd
import polars as pl

import dr_datacube.behavior as behavior_module
from dr_datacube.behavior import (
    behavior_summary,
    get_behavior_pass_session_ids,
    is_behavior_pass,
)


def _performance(**columns) -> dict:
    n = len(columns["rewarded_modality"])
    defaults = {
        "n_contingent_rewards": [20] * n,
        "cross_modality_dprime": [2.0] * n,
        "vis_dprime": [2.0] * n,
        "aud_dprime": [2.0] * n,
    }
    return defaults | columns


class TestBehaviorCriteria(unittest.TestCase):
    def test_brainwide_requires_two_good_engaged_blocks_per_modality(self) -> None:
        # 4 passing blocks, but only one of them aud: previously passed `n_passing_blocks > 4`-style
        # checks inconsistently; the criterion is per modality.
        two_each = pl.DataFrame(_performance(rewarded_modality=["vis", "aud", "vis", "aud"]))
        self.assertTrue(is_behavior_pass(two_each, "brainwide"))

        one_aud = pl.DataFrame(
            _performance(rewarded_modality=["vis", "aud", "vis", "aud"], cross_modality_dprime=[2.0, 2.0, 2.0, 0.5])
        )
        self.assertFalse(is_behavior_pass(one_aud, "brainwide"))

        disengaged = pl.DataFrame(
            _performance(rewarded_modality=["vis", "aud", "vis", "aud"], n_contingent_rewards=[20, 20, 20, 9])
        )
        self.assertFalse(is_behavior_pass(disengaged, "brainwide"))

    def test_naive_requires_four_engaged_blocks(self) -> None:
        rewards = [10, 10, 10, 10, 5, 5]  # threshold is inclusive
        performance = pl.DataFrame(
            _performance(rewarded_modality=["vis", "aud"] * 3, n_contingent_rewards=rewards, cross_modality_dprime=[0.0] * 6)
        )
        self.assertTrue(is_behavior_pass(performance, "naive"))
        self.assertFalse(is_behavior_pass(performance[1:], "naive"))

    def test_templeton_single_modality_dprime(self) -> None:
        performance = pl.DataFrame(
            {
                "rewarded_modality": ["vis"],
                "n_contingent_rewards": [0],
                "cross_modality_dprime": [None],
                "vis_dprime": [1.5],
                "aud_dprime": [None],
            },
            schema_overrides={"cross_modality_dprime": pl.Float64, "aud_dprime": pl.Float64},
        )
        self.assertTrue(is_behavior_pass(performance, "templeton"))
        self.assertFalse(is_behavior_pass(performance.with_columns(vis_dprime=pl.lit(0.5)), "templeton"))

    def test_pandas_nan_is_not_applicable(self) -> None:
        # npc_sessions performance tables are pandas, with NaN where d' doesn't apply.
        # NaN >= 1 is True in polars, so NaN must be treated as null.
        nan = float("nan")
        performance = pd.DataFrame(
            _performance(
                rewarded_modality=["vis", "aud", "vis", "aud"],
                cross_modality_dprime=[2.0, 2.0, nan, nan],
            )
        )
        self.assertFalse(is_behavior_pass(performance, "brainwide"))
        self.assertEqual(behavior_summary(performance)["n_good_single_modality_blocks"].item(), 2)
        self.assertEqual(behavior_summary(performance)["n_good_blocks"].item(), 2)

    def test_missing_dprime_columns_are_not_applicable(self) -> None:
        performance = pd.DataFrame({"rewarded_modality": ["aud"], "n_contingent_rewards": [3], "aud_dprime": [2.0]})
        self.assertTrue(is_behavior_pass(performance, "templeton"))
        self.assertFalse(is_behavior_pass(performance, "brainwide"))

    def test_multiple_sessions(self) -> None:
        performance = pl.concat(
            [
                pl.DataFrame(_performance(rewarded_modality=["vis", "aud", "vis", "aud"])).with_columns(
                    session_id=pl.lit("pass")
                ),
                pl.DataFrame(_performance(rewarded_modality=["vis", "aud"])).with_columns(session_id=pl.lit("fail")),
            ]
        )
        self.assertEqual(get_behavior_pass_session_ids(performance, "brainwide"), ["pass"])
        self.assertEqual(get_behavior_pass_session_ids(performance, ["brainwide", "naive"]), ["pass"])
        with self.assertRaises(ValueError):
            is_behavior_pass(performance, "brainwide")

    def test_doctests(self) -> None:
        self.assertEqual(doctest.testmod(behavior_module).failed, 0)


if __name__ == "__main__":
    unittest.main()
