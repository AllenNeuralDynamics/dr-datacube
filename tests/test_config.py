import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import lazynwb

from dr_datacube import config
from dr_datacube.datacube import DatacubeConfig, _asset_name_has_version


def resolved_lazynwb_anon() -> bool:
    return lazynwb.file_io._resolve_anon_setting()


class TestDatacubeConfigOverride(unittest.TestCase):
    def test_override_does_not_change_global_config(self) -> None:
        original = config.model_dump()

        with config.override(version="temporary", use_cache=not config.use_cache) as temporary:
            self.assertIsNot(temporary, config)
            self.assertEqual(temporary.version, "vtemporary")
            self.assertEqual(temporary.use_cache, not config.use_cache)
            self.assertEqual(config.model_dump(), original)

        self.assertEqual(config.model_dump(), original)

    def test_overrides_can_be_nested(self) -> None:
        with config.override(version="outer") as outer:
            with config.override(version="inner") as inner:
                self.assertEqual(inner.version, "vinner")
                self.assertEqual(inner.use_cache, outer.use_cache)
            self.assertEqual(outer.version, "vouter")

    def test_version_is_prefixed(self) -> None:
        with config.override(version="0.0.289") as temporary:
            self.assertEqual(temporary.version, "v0.0.289")

        with config.override(version="v0.0.289") as temporary:
            self.assertEqual(temporary.version, "v0.0.289")

    def test_version_assignment_is_prefixed(self) -> None:
        with config.override() as temporary:
            temporary.version = "0.0.289"
            self.assertEqual(temporary.version, "v0.0.289")

    def test_override_is_restored_after_exception(self) -> None:
        original = config.model_dump()
        original_anon = resolved_lazynwb_anon()

        with self.assertRaises(RuntimeError), config.override(anon=not original_anon):
            self.assertEqual(resolved_lazynwb_anon(), not original_anon)
            raise RuntimeError("test cleanup")

        self.assertEqual(config.model_dump(), original)
        self.assertEqual(resolved_lazynwb_anon(), original_anon)

    def test_generic_environment_variables_are_ignored(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"VERSION": "1.2.3", "USE_CACHE": "not-a-boolean"},
            clear=True,
        ):
            temporary = DatacubeConfig()

        self.assertEqual(temporary.version, "v0.0.289")
        self.assertFalse(temporary.use_cache)

    def test_namespaced_environment_variables_are_used(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"DR_DATACUBE_VERSION": "1.2.3", "DR_DATACUBE_USE_CACHE": "true"},
            clear=True,
        ):
            temporary = DatacubeConfig()

        self.assertEqual(temporary.version, "v1.2.3")
        self.assertTrue(temporary.use_cache)

    def test_similar_local_asset_version_is_not_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "dynamicrouting_datacube_v0.0.289").mkdir()
            temporary = DatacubeConfig(
                version="v0.0.28",
                disable_asset_streaming=True,
            )

            with (
                mock.patch("dr_datacube.datacube.on_codeocean", return_value=True),
                mock.patch("dr_datacube.datacube.is_pipeline", return_value=False),
                mock.patch(
                    "dr_datacube.datacube.capsule_data_dir",
                    return_value=data_dir,
                ),
                self.assertRaises(FileNotFoundError),
            ):
                _ = temporary.asset_dir

    def test_asset_version_matching_uses_a_complete_token(self) -> None:
        self.assertTrue(
            _asset_name_has_version(
                "dynamicrouting_datacube_v0.0.28-copy",
                "v0.0.28",
            )
        )
        self.assertFalse(
            _asset_name_has_version(
                "dynamicrouting_datacube_v0.0.289",
                "v0.0.28",
            )
        )
        self.assertFalse(
            _asset_name_has_version(
                "dynamicrouting_datacube_v0.0.28.naive",
                "v0.0.28",
            )
        )


class TestConcurrentDatacubeConfigOverride(unittest.IsolatedAsyncioTestCase):
    async def test_anonymous_access_is_context_local(self) -> None:
        original_anon = config.anon
        config.anon = False
        first_entered = asyncio.Event()
        contexts_overlap = asyncio.Event()
        first_inspected = asyncio.Event()

        async def first_context() -> bool:
            with config.override(anon=True):
                first_entered.set()
                await contexts_overlap.wait()
                result = resolved_lazynwb_anon()
                first_inspected.set()
                return result

        async def second_context() -> bool:
            await first_entered.wait()
            with config.override(anon=False):
                result = resolved_lazynwb_anon()
                contexts_overlap.set()
                await first_inspected.wait()
                return result

        try:
            first_anon, second_anon = await asyncio.gather(
                first_context(),
                second_context(),
            )
            self.assertTrue(first_anon)
            self.assertFalse(second_anon)
            self.assertFalse(resolved_lazynwb_anon())
        finally:
            config.anon = original_anon


if __name__ == "__main__":
    unittest.main()
