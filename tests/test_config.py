import unittest

import lazynwb

from dr_datacube import config


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
        original_anon = lazynwb.config.anon

        with self.assertRaises(RuntimeError), config.override(anon=not original_anon):
            self.assertEqual(lazynwb.config.anon, not original_anon)
            raise RuntimeError("test cleanup")

        self.assertEqual(config.model_dump(), original)
        self.assertEqual(lazynwb.config.anon, original_anon)


if __name__ == "__main__":
    unittest.main()
