"""Settings loading and path resolution tests."""

from weavevision.settings import load_settings


def test_load_settings_resolves_project_paths() -> None:
    settings = load_settings()
    assert settings.resolved_data_root().is_absolute()
    assert settings.resolved_database().name == "weavevision.sqlite3"
    assert settings.runtime.seed == 42
