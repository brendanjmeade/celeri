import pytest


def pytest_generate_tests(metafunc):
    if "config_name" in metafunc.fixturenames:
        metafunc.parametrize(
            "config_name",
            ["test_japan_config", "test_wna_config"],
        )
