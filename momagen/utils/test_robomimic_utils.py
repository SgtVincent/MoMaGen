import os
from types import SimpleNamespace

from momagen.utils import robomimic_utils


def test_create_env_forwards_curobo_backend_version_and_restores_env(monkeypatch):
    captured = {}

    class FakeEnvOmniGibson:
        @classmethod
        def create_for_data_processing(cls, **kwargs):
            captured["backend_env"] = os.environ.get("MOMAGEN_CUROBO_BACKEND_VERSION")
            captured["kwargs"] = kwargs
            return SimpleNamespace(created=True)

    class FakeLoader:
        def exec_module(self, module):
            module.EnvOmniGibson = FakeEnvOmniGibson

    monkeypatch.setattr(robomimic_utils.EnvUtils, "get_env_type", lambda env_meta: 4)
    monkeypatch.setattr(
        robomimic_utils.importlib.util,
        "spec_from_file_location",
        lambda *args, **kwargs: SimpleNamespace(loader=FakeLoader()),
    )
    monkeypatch.setattr(
        robomimic_utils.importlib.util,
        "module_from_spec",
        lambda spec: SimpleNamespace(),
    )
    monkeypatch.setenv("MOMAGEN_CUROBO_BACKEND_VERSION", "v1")

    env_meta = {
        "env_name": "r1_turning_on_radio",
        "env_kwargs": {
            "env": {},
            "task": {},
        },
    }
    env = robomimic_utils.create_env(env_meta, init_curobo=True, curobo_backend_version="v2")

    assert env.created
    assert captured["backend_env"] == "v2"
    assert captured["kwargs"]["curobo_backend_version"] == "v2"
    assert captured["kwargs"]["init_curobo"] is True
    assert os.environ["MOMAGEN_CUROBO_BACKEND_VERSION"] == "v1"
