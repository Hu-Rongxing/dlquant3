from config import settings


def test_dynaconf_settings():
    print(settings.to_dict())
    assert settings.get("testtoml.TEST_SETTING", "") == 'test_toml_settings'
    assert settings.get("testsecr.TEST_SETTING", "") == 'test_secrets_settings'
    print(__name__)

