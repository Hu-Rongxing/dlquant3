# config/dyna_config.py
import os
from dynaconf import Dynaconf

# 使用绝对路径
settings = Dynaconf(
    settings_files=[
        os.path.join(os.path.dirname(__file__), 'conf_app.toml'),
        os.path.join(os.path.dirname(__file__), "conf_strategy.toml"),
        os.path.join(os.path.dirname(__file__), ".secrets.toml")
    ],
    environments=False,
)
