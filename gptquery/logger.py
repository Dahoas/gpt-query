import json
import pathlib
import os


class Logger:
    log_folder: str = None

    @classmethod
    def is_initialized(cls):
        return cls.log_folder is not None

    @classmethod
    def init(cls, log_folder):
        """
        + log_folder: path of logging directory
        """
        cls.log_folder = log_folder
        pathlib.Path(os.path.dirname(cls.log_folder)).mkdir(parents=True, exist_ok=True)
        print(f"Logging in {cls.log_folder}")

    @classmethod
    def log(cls, dicts: list[dict]):
        assert cls.is_initialized() and type(dicts) is list
        with open(f'{cls.log_folder}', 'a+') as f:
            for dict_t in dicts:
                assert type(dict_t) is dict
                json.dump(dict_t, f)
                f.write('\n')