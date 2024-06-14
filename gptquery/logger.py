import json
import pathlib
import os
from typing import List, Dict
from copy import deepcopy

from gptquery.utils import recursively_serialize


class Logger:
    log_folders: Dict[str, str] = {}

    @classmethod
    def init(cls, log_folder, identity="default"):
        """
        + log_folder: path of logging directory
        """
        cls.log_folders[identity] = log_folder
        pathlib.Path(os.path.dirname(log_folder)).mkdir(parents=True, exist_ok=True)
        print(f"Logging identity {identity} in {log_folder}")

    @classmethod
    def log(cls, dicts: List[dict], identity="default"):
        assert identity in cls.log_folders
        assert identity != "default" or len(cls.log_folders) < 2
        assert type(dicts) is list or type(dicts) is dict
        log_folder = cls.log_folders[identity]
        with open(f'{log_folder}', 'a+') as f:
            if type(dicts) is dict:
                dicts = [dicts]
            for dict_t in dicts:
                assert type(dict_t) is dict
                try:
                    json.dump(dict_t, f)
                except TypeError:
                    # Try casting non-primitive fields to a string
                    dict_t = recursively_serialize(deepcopy(dict_t))
                    json.dump(dict_t, f)
                f.write('\n')