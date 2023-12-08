import json

class Logger:
    name: str = None

    @classmethod
    def is_initialized(cls):
        return cls.name is not None

    @classmethod
    def init(cls, name):
        cls.name = name
        print(f"Logging in {cls.name}")

    @classmethod
    def log(cls, dicts):
        assert cls.is_initialized() and type(dicts) is list
        with open(f'{cls.name}', 'a+') as f:
            for dict_t in dicts:
                assert type(dict_t) is dict
                json.dump(dict_t, f)
                f.write('\n')