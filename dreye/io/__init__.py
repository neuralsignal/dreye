"""
I/O functionalities
"""

from dreye.io.serialization import (
    read_json, write_json, read_pickle, write_pickle
)

# TODO test serialization API
# TODO add binary (pickle) serialization?


__all__ = [
    'read_json', 'write_json',
    'read_pickle', 'write_pickle'
]
