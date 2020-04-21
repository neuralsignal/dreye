"""Classes for error messages
"""

from pint import DimensionalityError


class DreyeUnitError(DimensionalityError):
    pass


class DreyeError(Exception):
    pass


class DreyeSerializerError(Exception):
    def __init__(self, arg, *args, **kwargs):
        default_message = f"Data of type '{type(arg)}' is not serializable"
        args = (default_message, ) + args
        # Call super constructor
        super().__init__(*args, **kwargs)
