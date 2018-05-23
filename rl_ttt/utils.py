"""
Module for utils functions
"""


def log_call(log_args=False, log_result=False):
    def wrapper(f):
        def inner(obj, *args, **kwargs):
            result = f(obj, *args, **kwargs)
            fmt_str = "[{}] {}.{}{}{}".format(
                obj.name, obj.__class__.__name__, f.__name__,
                (args, kwargs) if log_args else "(...)",
                ' => {}'.format(result) if log_result else ''
            )
            print(fmt_str)
            return result
        return inner
    return wrapper


def get_stop_condition(*upper_limits):
    if all(limit is None for limit in upper_limits):
        raise ValueError('At lest one upper limit must be set!')

    def inner(*current_values):
        return all(curr < limit
                   for curr, limit in zip(current_values, upper_limits)
                   if limit is not None)

    return inner
