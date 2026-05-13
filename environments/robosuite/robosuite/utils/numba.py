"""
Numba utils.
"""
import numba

import robosuite.macros as macros


def jit_decorator(func):
    if macros.ENABLE_NUMBA:
        try:
            return numba.jit(nopython=True, cache=macros.CACHE_NUMBA)(func)
        except RuntimeError as e:
            if macros.CACHE_NUMBA and "no locator available" in str(e):
                return numba.jit(nopython=True, cache=False)(func)
            raise
    return func
