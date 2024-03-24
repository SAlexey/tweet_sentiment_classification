import functools
from typing import Any, List, Union


def dictwrap(f):
    """
    Decorator to apply a function and return the result as a dictionary.

    Args:
        f (function): The function to apply.

    Returns:
        function: The wrapped function.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        """
        Wrapper to apply a function to a dataset.

        Args:
            *args: The inputs to apply the function to.
            **kwargs: The keyword inputs to apply the function to.

        Returns:
            Any: The result of the function.
        """
        key = kwargs.pop("key", None)
        if key:
            return {key: f(*args, **kwargs)}
        return f(*args, **kwargs)

    return wrapper


def batched(f):
    """
    Decorator to apply a function to a batch of inputs.

    Args:
        f (function): The function to apply.

    Returns:
        function: The wrapped function.

    Notes:
        If the first argument is a list, the function will be applied to each element of the list.
        If there are multiple arguments, the function will be applied to each element of the zipped lists.
    """

    @functools.wraps(f)
    def wrapper(*args: Union[Any, List[Any]], **kwargs: Any) -> Union[Any, List[Any]]:
        """
        Wrapper to apply a function to a batch of inputs.

        Args:
            *args (Any | List[Any]): The inputs to apply the function to.
            **kwargs (Any): The keyword inputs to apply the function to.

        Returns:
            Any | List[Any]: The result of the function.
        """
        if isinstance(args[0], list):
            if len(args) == 1:
                return [f(arg, **kwargs) for arg in args[0]]
            return [f(*arg, **kwargs) for arg in zip(*args)]
        return f(*args, **kwargs)

    return wrapper
