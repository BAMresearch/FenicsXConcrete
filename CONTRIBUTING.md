# Contributing

## Code coverage

When contributing new code, please make sure your code is covered by respective tests.

The coverage is automatically evaluated and the tests will fail if you try to introduce changes which will reduce the coverage, compared to the main branch.

## Coding conventions

These are the conventions that you should follow when contributing to the repository. Some of them will be checked automatically during a pull request. Following these conventions will make code reviews much easier.

### Docstrings and type hints

We use the [Google docstring format](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html) without type hints in the string, since they will we added in the function header according to [Python docs](https://docs.python.org/3/library/typing.html). 
**Example:**

```python
def multiply_by_2(x : int) -> int:
    """
    This function multiplies a number by 2.
    
    Args:
        x: Parameter description
    Returns:
        The product x * 2
    """
    
    return x * 2

class MyClass:
    """
    This class just contains two numbers.

    Attributes:
        attribute_1 (int): Description of Attribute 1
        attribute_2 (float): Description of Attribute 2

    Args:
        i: First parameter of the init
        x: Second parameter
    """
    
    def __init__(self, i : int, x : float = 3.14) -> None:
        # Comments starting with "#" are not docstrings
        # It's also possible to have a docstring with "Args"
        # in the init instead of the class docstring, but 
        # you should only do one of both.
        self.attribute_1 = i
        self.attribute_2 = x

# Special case: Union type format in PEP 604
def add(a : int | float, b : int | float) ->  int | float:
    """
    This function adds two numbers
    
    Args:
        a: A number
        b: Another number
    
    Returns:
        The sum of both numbers
    """
    return a + b

# None as default value requires a Union type
def optional(s : str | None = None) -> None:
    """
    This function prints a string

    Args:
        s: A string to be printed
    
    Returns:
        None
    """
    print(s if s is not None else "No input provided")
    
```

The following docstring currently produces a bug in sphinx:

```python
class MyClass:
    """
    Docstring
 
    Attributes:
        attribute_1: Description of Attribute 1
        attribute_2: Description of Attribute 2
    """
    attribute_1: int
    attribute_2: float
```
Therefore, please don't use type annotations for class attributes, instead write them into the docstring. We will try to fix this.

### Formatting

We use `black` and `isort` as automated formatters. On pull requests, this is automatically checked and you can only merge if your code passes the check. You can include both in a [pre-commit](https://pre-commit.com/) locally.

**Note:** We may decide on slight variations in the code formatting, like the allowed line length. This will be mentioned here in the future.
