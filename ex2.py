from logger.log import log_debug
from functools import reduce

# List Comprehensions
numbers = [1, 2, 3, 4, 5]
squared_numbers = [num**2 for num in numbers if num % 2 == 0]
log_debug(squared_numbers)

# Generator Function


def fibonacci_generator(limit):
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b


fibonacci_numbers = list(fibonacci_generator(5))
log_debug(fibonacci_numbers)

# Lambda Functions


def add(x, y): return x + y


result = add(5, 10)
log_debug(result)

# Map, Filter, and Reduce
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x**2, numbers))
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
sum_of_numbers = reduce(lambda x, y: x + y, numbers)
log_debug(squared_numbers)
log_debug(even_numbers)
log_debug(sum_of_numbers)

# Decorators


def my_decorator(func):
    def wrapper():
        log_debug("Something is happening before the function is called.")
        func()
        log_debug("Something is happening after the function is called.")
    return wrapper


@my_decorator
def say_hello():
    log_debug("Hello!")


say_hello()
