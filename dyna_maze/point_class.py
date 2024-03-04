"""
This module defines the Point class, which represents a point in a two-dimensional
space with functionality for arithmetic operations, boundary enforcement, and
conversions between Point objects and tuples, designed to support grid-based
environments.
"""


from config import X_SIZE, Y_SIZE


class Point:
    """
    Represents a point in two-dimensional space with x and y coordinates. Provides
    functionality for basic arithmetic operations, boundary enforcement, and conversions
    between Point objects and tuples.

    Attributes:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.

    Methods:
        enforce_boundaries(x, y): Enforces the grid boundaries on the x and y coordinates.
        __eq__(other): Checks equality between two Point objects.
        __add__(other): Adds two Point objects, considering grid boundaries.
        __sub__(other): Subtracts two Point objects, considering grid boundaries.
        to_tuple(): Converts the Point object to a tuple.
        from_tuple(xy_tuple): Creates a Point object from a tuple.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def enforce_boundaries(self, x, y):
        x, y = min(x, X_SIZE - 1), min(y, Y_SIZE - 1)
        x, y = max(x, 0), max(y, 0)

        return x, y

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

    def __add__(self, other):
        if isinstance(other, Point):
            x, y = self.x + other.x, self.y + other.y
            x, y = self.enforce_boundaries(x, y)

            return Point(x, y)

        raise TypeError(
            f"Unsupported operand type(s) for +: 'Point' and '{type(other).__name__}'"
        )

    def __sub__(self, other):
        if isinstance(other, Point):
            x, y = self.x - other.x, self.y - other.y
            x, y = self.enforce_boundaries(x, y)

            return Point(x, y)

        raise TypeError(
            f"Unsupported operand type(s) for -: 'Point' and '{type(other).__name__}'"
        )

    def to_tuple(self):
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, xy_tuple):
        return cls(*xy_tuple)
