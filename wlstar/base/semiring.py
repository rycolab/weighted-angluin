from collections import defaultdict as dd
from fractions import Fraction
from math import exp, log

import numpy as np
from frozendict import frozendict


# base code from
# https://github.com/timvieira/hypergraphs/blob/master/hypergraphs/semirings/boolean.py
class Semiring:
    zero: "Semiring"
    one: "Semiring"
    idempotent = False

    def __init__(self, value):
        self.value = value

    @classmethod
    def zeros(cls, N, M):
        import numpy as np

        return np.full((N, M), cls.zero)

    @classmethod
    def chart(cls, default=None):
        if default is None:
            default = cls.zero
        return dd(lambda: default)

    @classmethod
    def diag(cls, N):
        W = cls.zeros(N, N)
        for n in range(N):
            W[n, n] = cls.one

        return W

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

class Count(Semiring):
    def __init__(self, value):
        super().__init__(value)

    def star(self):
        return self.one

    def __add__(self, other):
        if other is self.zero:
            return self
        if self is self.zero:
            return other
        return Count(self.value + other.value)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Count(self.value * other.value)

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return f"{self.value}"

    def __hash__(self):
        return hash(self.value)

    def __float__(self):
        return float(self.value)

Count.zero = Count(0)
Count.one = Count(1)
Count.idempotent = False

class String(Semiring):
    def __init__(self, value):
        super().__init__(value)

    def star(self):
        return String.one

    def __add__(self, other):
        from wlstar.base.misc import lcp

        if other is self.zero:
            return self
        if self is self.zero:
            return other
        return String(lcp(self.value, other.value))

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return String(self.value + other.value)

    def __truediv__(self, other):
        from wlstar.base.misc import lcp

        prefix = lcp(self.value, other.value)
        return String(self.value[len(prefix) :])

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return f"{self.value}"

    def __hash__(self):
        return hash(self.value)

String.zero = String("∞")
String.one = String("")
String.idempotent = False
String.cancellative = False

class Boolean(Semiring):
    def __init__(self, value):
        assert type(value) == bool
        super().__init__(value)

    def star(self):
        return Boolean.one

    def __add__(self, other):
        return Boolean(self.value or other.value)

    def __mul__(self, other):
        if other.value is self.one:
            return self.value
        if self.value is self.one:
            return other.value
        if other.value is self.zero:
            return self.zero
        if self.value is self.zero:
            return self.zero
        return Boolean(other.value and self.value)

    def __invert__(self):
        assert self.value != False, "division by zero!"
        return Boolean.one

    def __truediv__(self, other):
        assert other.value != False, "division by zero!"    
        return self
       

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"{self.value}"

    def __str__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.value)


Boolean.zero = Boolean(False)
Boolean.one = Boolean(True)
Boolean.idempotent = True
Boolean.cancellative = True


class MaxPlus(Semiring):
    def __init__(self, value):
        super().__init__(value)

    def star(self):
        return self.one

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        return MaxPlus(max(self.value, other.value))

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return MaxPlus(self.value + other.value)

    def __invert__(self):
        return MaxPlus(-self.value)

    def __truediv__(self, other):
        return MaxPlus(self.value - other.value)

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"MaxPlus({self.value})"


MaxPlus.zero = MaxPlus(float("-inf"))
MaxPlus.one = MaxPlus(0.0)
MaxPlus.idempotent = True
MaxPlus.superior = True
MaxPlus.cancellative = True


class Tropical(Semiring):
    def __init__(self, value):
        self.value = value

    def star(self):
        return self.one

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __add__(self, other):
        return Tropical(min(self.value, other.value))

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Tropical(self.value + other.value)

    def __invert__(self):
        return Tropical(-self.value)

    def __truediv__(self, other):
        return Tropical(self.value - other.value)

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"Tropical({self.value})"

    def __str__(self):
        return str(self.value)


Tropical.zero = Tropical(float("inf"))
Tropical.one = Tropical(0.0)
Tropical.idempotent = True
Tropical.superior = True
Tropical.cancellative = True


class Rational(Semiring):
    def __init__(self, value):
        self.value = Fraction(value)

    def star(self):
        return Rational(Fraction("1") / (Fraction("1") - self.value))

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        return Rational(self.value + other.value)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Rational(self.value * other.value)

    def __invert__(self):
        return Rational(1 / self.value)

    def __truediv__(self, other):
        return Rational(self.value / other.value)

    def __eq__(self, other):
        return np.allclose(float(self.value), float(other.value))

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"{self.value}"
    
    def __hash__(self):
        return hash(self.value)


Rational.zero = Rational(Fraction("0"))
Rational.one = Rational(Fraction("1"))
Rational.idempotent = False
Rational.cancellative = True


class Real(Semiring):
    def __init__(self, value):
        self.value = value

    def star(self):
        return Real(1.0 / (1.0 - self.value))

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        return Real(self.value + other.value)

    def __sub__(self, other):
        return Real(self.value - other.value)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Real(self.value * other.value)

    def __invert__(self):
        return Real(1.0 / self.value)

    def __pow__(self, other):
        return Real(self.value**other)

    def __truediv__(self, other):
        return Real(self.value / other.value)

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"{round(self.value, 5)}"

    def __eq__(self, other):
        return np.allclose(float(self.value), float(other.value), atol=1e-100)

    def __hash__(self):
        return hash(self.value)


Real.zero = Real(0.0)
Real.one = Real(1.0)
Real.idempotent = False
Real.cancellative = True

class Integer(Semiring):
    def __init__(self, value):
        # TODO: this is hack to deal with the fact
        # that we have to hash weights
        self.value = value

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        return Integer(self.value + other.value)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Integer(self.value * other.value)

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"Integer({self.value})"

    def __eq__(self, other):
        return float(self.value) == float(other.value)

    def __hash__(self):
        return hash(self.value)


Integer.zero = Integer(0)
Integer.one = Integer(1)
Integer.idempotent = False
Integer.cancellative = True
