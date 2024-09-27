import random
from pytest import mark

from wlstar.base.semiring import Real, Tropical, Boolean
from wlstar.base.symbol import Sym
from wlstar.angluin import l_star, Oracle
from wlstar.base.fsa import FSA, State
from wlstar.base.misc import _random_weight as rw

a = Sym("a")
b = Sym("b")
c = Sym("c")

alphabet ={a,b,c}
 
# FIRST TYPE OF MACHINE

fsa1 = FSA(R=Real)
fsa1.add_arc(State(0), a, State(0), rw(Real)) #Insert some random weights
fsa1.add_arc(State(0), b, State(1), rw(Real))   
fsa1.add_arc(State(1), a, State(1), w=Real(0.3))
fsa1.set_I(State(0), w=Real(0.5))
fsa1.add_F(State(0), rw(Real))
fsa1.add_F(State(1), w=Real(0.2))

fsa2 = FSA(R=Tropical)
fsa2.add_arc(State(0), a, State(0), w=Tropical(1))
fsa2.add_arc(State(0), b, State(1), w=Tropical(2))
fsa2.add_arc(State(1), a, State(1), w=Tropical(3))
fsa2.set_I(State(0), w=Tropical(1))
fsa2.add_F(State(0), w=Tropical(1))
fsa2.add_F(State(1), w=Tropical(2))

fsa3 = FSA(R=Boolean)
fsa3.add_arc(State(0), a, State(0), w=Boolean(True))
fsa3.add_arc(State(0), b, State(1), w=Boolean(True))
fsa3.add_arc(State(1), a, State(1), w=Boolean(True))
fsa3.set_I(State(0), w=Boolean(True))
fsa3.add_F(State(0), w=Boolean(True))
fsa3.add_F(State(1), w=Boolean(True))

#SECOND TYPE OF MACHINE

fsa4 = FSA(R=Real)
fsa4.add_arc(State(0), a, State(1), w=Real(0.5))
fsa4.add_arc(State(0), b, State(2), w=Real(0.2))
fsa4.add_arc(State(1), c, State(1), rw(Real))
fsa4.add_arc(State(2), c, State(2), w=Real(0.2))
fsa4.add_arc(State(1), b, State(2), w=Real(0.3))
fsa4.add_arc(State(2), a, State(1), rw(Real))
fsa4.add_arc(State(1), a, State(3), w=Real(0.3))
fsa4.add_arc(State(2), b, State(3), w=Real(0.2))
fsa4.set_I(State(0), w=Real(0.1))
fsa4.set_F(State(3), rw(Real))

fsa5 = FSA(R=Tropical)
fsa5.add_arc(State(0), a, State(1), w=Tropical(1))
fsa5.add_arc(State(0), b, State(2), w=Tropical(2))
fsa5.add_arc(State(1), c, State(1), w=Tropical(3))
fsa5.add_arc(State(2), c, State(2), w=Tropical(2))
fsa5.add_arc(State(1), b, State(2), w=Tropical(3))
fsa5.add_arc(State(2), a, State(1), w=Tropical(2))
fsa5.add_arc(State(1), a, State(3), w=Tropical(3))
fsa5.add_arc(State(2), b, State(3), w=Tropical(2))
fsa5.set_I(State(0), w=Tropical(1))
fsa5.set_F(State(3), w=Tropical(3))

fsa6 = FSA(R=Boolean)
fsa6.add_arc(State(0), a, State(1), w=Boolean(True))
fsa6.add_arc(State(0), b, State(2), w=Boolean(True))
fsa6.add_arc(State(1), c, State(1), w=Boolean(True))
fsa6.add_arc(State(2), c, State(2), w=Boolean(True))
fsa6.add_arc(State(1), b, State(2), w=Boolean(True))
fsa6.add_arc(State(2), a, State(1), w=Boolean(True))
fsa6.add_arc(State(1), a, State(3), w=Boolean(True))
fsa6.add_arc(State(2), b, State(3), w=Boolean(True))
fsa6.set_I(State(0), w=Boolean(True))
fsa6.set_F(State(3), w=Boolean(True))



def test_1():
    o = Oracle(fsa1)
    output = l_star(Real, o)
    assert fsa1.equivalent(output, strategy = "DETERMINISTIC")
    print("test 1 passed")

def test_2():
    o = Oracle(fsa2)
    output = l_star(Tropical, o)
    assert fsa2.equivalent(output, strategy="DETERMINISTIC")
    print("test 2 passed")

def test_3():
    o = Oracle(fsa3)
    output = l_star(Boolean, o)
    assert fsa3.equivalent(output, strategy="DETERMINISTIC")
    print("test 3 passed")

######################################################


def test_4():
    o = Oracle(fsa4)
    output = l_star(Real, o)
    assert fsa4.equivalent(output, strategy = "DETERMINISTIC")
    print("test 4 passed")

def test_5():
    o = Oracle(fsa5)
    output = l_star(Tropical, o)
    assert fsa5.equivalent(output, strategy = "DETERMINISTIC")
    print("test 5 passed")

def test_6():
    o = Oracle(fsa6)
    output = l_star(Boolean, o)
    assert fsa6.equivalent(output, strategy = "DETERMINISTIC")
    print("test 6 passed")


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()