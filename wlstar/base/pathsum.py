from typing import DefaultDict
from frozendict import frozendict

from wlstar.base.semiring import Semiring
from wlstar.base.state import State

class Pathsum:
    def __init__(self, fsa):
        # basic FSA stuff
        self.fsa = fsa
        self.R = fsa.R
        self.N = self.fsa.num_states

        # state dictionary
        self.I = {}  # noqa: E741
        for n, q in enumerate(self.fsa.Q):
            self.I[q] = n

        # lift into the semiring
        self.W = self.lift()

    def lift(self):
        """creates the weight matrix from the automaton"""
        W = self.R.zeros(self.N, self.N)
        for p in self.fsa.Q:
            for a, q, w in self.fsa.arcs(p):
                W[self.I[p], self.I[q]] += w
        return W

    def pathsum(self) -> float:  
        return self.lehmann_pathsum()

    def _lehmann(self, zero=True):
        """
        Lehmann's (1977) algorithm.
        """

        # initialization
        V = self.W.copy()
        U = self.W.copy()

        # basic iteration
        for j in range(self.N):
            V, U = U, V
            V = self.R.zeros(self.N, self.N)
            for i in range(self.N):
                for k in range(self.N):
                    # i ➙ j ⇝ j ➙ k
                    V[i, k] = U[i, k] + U[i, j] * U[j, j].star() * U[j, k]

        # post-processing (paths of length zero)
        if zero:
            for i in range(self.N):
                V[i, i] += self.R.one

        return V

    def lehmann(self, zero=True):

        V = self._lehmann(zero=zero)

        W = {}
        for p in self.fsa.Q:
            for q in self.fsa.Q:
                if p in self.I and q in self.I:
                    W[p, q] = V[self.I[p], self.I[q]]
                elif p == q and zero:
                    W[p, q] = self.R.one
                else:
                    W[p, q] = self.R.zero

        return frozendict(W)

    def lehmann_pathsum(self):
        return self.allpairs_pathsum(self.lehmann())

    def lehmann_fwd(self) -> DefaultDict[State, float]:
        return self.allpairs_fwd(self.lehmann())

    def lehmann_bwd(self) -> DefaultDict[State, Semiring]:
        return self.allpairs_bwd(self.lehmann())
    
    def backward(self) -> DefaultDict[State, Semiring]:  # noqa: C901
            return self.lehmann_bwd()
    
    def allpairs_pathsum(self, W):
        pathsum = self.R.zero
        for p in self.fsa.Q:
            for q in self.fsa.Q:
                pathsum += self.fsa.λ[p] * W[p, q] * self.fsa.ρ[q]
        return pathsum

    def allpairs_fwd(self, W):
        α = self.R.chart()
        for p in self.fsa.Q:
            for q in self.fsa.Q:
                α[q] += self.fsa.λ[p] * W[p, q]
        return frozendict(α)

    def allpairs_bwd(self, W):
        β = self.R.chart()
        W = self.lehmann()
        for p in self.fsa.Q:
            for q in self.fsa.Q:
                β[p] += W[p, q] * self.fsa.ρ[q]
        return frozendict(β)