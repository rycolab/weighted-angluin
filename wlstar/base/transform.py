from typing import Set
from collections import defaultdict as dd

from wlstar.base.partitions import PartitionRefinement
from wlstar.base.fst import FST
from wlstar.base.semiring import Boolean
from wlstar.base.fsa import FSA
from wlstar.base.state import MinimizeState, PowerState, State
from wlstar.base.pathsum import Pathsum


class Transform:
    @staticmethod
    def _add_trim_arcs(F: FSA, T: FSA, AC: Set[State]):
        for i in AC:
            if isinstance(F, FST):
                for (a, b), j, w in F.arcs(i):
                    if j in AC:
                        T.add_arc(i, a, b, j, w)

            else:
                for a, j, w in F.arcs(i):
                    if j in AC:
                        T.add_arc(i, a, j, w)

    @staticmethod
    def trim(F: FSA) -> FSA:
        """trims the machine"""

        # compute accessible and co-accessible arcs
        A, C = F.accessible(), F.coaccessible()
        AC = A.intersection(C)

        # create a new F with only the pruned arcs
        T = F.spawn()
        Transform._add_trim_arcs(F, T, AC)

        # add initial state
        for q, w in F.I:
            if q in AC:
                T.set_I(q, w)

        # add final state
        for q, w in F.F:
            if q in AC:
                T.set_F(q, w)

        return T
    
    @staticmethod
    def push(fsa):
        W = Pathsum(fsa).backward()
        pfsa = Transform._push(fsa, W)
        return pfsa
    
    @staticmethod
    def _push(fsa, V):
        """
        Mohri (2001)'s weight pushing algorithm. See Eqs 1, 2, 3.
        Link: www.isca-speech.org/archive_v0/archive_papers/eurospeech_2001/e01_1603.pdf
        """
        from wlstar.base.fst import FST

        pfsa = fsa.spawn()
        for i in fsa.Q:
            pfsa.set_I(i, fsa.λ[i] * V[i])
            pfsa.set_F(i, ~V[i] * fsa.ρ[i])
            for a, j, w in fsa.arcs(i):
                if isinstance(fsa, FST):
                    pfsa.add_arc(i, a[0], a[1], j, ~V[i] * w * V[j])
                else:
                    pfsa.add_arc(i, a, j, ~V[i] * w * V[j])

        return pfsa

    @staticmethod
    def lift_weights_to_labels(fsa):
        assert fsa.pushed
        assert fsa.deterministic

        if_weights = {"init": {}, "final": {}}

        lifted_fsa = FSA(R=Boolean)
        for i, w in fsa.I:
            if_weights["init"][0] = w
            lifted_fsa.add_I(i, Boolean.one)
        for f, w in fsa.F:
            if_weights["final"][f] = w
            lifted_fsa.add_F(f, Boolean.one)
        for p in fsa.Q:
            for a, q, w in fsa.arcs(p):
                lifted_fsa.add_arc(p, (a, w), q, Boolean.one)
        return lifted_fsa, if_weights

    @staticmethod
    def get_weights_from_labels(fsa, R, if_weights):
        wfsa = FSA(R=R)
        for i, _ in fsa.I:
            wfsa.add_I(i, if_weights["init"][0])
        for f, _ in fsa.F:
            if f in if_weights["final"].keys():  # otherwise the states got merged
                wfsa.add_F(f, if_weights["final"][list(f.idx).pop()])
            else:
                wfsa.add_F(f, if_weights["final"][list(f.idx).pop()])
        for p in fsa.Q:
            for label, q, _ in fsa.arcs(p):
                a, w = label.value
                wfsa.add_arc(p, a, q, w)
        return wfsa
    
    @staticmethod
    def _construct_minimized(fsa, clusters):
        """Takes in the produced minimized states (subsets) and constructs a
        new FSA with those states and correct arcs between them."""

        # create new power states
        minstates = {}
        for qs in clusters:
            minstate = MinimizeState(frozenset(qs))
            for q in qs:
                minstates[q] = minstate

        # create minimized FSA
        mfsa = fsa.spawn()

        # add arcs
        for q in fsa.Q:
            for a, j, w in fsa.arcs(q):
                mfsa.add_arc(minstates[q], a, minstates[j], w=w)

        # add initial states
        for q, w in fsa.I:
            mfsa.add_I(minstates[q], w)

        # add final states
        for q, w in fsa.F:
            mfsa.add_F(minstates[q], w)

        return mfsa
    
    @staticmethod
    def minimize(fsa, strategy=None):
        assert fsa.deterministic
        assert fsa.R == Boolean

        F = set([q for q, _ in fsa.F])
        P = frozenset([frozenset(F), frozenset(fsa.Q - F)])

        for a in fsa.Sigma:
            f_a = {}
            for p in fsa.Q:
                for b, q, _ in fsa.arcs(p):
                    if a == b:
                        f_a[p] = q

            Q = frozenset(f_a.keys())

            for q in fsa.Q - f_a.keys():
                f_a[q] = q

            if strategy is None or strategy == "fast":
                P = PartitionRefinement(f_a, Q).hopcroft_fast(P)
            elif strategy == "partition":
                P = PartitionRefinement(f_a, Q).hopcroft(P)
        return Transform._construct_minimized(fsa, P)
    
    @staticmethod
    def _powerarcs(fsa, Q):
        """This helper method group outgoing arcs for determinization."""

        symbol2arcs, unnormalized_residuals = dd(set), fsa.R.chart()

        for q, old_residual in Q.residuals.items():
            for a, p, w in fsa.arcs(q):
                symbol2arcs[a].add(p)
                unnormalized_residuals[(a, p)] += old_residual * w

        for a, ps in symbol2arcs.items():
            normalizer = sum(
                [unnormalized_residuals[(a, p)] for p in ps], start=fsa.R.zero
            )
            # this does not assume commutivity
            residuals = {p: ~normalizer * unnormalized_residuals[(a, p)] for p in ps}

            yield a, PowerState(residuals), normalizer


    @staticmethod
    def determinize(fsa, timeout=1000):
        """
        The on-the-fly determinization method (Mohri 2009).
        Link: https://link.springer.com/chapter/10.1007/978-3-642-01492-5_6
        """
        D = fsa.spawn()

        stack, visited = [], set([])
        Q = PowerState({p: w for p, w in fsa.I})
        D.set_I(Q, fsa.R.one)
        stack.append(Q)
        visited.add(Q)

        counter = 0
        while stack:
            if counter > timeout:
                raise TimeoutError

            P = stack.pop()
            for a, Q, w in Transform._powerarcs(fsa, P):
                if Q not in visited:
                    stack.append(Q)
                    visited.add(Q)

                # TODO: can we propogate a change where we make this add_arc in FSA add?
                D.add_arc(P, a, Q, w)

            counter += 1

        for powerstate in D.Q:
            for q in powerstate.idx:
                D.add_F(powerstate, fsa.ρ[q] * powerstate.residuals[q])

        return D
