import copy
from typing import Callable, Dict, Generator, List, Optional, Sequence, Set, Tuple, Type, Union
from frozendict import frozendict
import numpy as np
from collections import Counter, defaultdict as dd, deque

from wlstar.base.alphabet import Alphabet
from wlstar.base.pathsum import Pathsum
from wlstar.base.semiring import Boolean, Real, Semiring
from wlstar.base.state import PairState, State
from wlstar.base.symbol import Sym, ε, ε_1, ε_2

class FSA:
    def __init__(self, R: Type[Semiring] = Real):
        # DEFINITION
        # A weighted finite-state automaton is a 5-tuple <R, Σ, Q, δ, λ, ρ> where
        # • R is a semiring;
        # • Σ is an alphabet of symbols;
        # • Q is a finite set of states;
        # • δ is a finite relation Q × Σ × Q × R;
        # • λ is an initial weight function;
        # • ρ is a final weight function.

        # NOTATION CONVENTIONS
        # • single states (elements of Q) are denoted q
        # • multiple states not in sequence are denoted, p, q, r, ...
        # • multiple states in sequence are denoted i, j, k, ...
        # • symbols (elements of Σ) are denoted lowercase a, b, c, ...
        # • single weights (elements of R) are denoted w
        # • multiple weights (elements of R) are denoted u, v, w, ...alphabet

        # semiring
        self.R = R

        # alphabet of symbols
        self.Sigma = set([])
        self.symbol2idx, self.idx2symbol = {}, {}

        # a finite set of states
        self.Q = set([])
        self.state2idx, self.idx2state = {}, {}

        # transition function : Q × Σ × Q → R
        self.δ = dd(lambda: dd(lambda: dd(lambda: self.R.zero)))
        # We also define the inverse transition function δ_inv
        self.δ_inv = dd(lambda: dd(lambda: dd(lambda: self.R.zero)))

        # initial weight function
        self.λ = R.chart()

        # final weight function
        self.ρ = R.chart()

        # For displaying the FSA in a juptyer notebook
        self.theme = "dark"  # Set to "light" for a light theme

    def add_state(self, q: State) -> None:
        """Adds a state to the automaton.
        This method should mainly be accessed through the add_arc method.

        Args:
            q (State): The state to be added.
        """
        assert isinstance(self.Q, set), "Cannot add to frozen FSA"
        self.Q.add(q)

    def add_states(self, Q: Union[List[State], Set[State], Tuple[State, ...]]) -> None:
        """Adds a list of states to the automaton."""
        for q in Q:
            if q not in self.state2idx:
                self.state2idx[q] = len(self.state2idx)
                self.idx2state[self.state2idx[q]] = q
            self.add_state(q)

    def add_arc(self, i: State, a: Sym, j: State, w: Optional[Semiring] = None):
        assert isinstance(self.Sigma, set), "Cannot add to frozen FSA"
        if w is None:
            w = self.R.one

        if not isinstance(i, State):
            i = State(i)
        if not isinstance(j, State):
            j = State(j)
        if not isinstance(a, Sym):
            a = Sym(a)
        if not isinstance(w, self.R):
            w = self.R(w)

        self.add_states([i, j])
        self.Sigma.add(a)
        if a not in self.symbol2idx:
            self.symbol2idx[a] = len(self.symbol2idx)
            self.idx2symbol[self.symbol2idx[a]] = a
        self.δ[i][a][j] += w
        self.δ_inv[j][a][i] += w

    def set_arc(self, i: State, a: Sym, j: State, w: Optional[Semiring] = None):
        assert isinstance(self.Sigma, set), "Cannot add to frozen FSA"
        if w is None:
            w = self.R.one

        if not isinstance(i, State):
            i = State(i)
        if not isinstance(j, State):
            j = State(j)
        if not isinstance(a, Sym):
            a = Sym(a)
        if not isinstance(w, self.R):
            w = self.R(w)

        self.add_states([i, j])
        self.Sigma.add(a)
        if a not in self.symbol2idx:
            self.symbol2idx[a] = len(self.symbol2idx)
            self.idx2symbol[self.symbol2idx[a]] = a
        self.δ[i][a][j] = w
        self.δ_inv[j][a][i] = w

    def set_I(self, q, w=None):
        assert isinstance(self.λ, dict), "Cannot add to frozen FSA"
        if not isinstance(q, State):
            q = State(q)

        if w is None:
            w = self.R.one
        self.add_state(q)
        self.λ[q] = w

    def set_F(self, q, w=None):
        assert isinstance(self.ρ, dict), "Cannot add to frozen FSA"
        if not isinstance(q, State):
            q = State(q)

        if w is None:
            w = self.R.one
        self.add_state(q)
        self.ρ[q] = w

    def add_I(self, q, w):
        assert isinstance(self.λ, dict), "Cannot add to frozen FSA"
        self.add_state(q)
        self.λ[q] += w

    def add_F(self, q, w):
        assert isinstance(self.ρ, dict), "Cannot add to frozen FSA"
        self.add_state(q)
        self.ρ[q] += w

    def freeze(self):
        self.Sigma = frozenset(self.Sigma)
        self.Q = frozenset(self.Q)
        self.δ = frozendict(self.δ)
        self.δ_inv = frozendict(self.δ_inv)
        self.λ = frozendict(self.λ)
        self.ρ = frozendict(self.ρ)

    @property
    def I(self) -> Generator[Tuple[State, Semiring], None, None]:  # noqa: E741, E743
        """Returns the initial states of the FSA.

        Yields:
            Generator[Tuple[State, Semiring], None, None]:
            Generator of the initial states of the FSA.
        """
        for q, w in self.λ.items():
            if w != self.R.zero:
                yield q, w

    @property
    def F(self) -> Generator[Tuple[State, Semiring], None, None]:
        """Returns the final states of the FSA.

        Yields:
            Generator[Tuple[State, Semiring], None, None]:
            Generator of the final states of the FSA.
        """
        for q, w in self.ρ.items():
            if w != self.R.zero:
                yield q, w

    def arcs(
        self, i: State, no_eps: bool = False, nozero: bool = True, reverse: bool = False
    ) -> Generator[Tuple[Sym, State, Semiring], None, None]:
        """Returns the arcs stemming from state i or going into the state i in the FSA.
        in the form of tuples (a, j, w) where a is the symbol, j is the target state of
        the transition and w is the weight.

        Args:
            i (State): The state out of which the arcs stem or into which the arcs go.
            no_eps (bool, optional): If True, epsilon arcs are not returned.
                Defaults to False.
            nozero (bool, optional): If True, zero-weight arcs are not returned.
                Defaults to True.
            reverse (bool, optional): If False, the arcs stemming from state i are
                returned. If True, the arcs going into the state i are returned.
                Defaults to False.

        Yields:
            Generator[Tuple[Sym, State, Semiring], None, None]:
            Generator of the arcs stemming from state i in the FSA.
        """
        δ = self.δ if not reverse else self.δ_inv
        for a, transitions in δ[i].items():
            if no_eps and a == ε:
                continue
            for j, w in transitions.items():
                if w == self.R.zero and nozero:
                    continue
                yield a, j, w

    def a_out_arcs(
        self, q: State, a: Sym
    ) -> Generator[Tuple[State, Semiring], None, None]:
        """Returns the arcs stemming from state q with label a.

        Args:
            q (State): The state out of which the arcs stem.
            a (Sym): The label of the arcs.

        Yields:
            Generator[Tuple[State, Semiring], None, None]:
            Generator of the arcs stemming from state q with label a.
        """
        for j, w in self.δ[q][a].items():
            yield j, w

    def transition_matrix(self, a: Sym) -> List[List[Semiring]]:
        """Returns the transition matrix of the FSA for a given symbol.

        Args:
            a (Sym): The symbol for which the transition matrix is returned.

        Returns:
            List[List[Semiring]]: The transition matrix of the FSA for a given symbol.
        """
        n = self.num_states
        T_a = [[self.R.zero] * n for _ in range(n)]
        for q in self.Q:
            for r in self.Q:
                T_a[self.state2idx[q]][self.state2idx[r]] = self.δ[q][a][r]
        return T_a

    def accept(self, string: Union[str, Sequence[Sym]]) -> Semiring:
        """Determines the stringsum/acceptance weight of the string `string`
        in the rational series defined by the WFSA.

        Args:
            string (Union[str, Sequence[Union[Sym, NT]]]):
                The string whose stringsum is to be determined.

        Returns:
            Semiring: The stringsum value.
        """

        fsa_s = self.intersect(FSA.string_fsa(string, self.R))

        return Pathsum(fsa_s).pathsum()
    
    @staticmethod
    def string_fsa(y: Union[str, Sequence[Sym]], R: Type[Semiring], fst: bool = False):
        """Returns a WFSA that accepts the string y.

        Args:
            y (str): The string to accept.
            R (Type[Semiring]): The semiring to use.
            fst (bool, optional): Whether to return an FST. Defaults to False.

        Returns:
            FSA: The WFSA.
        """

        A = FSA(R=R)
        for i, x in enumerate(list(y)):
            x = Sym(x) if isinstance(x, str) else (x if isinstance(x, Sym) else Sym(x._X))
            A.add_arc(State(i), x, State(i + 1), R.one)

        A.set_I(State(0), R.one)
        A.add_F(State(len(y)), R.one)

        return A if not fst else A.to_identity_fst()


    @staticmethod
    def get_epsilon_filter(R: Type[Semiring], Sigma: Alphabet):
        """Returns the epsilon filtered FST required for the correct composition of WFSTs
        with epsilon transitions.

        Returns:
            FST: The 3-state epsilon filtered WFST.
        """
        from wlstar.base.fst import FST
        F = FST(R)

        # 0 ->
        for a in Sigma:
            F.add_arc(State(0), a, a, State(0), R.one)
        F.add_arc(State(0), ε_2, ε_1, State(0), R.one)
        F.add_arc(State(0), ε_1, ε_1, State(1), R.one)
        F.add_arc(State(0), ε_2, ε_2, State(2), R.one)

        # 1 ->
        for a in Sigma:
            F.add_arc(State(1), a, a, State(0), R.one)
        F.add_arc(State(1), ε_1, ε_1, State(1), R.one)

        # 2 ->
        for a in Sigma:
            F.add_arc(State(2), a, a, State(0), R.one)
        F.add_arc(State(2), ε_2, ε_2, State(2), R.one)

        F.set_I(State(0), R.one)

        F.set_F(State(0), R.one)
        F.set_F(State(1), R.one)
        F.set_F(State(2), R.one)

        return F


    @property
    def num_states(self) -> int:
        """Returns the number of states of the FSA."""
        return len(self.Q)

    @property
    def num_initial_states(self) -> int:
        """Returns the number of initial states of the FSA."""
        return len(list(self.I))

    @property
    def num_final_states(self) -> int:
        """Returns the number of final states of the FSA."""
        return len(list(self.F))

    @property
    def acyclic(self):
        cyclic, _ = self.dfs()
        return not cyclic

    @property
    def deterministic(self) -> bool:
        if len(list(self.I)) > 1: # previously: if not one
            print("multi-initial!")
            return False
        for q in self.Q:
            counter = Counter()
            for a, _, _ in self.arcs(q):
                if a == ε:  # a deterministic fsa cannot have ε transitions
                    print("epsilon !")
                    return False
                counter[a] += 1
            most_common = counter.most_common(1)
            if len(most_common) > 0 and most_common[0][1] > 1:
                print("multi out edges with same label")
                return False
        return True

    @property
    def pushed(self) -> bool:
        for i in self.Q:
            total = self.ρ[i]
            for _, _, w in self.arcs(i):
                total += w
            if total != self.R.one:
                return False
        return True

    @property
    def probabilistic(self) -> bool:  # noqa: C901
        assert self.R == Real

        total = self.R.zero
        for i, w in self.I:
            if not w.value >= 0:
                return False, "Initial weights must be non-negative."
            total += w
        if total != self.R.one:
            return False, "Total weight of initial states must be 1."

        for i in self.Q:
            if not self.ρ[i].value >= 0:
                return False, "Final weights must be non-negative."
            total = self.ρ[i]
            for _, _, w in self.arcs(i):
                if not w.value >= 0:
                    return False, "Transition weights must be non-negative."
                total += w
            if total != self.R.one:
                return False, "Total weight of outgoing arcs must be 1."
        return True

    @property
    def epsilon(self):
        for q in self.Q:
            for a, _, _ in self.arcs(q):
                if a == ε:
                    return True
        return False

    @property
    def ordered_states(self):
        """Returns a list of states ordered by their lexicographical index"""
        Q = list(self.Q)
        Q.sort(key=lambda a: str(a.idx))
        return Q

    @property
    def T(self) -> Dict[Sym, np.ndarray]:
        """Returns a dictionary of symbols to transition matrices.

        The matrices are indexed by state idx in lexicographical order.
        Matrix entry [i, j] corresponds to the transition weight
        from state i to state j.

        Returns:
            Dictionary of transition matrices M (one for each symbol).
        """

        assert self.R.is_field

        M = {}
        n = self.num_states
        Q = self.ordered_states

        for a in self.Sigma:
            M[a] = np.zeros((n, n))
            for i, p in enumerate(Q):
                if a in self.δ[p]:
                    for j, q in enumerate(Q):
                        M[a][i, j] = self.δ[p][a][q]

        return M

    @property
    def init_vector(self) -> np.ndarray:
        """Returns a vector of initial weights of the states, sorted by state idx
        in lexicographical order."""

        assert self.R.is_field

        n = self.num_states
        Q = self.ordered_states
        λ = np.zeros(n)

        for i, q in enumerate(Q):
            λ[i] = self.λ[q]

        return λ

    @property
    def final_vector(self) -> np.ndarray:
        """Returns a vector of final weights of the states, sorted by state idx
        in lexicographical order."""

        assert self.R.is_field

        n = self.num_states
        Q = self.ordered_states
        ρ = np.zeros(n)

        for i, q in enumerate(Q):
            ρ[i] = self.ρ[q]

        return ρ

    def copy(self):
        """deep copies the machine"""
        return copy.deepcopy(self)

    def spawn(self, keep_init=False, keep_final=False):
        """returns a new FSA in the same semiring"""
        F = FSA(R=self.R)

        if keep_init:
            for q, w in self.I:
                F.set_I(q, w)
        if keep_final:
            for q, w in self.F:
                F.set_F(q, w)

        return F
    
    def minimize(self, strategy=None):
        from wlstar.base.transform import Transform

        assert self.deterministic

        if self.R != Boolean:
            trim_fsa = self.trim()
            pushed_fsa = trim_fsa.push()
            lifted_fsa, if_weights = Transform.lift_weights_to_labels(pushed_fsa)
            min_fsa = Transform.minimize(lifted_fsa, strategy=strategy)
            return Transform.get_weights_from_labels(min_fsa, self.R, if_weights)

        else:
            return Transform.minimize(self, strategy=strategy)
    
    def push(self):
        from wlstar.base.transform import Transform

        return Transform.push(self)

    def equivalent(self, fsa, strategy=None):
        """Tests equivalence. If counterexample is set to True it returns a counterexample in the case of non-equivalence."""
        from wlstar.base.transform import Transform
        
        if self.R is not fsa.R:
            print("Not equivalent due to different semiring")
            return False

        if self.Sigma != fsa.Sigma:
            print("Not equivalent due to different alphabet")
            return False

        if strategy == "DETERMINISTIC":
            assert fsa.deterministic and self.deterministic, "The automata are not deterministic"
            fsa_1 = self.minimize()
            fsa_2 = fsa.minimize()
            # print(f"fsa1 states {fsa_1.Q}")
            # print(f"fsa2 states {fsa_2.Q}")

            queue = []
            visited = set()

            if len(list(fsa_1.I)) != len(list(fsa_2.I)):
                return False
            elif len(list(fsa_1.I)) == 0:
                return True

            q, λ1 = list(fsa_1.I)[0]  #Since we have the determinism assertion, we know that the automaton can have at most one initial state.
            p, λ2 = list(fsa_2.I)[0]
            if λ1 != λ2:
                return False

            for a in fsa_1.Sigma:
                s = str(a)
                fsa_1_arcs = list(fsa_1.a_out_arcs(q,a))
                fsa_2_arcs = list(fsa_2.a_out_arcs(p,a))
                if len(fsa_1_arcs) == 0 or len(fsa_2_arcs) == 0:
                    if len(fsa_1_arcs) == 1 and fsa_1_arcs[0][1] != fsa_1.R.zero \
                    or len(fsa_2_arcs) == 1 and fsa_2_arcs[0][1] != fsa_2.R.zero:
                        return False
                    else:
                        continue
                    
                q1, w1 = fsa_1_arcs[0]
                q2, w2 = fsa_2_arcs[0]
                if w1 != w2 or fsa_1.ρ[q1] != fsa_2.ρ[q2]: 
                    return False
                if not (q1,q2) in visited:
                    queue.append((q1,q2,s))
                    visited.add((q1,q2))

            while queue:
                (q, p, s) = queue.pop()
                for a in fsa_1.Sigma:
                    s += str(a)
                    fsa_1_arcs = list(fsa_1.a_out_arcs(q,a))
                    fsa_2_arcs = list(fsa_2.a_out_arcs(p,a))
                    if len(fsa_1_arcs) == 0 or len(fsa_2_arcs) == 0:
                        if len(fsa_1_arcs) == 1 and fsa_1_arcs[0][1] != fsa_1.R.zero \
                        or len(fsa_2_arcs) == 1 and fsa_2_arcs[0][1] != fsa_2.R.zero:
                            return False
                        else:
                            continue
                    for q1, w1 in fsa_1.a_out_arcs(q,a):
                        for q2, w2 in fsa_2.a_out_arcs(p,a):
                            if w1 != w2 or fsa_1.ρ[q1] != fsa_2.ρ[q2]:
                                return False
                            if not (q1,q2) in visited:
                                queue.append((q1,q2,s))
                                visited.add((q1,q2))
            return True

        fsa0 = Transform.determinize(
            Transform.epsremoval(self.single_I().booleanize())
        ).trim()
        fsa1 = Transform.determinize(
            Transform.epsremoval(fsa.single_I().booleanize())
        ).trim()

        fsa2 = fsa0.intersect(fsa1.complement())
        fsa3 = fsa1.intersect(fsa0.complement())

        U = fsa2.union(fsa3)

        return U.trim().num_states == 0
    
    def reverse(self):
        """creates a reversed machine"""

        # create the new machine
        R = self.spawn()

        # add the arcs in the reversed machine
        for i in self.Q:
            for a, j, w in self.arcs(i):
                R.add_arc(j, a, i, w)

        # reverse the initial and final states
        for q, w in self.I:
            R.set_F(q, w)
        for q, w in self.F:
            R.set_I(q, w)

        return R

    def accessible(self):
        """computes the set of accessible states"""
        A = set()
        stack = [q for q, w in self.I if w != self.R.zero]
        while stack:
            i = stack.pop()
            for _, j, _ in self.arcs(i):
                if j not in A:
                    stack.append(j)
            A.add(i)

        return A

    def coaccessible(self):
        """computes the set of co-accessible states"""
        return self.reverse().accessible()

    def is_parent(self, p: State, q: State) -> bool:
        """Checks whether `p` is a parent of `q` in the FSA.

        Args:
            p (State): The candidate parent
            q (State): The candidate child

        Returns:
            bool: Whether `p` is a parent of `q`
        """
        return q in [t for _, t, _ in self.arcs(p)]

    def connected_by_symbol(self, p: State, q: State, symbol: Sym) -> bool:
        """Checks whereher there is an edge from `p` to `q` with the label `symbol`.

        Args:
            p (State): The candidate parent
            q (State): The candidate child
            symbom (Sym): The arc label to check

        Returns:
            bool: Whereher there is an edge from `p` to `q` with the label `symbol`
        """
        return symbol in self.δ[p] and q in self.δ[p][symbol]

    def has_incoming_arc(self, q: State, symbol: Sym) -> bool:
        """Checks whereher there is an incoming edge into `q` with the label `symbol`.

        Args:
            q (State): The state
            symbom (Sym): The arc label to check

        Returns:
            bool: Whereher there is an incoming edge into `q` with the label `symbol`.
        """
        for p in self.Q:
            for a, t, _ in self.arcs(p):
                if a == symbol and t == q:
                    return True
        return False

    def has_outgoing_arc(self, q: State, symbol: Sym) -> bool:
        """Checks whereher there is an outgoing edge into `q` with the label `symbol`.

        Args:
            q (State): The state
            symbol (Sym): The arc label to check

        Returns:
            bool: Whether there is an outgoing edge into `q` with the label `symbol`.
        """
        return symbol in self.δ[q]

    def transition(
        self, q: State, a: Sym, weight: bool = False
    ) -> Optional[Union[State, Tuple[State, Semiring]]]:
        """If the FSA is deterministic and there exists an a-transition out of q,
            then the function returns the target state of the transition.

        Args:
            q (State): The state.
            a (Sym): The symbol.
            weight (bool, optional): Whether to return the weight of the transition.

        Returns:
            State: The target state of the transition.
        """
        assert self.deterministic

        if self.has_outgoing_arc(q, a):
            if weight:
                return list(self.δ[q][a].items())[0]
            else:
                return list(self.δ[q][a].keys())[0]
        else:
            return None

    def toposort(self, rev=False):
        return self.finish(rev=rev, acyclic_check=True)

    def trim(self):
        from wlstar.base.transform import Transform

        return Transform.trim(self)

    def pathsum(self):
        pathsum = Pathsum(self)
        return pathsum.pathsum()

    def forward(self):
        pathsum = Pathsum(self)
        return pathsum.forward()

    def backward(self):
        pathsum = Pathsum(self)
        return pathsum.backward()

    def allpairs(self):
        pathsum = Pathsum(self)
        return pathsum.allpairs()

    def booleanize(self):
        fsa = FSA(Boolean)

        for q, w in self.I:
            fsa.add_I(q, fsa.R(w != self.R.zero))

        for q, w in self.F:
            fsa.add_F(q, fsa.R(w != self.R.zero))

        for q in self.Q:
            for a, j, w in self.arcs(q):
                fsa.add_arc(q, a, j, fsa.R(w != self.R.zero))

        return fsa

    def difference(self, fsa):
        """coputes the difference with a uniweighted DFA"""

        fsa = fsa.complement()

        # fsa will be a boolean FSA, need to make the semirings compatible
        F = FSA(self.R)
        for q, w in fsa.I:
            F.add_I(q, F.R(w.value))
        for q, w in fsa.F:
            F.add_F(q, F.R(w.value))
        for q in fsa.Q:
            for a, j, w in fsa.arcs(q):
                F.add_arc(q, a, j, F.R(w.value))

        return self.intersect(F)

    def _union_add(self, A: "FSA", U: "FSA", idx: int):
        for i in A.Q:
            for a, j, w in A.arcs(i):
                U.add_arc(PairState(State(idx), i), a, PairState(State(idx), j), w)

        for q, w in A.I:
            U.set_I(PairState(State(idx), q), w)

        for q, w in A.F:
            U.set_F(PairState(State(idx), q), w)

    def union(self, A: "FSA") -> "FSA":
        """construct the union of the two FSAs"""

        assert self.R == A.R

        U = self.spawn()

        # add arcs, initial and final states from self
        self._union_add(self, U, 1)

        # add arcs, initial and final states from argument
        self._union_add(A, U, 2)

        return U

    def single_I(self):
        """Returns an equivalent FSA with only 1 initial state"""
        if len([q for q, _ in self.I]) == 1:
            return self

        # Find suitable names for the new state
        ixs = [q.idx for q in self.Q]
        start_id = 0
        while f"single_I_{start_id}" in ixs:
            start_id += 1

        F = self.spawn(keep_final=True)

        for i in self.Q:
            for a, j, w in self.arcs(i):
                F.add_arc(i, a, j, w)

        for i, w in self.I:
            F.add_arc(State(f"single_I_{start_id}"), ε, i, w)

        F.set_I(State(f"single_I_{start_id}"), F.R.one)

        return F

    def intersect(self, other: "FSA") -> "FSA":
        """This method performs an on-the-fly weighted intersection of two FSA.
        It works by keeping a stack of accessible states in the intersection WFSA.
        It uses the epsilon filter to handle epsilon transitions.

        Args:
            fsa (FSA): The other FSA to intersect with.

        Returns:
            FSA: The intersection FSA.
        """

        # the two machines need to be in the same semiring
        assert self.R == other.R

        return self.to_identity_fst().compose(other.to_identity_fst()).project(1)

    def invert(self):
        """computes inverse"""

        inv = self.spawn(keep_init=True, keep_final=True)

        for i in self.Q:
            for a, j, w in self.arcs(i):
                inv.add_arc(i, a, j, ~w)

        return inv

    def complement(self):
        """create the complement of the machine"""

        assert self.deterministic
        assert self.R == Boolean

        one = self.R.one
        tfsa = self.complete()
        nfsa = FSA(R=tfsa.R)

        for i in tfsa.Q:
            for a, j, w in tfsa.arcs(i):
                if a == ε:  # ignore epsilon
                    continue
                nfsa.add_arc(i, a, j, w)

        for q, w in self.I:
            nfsa.set_I(q, w)

        finals = set([q for q, w in tfsa.F])
        for q in tfsa.Q:
            if q not in finals:
                nfsa.set_F(q, one)

        return nfsa

    def to_identity_fst(self):
        """Method converts FSA to FST.

        Returns:
            FST: FST object that accepts X:X iff `self` accepts X.
        """
        from wlstar.base.fst import FST

        fst = FST(self.R)
        for q in self.Q:
            for a, j, w in self.arcs(q):
                fst.add_arc(i=q, a=a, b=a, j=j, w=w)
        for q, w in self.I:
            fst.set_I(q=q, w=w)
        for q, w in self.F:
            fst.set_F(q=q, w=w)
        return fst

    def __call__(self, str):
        return self.accept(str)

    def __add__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.difference(other)

    def __and__(self, other):
        return self.intersect(other)

    def __or__(self, other):
        return self.union(other)

    def __repr__(self):
        return f"WFSA({self.num_states} states, {self.R})"

    def __str__(self):
        output = []
        for q, w in self.I:
            output.append(f"initial state:\t{q.idx}\t{w}")
        for q, w in self.F:
            output.append(f"final state:\t{q.idx}\t{w}")
        for p in self.Q:
            for a, q, w in self.arcs(p):
                output.append(f"{p}\t----{a}/{w}---->\t{q}")
        return "\n".join(output)

    def __getitem__(self, n):
        return list(self.Q)[n]

    def __len__(self):
        return len(self.Q)

    def rename_states(self) -> "FSA":
        """Renames the states with names from 0 to N-1,
        where N is the number of states.
        This is useful after performing transformations which augment the state space,
        such as determinization of intersection.

        Returns:
            FSA: Strongly equivalent FSA with renamed states.
        """

        A = self.spawn()

        q2ix = {q: ix for ix, q in enumerate(self.finish())}

        for q in self.Q:
            for a, j, w in self.arcs(q):
                A.add_arc(State(q2ix[q]), a, State(q2ix[j]), w)

        for q, w in self.I:
            A.add_I(State(q2ix[q]), w)

        for q, w in self.F:
            A.add_F(State(q2ix[q]), w)

        return A

    def degrees(self, collapse_symbols: bool = True) -> Dict[State, int]:
        """Computes the out-degree of each state.

        Args:
            collapse_symbols (bool, optional): Whether to collapse the symbols.

        Returns:
            Dict[State, int]: The out-degree of each state.
        """
        if not collapse_symbols:
            return {q: len([a for a, _, _ in self.arcs(q)]) for q in self.Q}
        else:
            return {q: len(set([a for a, _, _ in self.arcs(q)])) for q in self.Q}

    def _repr_html_(self):  # noqa: C901
        """
        When returned from a Jupyter cell, this will generate the FST visualization
        Based on: https://github.com/matthewfl/openfst-wrapper
        """
        import json
        from collections import defaultdict
        from uuid import uuid4

        from wlstar.base.semiring import ProductSemiring, Real, String
        from wlstar.base.fst import FST

        def weight2str(w):
            if isinstance(w, Real):
                return f"{w.value:.3f}"
            return str(w)

        ret = []
        if self.num_states == 0:
            return "<code>Empty FST</code>"

        if self.num_states > 64:
            return (
                "FST too large to draw graphic, use fst.ascii_visualize()<br />"
                + f"<code>FST(num_states={self.num_states})</code>"
            )

        finals = {q for q, _ in self.F}
        initials = {q for q, _ in self.I}

        # print initial
        for q, w in self.I:
            if q in finals:
                label = f"{str(q)} / [{weight2str(w)} / {str(self.ρ[q])}]"
                color = "af8dc3"
            else:
                label = f"{str(q)} / {weight2str(w)}"
                color = "66c2a5"

            ret.append(
                f'g.setNode("{repr(q)}", '
                + f'{{ label: {json.dumps(label)} , shape: "circle" }});\n'
            )

            ret.append(f'g.node("{repr(q)}").style = "fill: #{color}"; \n')

        # print normal
        for q in (self.Q - finals) - initials:
            lbl = str(q)

            ret.append(
                f'g.setNode("{repr(q)}",{{label:{json.dumps(lbl)},shape:"circle"}});\n'
            )
            ret.append(f'g.node("{repr(q)}").style = "fill: #8da0cb"; \n')

        # print final
        for q, w in self.F:
            # already added
            if q in initials:
                continue

            if w == self.R.zero:
                continue
            lbl = f"{str(q)} / {weight2str(w)}"

            ret.append(
                f'g.setNode("{repr(q)}",{{label:{json.dumps(lbl)},shape:"circle"}});\n'
            )
            ret.append(f'g.node("{repr(q)}").style = "fill: #fc8d62"; \n')

        for q in self.Q:
            to = defaultdict(list)
            for a, j, w in self.arcs(q):
                if self.R is ProductSemiring and isinstance(w.value[0], String):
                    # the imporant special case of encoding transducers
                    label = f"{repr(a)}:{weight2str(w)}"
                elif isinstance(self, FST):
                    label = f"{repr(a[0])}:{repr(a[1])} / {weight2str(w)}"
                else:
                    a = str(repr(a))[1:-1]
                    label = f"{a} / {weight2str(w)}"
                to[j].append(label)

            for d, values in to.items():
                if len(values) > 6:
                    values = values[0:3] + [". . ."]
                label, qrep, drep = json.dumps("\n".join(values)), repr(q), repr(d)
                color = "rgb(192, 192, 192)" if self.theme == "dark" else "#333"
                ret.append(
                    f'g.setEdge("{qrep}","{drep}",{{arrowhead:"vee",label:{label},"style": "stroke: {color}; fill: none;", "labelStyle": "fill: {color}; stroke: {color}; ", "arrowheadStyle": "fill: {color}; stroke: {color};"}});\n'
                )

        # if the machine is too big, do not attempt to make the web browser display it
        # otherwise it ends up crashing and stuff...
        if len(ret) > 256:
            return (
                "FST too large to draw graphic, use fst.ascii_visualize()<br />"
                + f"<code>FST(num_states={self.num_states})</code>"
            )

        ret2 = [
            """
       <script>
       try {
       require.config({
       paths: {
       "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3",
       "dagreD3": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min"
       }
       });
       } catch {
       ["https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3.js",
       "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min.js"].forEach(
            function (src) {
            var tag = document.createElement('script');
            tag.src = src;
            document.body.appendChild(tag);
            }
        )
        }
        try {
        requirejs(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        try {
        require(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        </script>
        <style>
        .node rect,
        .node circle,
        .node ellipse {
        stroke: #333;
        fill: #fff;
        stroke-width: 1px;
        }

        .edgePath path {
        stroke: #333;
        fill: #333;
        stroke-width: 1.5px;
        }
        </style>
        """
        ]

        obj = "fst_" + uuid4().hex
        ret2.append(
            f'<center><svg width="850" height="600" id="{obj}"><g/></svg></center>'
        )
        ret2.append(
            """
        <script>
        (function render_d3() {
        var d3, dagreD3;
        try { // requirejs is broken on external domains
          d3 = require('d3');
          dagreD3 = require('dagreD3');
        } catch (e) {
          // for google colab
          if(typeof window.d3 !== "undefined" && typeof window.dagreD3 !== "undefined"){
            d3 = window.d3;
            dagreD3 = window.dagreD3;
          } else { // not loaded yet, so wait and try again
            setTimeout(render_d3, 50);
            return;
          }
        }
        //alert("loaded");
        var g = new dagreD3.graphlib.Graph().setGraph({ 'rankdir': 'LR' });
        """
        )
        ret2.append("".join(ret))

        ret2.append(f'var svg = d3.select("#{obj}"); \n')
        ret2.append(
            """
        var inner = svg.select("g");

        // Set up zoom support
        var zoom = d3.zoom().scaleExtent([0.3, 5]).on("zoom", function() {
        inner.attr("transform", d3.event.transform);
        });
        svg.call(zoom);

        // Create the renderer
        var render = new dagreD3.render();

        // Run the renderer. This is what draws the final graph.
        render(inner, g);

        // Center the graph
        var initialScale = 0.75;
        svg.call(zoom.transform, d3.zoomIdentity.translate(
            (svg.attr("width")-g.graph().width*initialScale)/2,20).scale(initialScale));

        svg.attr('height', g.graph().height * initialScale + 50);
        })();

        </script>
        """
        )

        return "".join(ret2)
