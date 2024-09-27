import random
import numpy as np
from collections import defaultdict as dd
from itertools import product

class PartitionRefinement:

	def __init__(self, f, Q):
	
		self.f = f
		self.Q = Q

		# compute the pre-image of f
		self.finv = dd(lambda : set([]))
		for n in Q: self.finv[self.f[n]].add(n)

	def stable(self, P):

		# definition of stable
		D = {}
		for n, B in enumerate(P):
			for q in B:
				D[q] = n

		for B in P:
			for p in B:
				for q in B:
					if D[self.f[p]] != D[self.f[q]]:
						return False
		return True

	def split(self, S, P):
		""" runs in O(|P|) time if Python is clever """
		return frozenset(P&S), frozenset(P-S)
	
	def hopcroft_fast(self, P):

		P = list(map(set, P))
		N = len(P)
		stack = list(zip(P, range(len(P))))

		inblock = { b : (B, idx) for B, idx in stack for b in B }

		while stack: # empties in O(log |Q|) steps			
			(S, idx) = stack.pop()

			# computes subset of the pre-image
			# O(|Sinv|) time
			Sinv = set([]).union(*[self.finv[x] for x in S])

			# O(|Sinv|) time
			lst = [(inblock[s]) + (s,) for s in Sinv]

			# O(|Sinv|) time
			count = dd(lambda : 0)
			for _, idx, _ in lst:
				count[idx] += 1

			# excludes the case where a block B is a subset of Sinv
			# O(|Sinv|) time
			covered = set([ idx for B, idx, _ in lst if len(B) == count[idx]])

			# O(|Sinv|) time
			Rs = dd(lambda : set([]))
			for (B, idx, s) in lst:
				if idx in covered:
					continue
				B.remove(s)
				Rs[idx].add(s)

			# O(|Sinv|) time
			for idx, R in Rs.items():				
				for u in R:
					inblock[u] = (R, N)
				stack.append((R, N))
				P.append(R)
				N += 1

		P = frozenset(map(frozenset, P))
		return P
