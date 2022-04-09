from typing import Collection

Ints = Collection[int]

DEFAULT_POOL = [
    (i, j, k)
    for i in range(1, 16)
    for j in range(i+1, 16)
    for k in range(j+1, 16)
]

def eliminate(pool=DEFAULT_POOL, guesses=()):
    raise RuntimeError
    return {
        triplet
        for triplet in pool
        if any(
            True
            for nums, matches in guesses
            if sum(1 for num in triplet if num in nums) != matches
        )
    }

def nums_matching(triplet: Ints = (), nums: Ints = ()):
    """Return nums matching the given triplet

    Arguments:
        triplet: the possible triplet to match against
        nums: the numbers being guessed

    Returns:
        amount of triplet numbers matching the guess numbers

    Example:
        >>> nums_matching(triplet=[1, 2, 3], nums=[1])
        1
        >>> nums_matching(triplet=[1, 2, 3], nums=[1, 2])
        2
        >>> nums_matching(triplet=[1, 2, 3], nums=[1, 2, 3])
        3
        >>> nums_matching(triplet=[1, 2, 3], nums=[1, 4])
        1
        >>> nums_matching(triplet=[1, 2, 3], nums=[4, 5])
        0

    """
    return sum(num in nums for num in triplet)

def ok_triplets(pool=DEFAULT_POOL, nums=(), matches=0):
    return [
        triplet
        for triplet in pool
        if sum(1 for num in triplet if num in nums) == matches
    ]

import math

def rank_guess(pool=DEFAULT_POOL, nums=()):
    N = len(nums)
    new_pools = [0] * (N + 1)
    for triplet in pool:
        new_pools[sum(1 for num in triplet if num in nums)] += 1
    average_bits = sum(
        new_pool * math.log(new_pool, 2)
        for new_pool in new_pools
        if new_pool
    ) / len(pool)
    return 2**average_bits / len(pool)
    # return max(new_pools) / len(pool)
    # return sum(new_pool**2 for new_pool in new_pools) / len(pool)**2

import itertools

def make_guesses(pool=DEFAULT_POOL):
    nums = sorted({num for triplet in pool for num in triplet})
    return itertools.chain.from_iterable(
        itertools.combinations(nums, r)
        for r in [4, 3, 2, 1]
    )

G = 0

def curse(
    pool=DEFAULT_POOL,
    # guesses=(),
    levels=0,
    debug=False,
    force_guesses=(),
):
    # with debug=True, check for partials using (0 in p and p[0] == "partial!")
    global G
    G += 1
    # if G % 100000 == 0:
        # print(G)
    # assert levels >= 0
    if len(pool) == 1:
        if debug:
            return [f'must be {" ".join(map(str, pool[0]))}']
        return True
    if levels == 0:
        if len(pool) == 1:
            if debug:
                return [f'must be {" ".join(map(str, pool[0]))}']
            return True
        else:
            if debug:
                return [
                    "partial!",
                    0,
                    "could be one of",
                    *(" ".join(map(str, nums)) for nums in pool),
                ]
            return None
    if force_guesses:
        nums = force_guesses[0]
        force_guesses = force_guesses[1:]
        guesses = [(rank_guess(pool=pool, nums=nums), nums)]
    else:
        guesses = sorted(
            (rank_guess(pool=pool, nums=nums), nums)
            for nums in make_guesses(pool)
        )
    best_path = None
    for i, (rank, nums) in enumerate(guesses):
        if i >= 1 and rank >= 0.55:
            break
        if i >= 6:
            break
        if debug:
            path = {}
            path_len = {}
        ok = None
        for matches in range(0, len(nums) + 1):
            # new_guesses = (guesses, (nums, matches))
            new_pool = ok_triplets(pool=pool, nums=nums, matches=matches)
            # if len(new_pool) == len(pool):
                # continue
            if len(new_pool) == 0:
                continue
            if len(new_pool) == 1:
                if ok is None:
                    ok = True
                if debug:
                    path[matches] = [f'must be {" ".join(map(str, new_pool[0]))}']
                    path_len[matches] = 1
                continue
            p = curse(
                pool=new_pool,
                # guesses=new_guesses,
                levels=levels - 1,
                debug=debug,
                force_guesses=force_guesses,
            )
            if not debug and not p or debug and 0 in p and p[0] == "partial!":
                ok = False
                if not debug:
                    break
            else:
                if ok is None:
                    ok = True
            if debug:
                if 0 in p and p[0] == "partial!":
                    p = p.copy()
                    del p[1]
                    del p[0]
                path[matches] = p
                path_len[matches] = len(new_pool)
        if ok:
            if debug:
                return {
                    f'guess {" ".join(map(str, nums))} ({i}={rank:.2f})': {
                        f'on {matches} ({path_len[matches] / sum(path_len.values()):.2f})': path[matches]
                        for matches in sorted(path, key=lambda matches: path_len[matches], reverse=True)
                    },
                }
            return True
        else:
            if debug and best_path is None:
                best_path = rank, nums, path, path_len, i
    if debug:
        if best_path is not None:
            rank, nums, path, path_len, i = best_path
            return {
                0: "partial!",  # denote partial result
                1: "...",
                f'guess {" ".join(map(str, nums))} ({i}={rank:.2f})': {
                    f'on {matches} ({path_len[matches] / sum(path_len.values()):.2f})': path[matches]
                    for matches in sorted(path, key=lambda matches: path_len[matches], reverse=True)
                },
            }
        else:
            return {
                0: "partial!",
                1: "...",
                f'guess 0 ({0:.2f})': {},
            }
    return None

# list(curse(levels=1))

A = ok_triplets(nums=(1, 2, 3, 4), matches=1)
B = ok_triplets(pool=A, nums=(5, 6, 7, 8), matches=1)
C = ok_triplets(pool=B, nums=(9, 10, 11, 12), matches=1)
D = ok_triplets(pool=C, nums=(1, 5), matches=2)

# curse(pool=C, levels=3)

'''
paste this line lol:
import re,json; print("\n".join(line[2:] for line in re.sub(r'[][{},"]', '', json.dumps(curse(levels=7, debug=1), indent=2)).splitlines() if line.strip()))

proves mtm is solvable (start with 1-4, 5-8, and 9-12):
{(i,j,k):curse(ok_triplets(ok_triplets(ok_triplets(nums=[1,2,3,4],matches=i),nums=[5,6,7,8],matches=j),nums=[9,10,11,12],matches=k),levels=4) for i in range(0,4) for j in range(0,4-i) for k in range(0,4-i-j)}

old thingy:
import time; start = time.time(); curse(pool=C, levels=3) and None; end = time.time(); end - start

simple test:
curse(pool=[(i, j) for i in range(1, 9) for j in range(i+1, 9)], levels=4)
'''
