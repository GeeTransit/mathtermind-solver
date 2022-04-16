import math
import itertools
import json
import re
import argparse
import sys
from collections import Counter, defaultdict
from typing import Optional, Any, List, Mapping, Collection

Ints = Collection[int]

DEFAULT_POOL = [
    (i, j, k)
    for i in range(1, 16)
    for j in range(i+1, 16)
    for k in range(j+1, 16)
]

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
        if nums_matching(triplet=triplet, nums=nums) == matches
    ]

def pool_counts(
    pool: Collection[Ints] = (),
    nums: Ints = (),
) -> Mapping[int, int]:
    """Return a dict with amount of matching triplets

    Arguments:
        pool: possible triplets at this point
        nums: the numbers being guessed

    Returns:
        mapping from nums matching to the amount of matching triplets

    Example:
        >>> pool = [(1, 2), (1, 3)]
        >>> pool_counts(pool=pool, nums=[1, 2])
        {2: 1, 1: 1}
        >>> pool_counts(pool=pool, nums=[2])
        {1: 1, 0: 1}
        >>> pool_counts(pool=pool, nums=[1])
        {1: 2}
        >>> pool_counts(pool=pool, nums=[4])
        {0: 2}

    """
    return dict(Counter(
        nums_matching(triplet=triplet, nums=nums)
        for triplet in pool
    ))

def pool_split(
    pool: Collection[Ints] = (),
    nums: Ints = (),
) -> Mapping[int, Collection[Ints]]:
    """Return a dict with pools for each possible nums matching

    Arguments:
        pool: possible triplets at this point
        nums: the numbers being guessed

    Returns:
        mapping from nums matching to the matching list of triplets

    Example:
        >>> pool = [(1, 2), (1, 3)]
        >>> pool_split(pool=pool, nums=[1, 2])
        {2: [(1, 2)], 1: [(1, 3)]}
        >>> pool_split(pool=pool, nums=[2])
        {1: [(1, 2)], 0: [(1, 3)]}
        >>> pool_split(pool=pool, nums=[1])
        {1: [(1, 2), (1, 3)]}
        >>> pool_split(pool=pool, nums=[4])
        {0: [(1, 2), (1, 3)]}

    """
    result = defaultdict(list)
    for triplet in pool:
        result[nums_matching(triplet=triplet, nums=nums)].append(triplet)
    return dict(result)

def entropy(probabilities: Ints = ()) -> float:
    """Return the entropy of the probability distribution

    Arguments:
        probabilities: list of probabilities

    Returns:
        entropy of the distribution

    Example:
        >>> entropy([1/2, 1/2])
        1.0
        >>> round(entropy([9/10, 1/10]), 7)
        0.4689956

    Note:
        The probabilities are normalized to sum to 1.

    """
    if not all(p > 0 for p in probabilities):
        raise ValueError("all probabilities must be positive")
    return (
        -math.fsum(p*math.log2(p) for p in probabilities)
        / sum(probabilities)
    )

def rank_guess(pool=DEFAULT_POOL, nums=()):
    new_pools = pool_counts(pool=pool, nums=nums).values()
    average_bits = entropy(new_pools)
    return 2 ** -average_bits / sum(new_pools)
    # return max(new_pools) / len(pool)
    # return sum(new_pool**2 for new_pool in new_pools) / len(pool)**2

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
    path=False,
    force_guesses=(),
):
    save_path = path
    del path
    if debug and save_path:
        raise ValueError("debug and path cannot be used together")
    # with debug=True, check for partials using (0 in p and p[0] == "partial!")
    global G
    G += 1
    # if G % 100000 == 0:
        # print(G)
    # assert levels >= 0
    if len(pool) == 1:
        if debug:
            return [f'must be {" ".join(map(str, pool[0]))}']
        if save_path:
            return {}
        return True
    if levels == 0:
        if debug:
            return [
                "partial!",
                0,
                "could be one of",
                *(" ".join(map(str, nums)) for nums in pool),
            ]
        if save_path:
            return {"partial": True}
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
        if debug or save_path:
            path = {}
            if debug:
                path_len = {}
        ok = None
        for matches, new_pool in pool_split(pool=pool, nums=nums).items():
            # new_guesses = (guesses, (nums, matches))
            # if len(new_pool) == len(pool):
                # continue
            p = curse(
                pool=new_pool,
                # guesses=new_guesses,
                levels=levels - 1,
                debug=debug,
                path=save_path,
                force_guesses=force_guesses,
            )
            if (
                not debug and not save_path and not p
                or debug and 0 in p and p[0] == "partial!"
                or save_path and p.get("partial")
            ):
                ok = False
                if not debug and not save_path:
                    break
            else:
                if ok is None:
                    ok = True
            if debug or save_path:
                if debug and 0 in p and p[0] == "partial!":
                    p = p.copy()
                    del p[1]
                    del p[0]
                path[matches] = p
                if debug:
                    path_len[matches] = len(new_pool)
        if ok:
            if debug:
                return {
                    f'guess {" ".join(map(str, nums))} ({i}={rank:.2f})': {
                        f'on {matches} ({path_len[matches] / sum(path_len.values()):.2f})': path[matches]
                        for matches in sorted(path, key=lambda matches: path_len[matches], reverse=True)
                    },
                }
            if save_path:
                return {"nums": nums, **path}
            return True
        else:
            if (debug or save_path) and best_path is None:
                best_path = rank, nums, path, path_len if debug else None, i
    if debug or save_path:
        if best_path is not None:
            rank, nums, path, path_len, i = best_path
            if save_path:
                return {"partial": True, "nums": nums, **path}
            return {
                0: "partial!",  # denote partial result
                1: "...",
                f'guess {" ".join(map(str, nums))} ({i}={rank:.2f})': {
                    f'on {matches} ({path_len[matches] / sum(path_len.values()):.2f})': path[matches]
                    for matches in sorted(path, key=lambda matches: path_len[matches], reverse=True)
                },
            }
        else:
            if save_path:
                return {"partial": True}
            return {
                0: "partial!",
                1: "...",
                f'guess 0 ({0:.2f})': {},
            }
    return None

def tree_from_path(
    path: dict = {"partial": True},
    pool: Collection[Ints] = (),
    force_guesses: Optional[Collection[Ints]] = None,
    levels: Optional[int] = None,
) -> Any:
    """Return a human readable search tree

    Arguments:
        pool: possible triplets at this point
        path: result from ``curse(..., path=True)``
        force_guesses: nums to guess at the start
        levels: depth of search

    Returns:
        human readable tree with final and partial triplets

    This is preferred over ``curse(..., debug=True)``.

    Other arguments should have the same value as the call to `curse`.

    """
    nums = path.get("nums")
    partial = path.get("partial")

    # Only the uppermost result should have "partial!"
    def _remove_partial(result):
        if 0 in result and result[0] == "partial!":
            del result[1]
            del result[0]
        return result

    if nums:
        if force_guesses is not None and force_guesses:
            assert nums == list(force_guesses)[0]
            i = 0  # Index is 0 because it's forced
        else:
            # Find the original index of the guess
            i = sorted(
                make_guesses(pool),
                key=lambda nums: (rank_guess(pool=pool, nums=nums), nums)
            ).index(nums)

        rank = rank_guess(pool=pool, nums=nums)
        splits = pool_split(pool=pool, nums=nums)

        return {
            # Denote partial result if needed
            **({
                0: "partial!",
                1: "...",
            } if partial else {}),
            # Show the numbers to guess, its index, and its rank
            f'guess {" ".join(map(str, nums))} ({i}={rank:.2f})': {
                # Show possible outcomes, their likelihood, and next steps
                f'on {matches} ({len(splits[matches]) / len(pool):.2f})': (
                    # Partial results are only at the top level
                    _remove_partial(tree_from_path(
                        pool=splits[matches],
                        path=path[matches],
                        force_guesses=(
                            list(force_guesses)[1:]
                            if force_guesses is not None
                            else None
                        ),
                        levels=(
                            levels - 1
                            if levels is not None
                            else None
                        ),
                    ))
                )
                # Most common outcome at the top
                for matches in sorted(
                    (key for key in path.keys() if isinstance(key, int)),
                    key=lambda matches: len(splits[matches]),
                    reverse=True,
                )
            },
        }

    # Only one possible triplet in the pool
    if not partial:
        assert len(pool) == 1
        return [f'must be {" ".join(map(str, list(pool)[0]))}']

    # Hit recursion limit
    if (
        # levels is specified and we've hit it
        levels is not None and levels == 0
        # Assume we hit the recursion limit if it's possible to guess
        # *something* (the triplets aren't all empty)
        or levels is None and any(pool)
    ):
        return [
            "partial!",
            0,
            "could be one of",
            *(" ".join(map(str, nums)) for nums in pool),
        ]

    # No possible guesses (such as for pool=[(),()])
    return {
        0: "partial!",
        1: "...",
        f'guess 0 ({0:.2f})': {},
    }

def tree_to_text(tree: Any) -> str:
    """Return a YAML-like version of the search tree

    Note:
        The result uses tabs for indentation, meaning changing them to spaces
        is a simple ``tree_to_text(...).replace("\t", " "*4)``.

    """
    return "\n".join(
        line[1:] if not line[:1].strip() else line  # Unindent if possible
        for line in json.dumps(tree, indent="\t").splitlines()
        for line in [re.sub(r'[][{},"]', "", line)]  # Remove most syntax
        if line.strip()  # Remove empty lines
    )

# list(curse(levels=1))

A = ok_triplets(nums=(1, 2, 3, 4), matches=1)
B = ok_triplets(pool=A, nums=(5, 6, 7, 8), matches=1)
C = ok_triplets(pool=B, nums=(9, 10, 11, 12), matches=1)
D = ok_triplets(pool=C, nums=(1, 5), matches=2)

# curse(pool=C, levels=3)

def make_parser() -> argparse.ArgumentParser:
    """Return CLI argument parser"""
    parser = argparse.ArgumentParser(description="Solve a Mathtermind game")
    parser.add_argument(
        "-g", "--guessed", action="append",
        help="a guess of the form #[,#]*:R where R is nums matching",
    )
    parser.add_argument(
        "--force-guess", action="append",
        help="a guess to force of the form #[,#]*",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="return a JSON of the paths taken",
    )
    parser.add_argument(
        "--levels", default="1:9",
        help="the range of levels to try (of the form #:# or #)",
    )
    return parser

def main(argv: Optional[List[str]] = None):
    """Entry point to Mathtermind solver"""
    args = make_parser().parse_args(argv)
    pool = DEFAULT_POOL
    if args.guessed is not None:
        for guessed in args.guessed:
            nums, matches = guessed.split(":")
            matches = int(matches)
            nums = [int(x) for x in nums.split(",")]
            pool = ok_triplets(pool=pool, nums=nums, matches=matches)
    force_guesses = []
    if args.force_guess is not None:
        for guess in args.force_guess:
            force_guesses.append([int(x) for x in guess.split(",")])
    if ":" in args.levels:
        start, stop = [int(x) for x in args.levels.split(":")]
    else:
        start = int(args.levels)
        stop = start
    for levels in range(start, stop + 1):
        if curse(pool=pool, levels=levels):
            break
    else:
        levels = start  # give best guess if not possible
    path = curse(pool=pool, levels=levels, force_guesses=force_guesses, path=True)
    if args.json:
        print(json.dumps(path))
    else:
        print(tree_to_text(tree_from_path(
            path=path,
            pool=pool,
            levels=levels,
            force_guesses=force_guesses,
        )))
    if path.get("partial"):
        sys.exit(1)

if __name__ == "__main__":
    main()

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
