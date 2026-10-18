"""
Makerere Campus Route Finder (CSC 2114 Assignment 3: weighted graph, UCS, IDS)

ONE generic search loop (search) serves BFS, DFS, UCS and, through a driver
loop (ids), IDS. The algorithms differ only in:
    * the frontier object   (QueueFrontier / StackFrontier / PriorityFrontier)
    * the rule that decides whether a child replaces the node in `reached`
    * the optional depth limit and cycle check (used by IDS)

Edge weights are walking distances in metres. They are read only by
step_cost(), by UCS's priority queue (through PATH-COST) and by route_cost().
BFS, DFS and IDS never use them to decide anything.

Submit as <group_number-Day A/B/Eve>_route_finder.py

Usage:
    python route_finder.py [--noreached] [--trace] [--verbose]
        --noreached  tree-like search for BFS, DFS, UCS (10,000-expansion cap)
                     (--no-reached is accepted as a synonym)
        --trace      UCS: print every push, removal, stale discard, replacement
        --verbose    also print Max frontier size (and IDS per-iteration counts)
"""
import heapq
import itertools
import sys
from collections import deque, namedtuple

WEIGHTED_EDGES = [
    ("Main Building", "Freedom Square", 120),
    ("Main Building", "Main Library", 150),
    ("Main Building", "School of Social Sciences", 200),
    ("Main Building", "School of Liberal and Performing Arts", 350),
    ("Freedom Square", "Main Library", 90),
    ("Freedom Square", "School of Law", 180),
    ("Freedom Square", "Central Teaching Facility", 220),
    ("Main Library", "School of Education", 160),
    ("Main Library", "School of Economics", 400),
    ("Central Teaching Facility", "CoCIS", 130),
    ("Central Teaching Facility", "School of Statistics and Planning", 300),
    ("CoCIS", "School of Statistics and Planning", 100),
    ("CoCIS", "School of Engineering", 250),
    ("School of Engineering", "School of the Built Environment", 90),
    ("School of Engineering", "Margaret Trowell School of Industrial and Fine Art", 140),
    ("School of the Built Environment", "Margaret Trowell School of Industrial and Fine Art", 70),
    ("Margaret Trowell School of Industrial and Fine Art", "School of Liberal and Performing Arts", 450),
    ("School of Education", "School of Social Sciences", 140),
    ("School of Education", "College of Natural Sciences", 300),
    ("College of Natural Sciences", "School of Food Technology, Nutrition and Bioengineering", 380),
    ("School of Food Technology, Nutrition and Bioengineering", "School of Agricultural Sciences", 200),
    ("School of Agricultural Sciences", "School of Veterinary Medicine", 260),
    ("School of Statistics and Planning", "School of Economics", 350),
    ("School of Economics", "School of Law", 120),
]

# Landmarks can be walked through but are never a source or destination.
LANDMARKS = {"Main Building", "Freedom Square", "Main Library",
             "Central Teaching Facility"}

EXPANSION_CAP = 10_000
ALGORITHMS = ("BFS", "DFS", "UCS", "IDS")

# Outcome of one call to search()
SOLUTION, FAILURE, CUTOFF, CAP = "solution", "failure", "cutoff", "cap"
SearchResult = namedtuple("SearchResult", "node expanded status max_frontier")


# ------------------------------------------------------------------ graph
def build_graph(edges):
    # creating a graph that that has front and back connections of places
    graph = {}
    for a, b, w in edges:
        graph.setdefault(a, {})[b] = w
        graph.setdefault(b, {})[a] = w
    return graph


def neighbours(graph, s):
    # .items returns a set of key values and they are sorted
    return sorted(graph[s].items())


def step_cost(graph, s, s2):
    # get the distance between two places
    return graph[s][s2]


def route_cost(graph, route):
    # some all the costs along a desired route
    return sum(graph[a][b] for a, b in zip(route, route[1:]))


# ------------------------------------------------------------------- node
class Node:
    """Search-tree node: the four lecture fields plus DEPTH."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state          # STATE
        self.parent = parent        # PARENT
        self.action = action        # ACTION (the place we walked to)
        self.path_cost = path_cost  # PATH-COST: metres from the root
        self.depth = 0 if parent is None else parent.depth + 1  # hops


def solution(node):
    """Follow PARENT pointers back to the root; return places in order."""
    route = []
    while node is not None:
        route.append(node.state)
        node = node.parent
    return route[::-1]


def on_path(node, state):
    #checking what has been visited on a specific path
    while node is not None:
        if node.state == state:
            return True
        node = node.parent
    return False


# -------------------------------------------------------------- frontiers
class QueueFrontier:
    """FIFO queue (BFS)."""

    def __init__(self):
        self._items = deque()

    def push(self, node):
        self._items.append(node)

    def pop(self):
        return self._items.popleft()

    def is_empty(self):
        return not self._items

    def __len__(self):
        return len(self._items)

    # push_order() is a small extra on the frontier interface: it lets the
    # generic loop stay free of "if algorithm == ..." branches while the
    # stack still receives children in reverse alphabetical order.
    def push_order(self, children):
        return children             # alphabetical first = removed first


class StackFrontier:
    """LIFO stack (DFS, IDS)."""

    def __init__(self):
        self._items = []

    def push(self, node):
        self._items.append(node)

    def pop(self):
        return self._items.pop()

    def is_empty(self):
        return not self._items

    def __len__(self):
        return len(self._items)

    def push_order(self, children):
        return reversed(children)   # so the alphabetically first pops first


class PriorityFrontier:
    """Binary heap ordered by PATH-COST (UCS).
    Entry = (path_cost, state_name, counter, node): lowest cost first, then
    alphabetically first state, then earliest inserted. The counter also
    stops Python from ever comparing two Node objects."""

    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def push(self, node):
        heapq.heappush(self._heap,
                       (node.path_cost, node.state, next(self._counter), node))

    def pop(self):
        return heapq.heappop(self._heap)[3]

    def is_empty(self):
        return not self._heap

    def __len__(self):
        return len(self._heap)

    def push_order(self, children):
        return children


# ---------------------------------------------------- reached-update rules
def never_replace(new, old):
    """BFS / DFS / IDS: the first node that reached a state is kept."""
    return False


def replace_if_cheaper(new, old):
    """UCS: a strictly cheaper path to a state replaces the stored node."""
    return new.path_cost < old.path_cost


# ---------------------------------------------------- the generic search
def search(graph, source, goal, frontier, replace_rule, use_reached=True,
           cycle_check=False, depth_limit=None, cap=EXPANSION_CAP, log=None):
    """
    The generic loop. The frontier object, the reached-update rule,
    the cycle check and the depth limit are the only things that change
    between algorithms; the loop has no 'if algorithm == ...' branch.
    Returns SearchResult(node, expanded, status, max_frontier).
    """
    root = Node(source)
    frontier.push(root)
    reached = {source: root} if use_reached else None
    expanded = 0
    max_frontier = len(frontier)
    cutoff = False

    while not frontier.is_empty():
        node = frontier.pop()                                  # remove

        # A cheaper node may have replaced this one in reached: discard it.
        if use_reached and reached[node.state] is not node:
            if log:
                log(f"discard stale {node.state} g={node.path_cost} "
                    f"(reached holds g={reached[node.state].path_cost})")
            continue

        if log:
            log(f"remove {node.state} g={node.path_cost}")

        if node.state == goal:                    # goal test on removal
            return SearchResult(node, expanded, SOLUTION, max_frontier)

        if depth_limit is not None and node.depth >= depth_limit:
            # Cutoff only if the limit really hid something: a node at the
            # limit with no unvisited neighbour is a genuine dead end.
            if any(not on_path(node, p) for p, _ in neighbours(graph, node.state)):
                cutoff = True
            continue

        if cap is not None and expanded >= cap:
            return SearchResult(None, expanded, CAP, max_frontier)
        expanded += 1                                          # expand

        children = []
        for place, _ in neighbours(graph, node.state):
            if cycle_check and on_path(node, place):
                continue                       # refuse to go round a cycle
            child = Node(place, node, place,
                         node.path_cost + step_cost(graph, node.state, place))
            if use_reached and place in reached:
                old = reached[place]
                if not replace_rule(child, old):
                    continue
                if log:
                    log(f"reached replace {place}: g={old.path_cost} -> "
                        f"g={child.path_cost}")
            children.append(child)

        for child in frontier.push_order(children):
            if use_reached:
                reached[child.state] = child
            frontier.push(child)
            if log:
                log(f"push {child.state} g={child.path_cost} "
                    f"(from {node.state})")
            max_frontier = max(max_frontier, len(frontier))

    status = CUTOFF if cutoff else FAILURE
    return SearchResult(None, expanded, status, max_frontier)


# ------------------------------------------------------------- IDS driver
def ids(graph, source, goal):
    """
    Iterative deepening: depth_limit = 0, 1, 2, ... over the SAME generic
    search (stack frontier, reached off, cycle check on).
    Returns (SearchResult with totals, final depth limit, [(limit, expanded)]).
    """
    total, max_frontier, per_iteration = 0, 0, []
    for limit in range(len(graph)):          # never above places - 1
        res = search(graph, source, goal, StackFrontier(), never_replace,
                     use_reached=False, cycle_check=True,
                     depth_limit=limit, cap=None)
        total += res.expanded
        max_frontier = max(max_frontier, res.max_frontier)
        per_iteration.append((limit, res.expanded))
        if res.status in (SOLUTION, FAILURE):
            return (SearchResult(res.node, total, res.status, max_frontier),
                    limit, per_iteration)
    return (SearchResult(None, total, FAILURE, max_frontier),
            len(graph) - 1, per_iteration)


# ------------------------------------------------------------ run helper
def run_algorithm(graph, algorithm, source, goal, use_reached=True,
                  trace=False):
    """Run one algorithm; return a dict describing the outcome."""
    lines = []
    log = lines.append if (trace and algorithm == "UCS") else None
    final_limit, iterations = None, []

    if algorithm == "IDS":
        res, final_limit, iterations = ids(graph, source, goal)
    else:
        frontier, rule = {"BFS": (QueueFrontier(), never_replace),
                          "DFS": (StackFrontier(), never_replace),
                          "UCS": (PriorityFrontier(), replace_if_cheaper)
                          }[algorithm]
        res = search(graph, source, goal, frontier, rule,
                     use_reached=use_reached, log=log)

    out = {"algorithm": algorithm, "status": res.status,
           "expanded": res.expanded, "max_frontier": res.max_frontier,
           "route": None, "cost": None, "hops": None,
           "final_limit": final_limit, "iterations": iterations,
           "trace": lines}
    if res.node is not None:
        out["route"] = solution(res.node)
        out["cost"] = res.node.path_cost
        out["hops"] = res.node.depth
        assert out["cost"] == route_cost(graph, out["route"]), \
            "path-cost bug: printed cost differs from route_cost()"
    return out


# --------------------------------------------------------------- output
def print_result(out, verbose=False, trace=False):
    if trace and out["trace"]:
        for line in out["trace"]:
            print(f"[trace] {line}")
        print()
    print(f"Algorithm: {out['algorithm']}")
    if out["status"] == CAP:
        print(f"Search stopped: expansion cap ({EXPANSION_CAP}) reached.")
    elif out["route"] is None:
        print("No route found.")
    else:
        print(f"Route: {' -> '.join(out['route'])}")
        print(f"Cost (metres): {out['cost']}")
        print(f"Hops: {out['hops']}")
        print(f"Nodes expanded: {out['expanded']}")
        if out["algorithm"] == "IDS":
            print(f"Final depth limit: {out['final_limit']}")
    if verbose:
        print(f"Max frontier size: {out['max_frontier']}")
        if out["algorithm"] == "IDS":
            for limit, count in out["iterations"]:
                print(f"limit {limit}: expanded {count}")


# ------------------------------------------------------------ interface
def show_places(schools):
    print("Valid places (schools): type the number or the name")
    for number, name in enumerate(schools, start=1):
        print(f"  {number:>2}. {name}")


def find_place(text, schools):
    # Match a menu number or a name (any case, surrounding spaces ignored).
    text = text.strip()
    if text.isdigit():
        number = int(text)
        return schools[number - 1] if 1 <= number <= len(schools) else None
    lookup = {name.lower(): name for name in schools}
    return lookup.get(text.lower())


def ask_place(prompt, schools):
    text = input(prompt)
    place = find_place(text, schools)
    if place is None:
        print(f'\nUnknown place: "{text.strip()}".')
        show_places(schools)
    return place


def main():
    flags = set(sys.argv[1:])
    use_reached = not (flags & {"--noreached", "--no-reached"})
    trace, verbose = "--trace" in flags, "--verbose" in flags

    graph = build_graph(WEIGHTED_EDGES)
    schools = sorted(name for name in graph if name not in LANDMARKS)

    print("Makerere Campus Route Finder (weighted)")
    print("---------------------------------------")
    show_places(schools)
    print()

    try:
        source = ask_place("Source: ", schools)
        if source is None:
            return
        destination = ask_place("Destination: ", schools)
        if destination is None:
            return
        choice = input("Algorithm (BFS, DFS, UCS, IDS or ALL): ").strip().upper()
        if choice != "ALL" and choice not in ALGORITHMS:
            print(f'\nUnknown algorithm: "{choice}". '
                  "Valid algorithms: BFS, DFS, UCS, IDS, ALL.")
            return
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return

    print()
    chosen = ALGORITHMS if choice == "ALL" else (choice,)
    for i, algorithm in enumerate(chosen):
        if i:
            print()
        out = run_algorithm(graph, algorithm, source, destination,
                            use_reached, trace)
        print_result(out, verbose, trace)


if __name__ == "__main__":
    main()

