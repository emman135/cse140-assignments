"""
In this file, you will implement code relating to simple single-agent searches.
"""

import random
import typing

import pacai.agents.searchproblem
import pacai.core.agent
import pacai.core.agentaction
import pacai.core.board
import pacai.core.gamestate
import pacai.core.search
import pacai.pacman.board
import pacai.search.common
import pacai.search.food
import pacai.search.position
import pacai.util.alias
import pacai.util.containers

def depth_first_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements depth first search (DFS).
    This means that it will search the deepest nodes in the search tree first.
    See: https://en.wikipedia.org/wiki/Depth-first_search .
    """

    # Graph search version of DFS - uses stack (LIFO) to explore deepest nodes first
    fringe = pacai.util.containers.Stack()
    
    # Track which nodes already seen to avoid infinite loops
    visited = set()
    
    start_node = problem.get_starting_node()
    actions_start = []
    cost_start = 0.0
    fringe.push((start_node, actions_start, cost_start))
    
    while not fringe.is_empty():
        node, actions, cost = fringe.pop()
        if problem.is_goal_node(node):
            return pacai.core.search.SearchSolution(actions, cost, node)
        if node in visited:
            continue
        visited.add(node)
        successors = problem.get_successor_nodes(node)

        for successor in successors:
            if successor.node not in visited:
                new_actions = actions + [successor.action]
                new_cost = cost + successor.cost
                fringe.push((successor.node, new_actions, new_cost))
    
    # Didn't find a goal
    raise pacai.core.search.SolutionNotFoundError()

def breadth_first_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements breadth first search (BFS).
    This means that it will search nodes based on what level in search tree they appear.
    See: https://en.wikipedia.org/wiki/Breadth-first_search .
    """

    # Graph search version of BFS - uses queue (FIFO) to explore level by level
    fringe = pacai.util.containers.Queue()
    
    # Track which nodes we've already seen to avoid infinite loops
    visited = set()
    
    start_node = problem.get_starting_node()
    actions_start = []
    cost_start = 0.0
    fringe.push((start_node, actions_start, cost_start))
    
    while not fringe.is_empty():
        node, actions, cost = fringe.pop()
        if problem.is_goal_node(node):
            return pacai.core.search.SearchSolution(actions, cost, node)
        if node in visited:
            continue
        visited.add(node)
        successors = problem.get_successor_nodes(node)
        
        for successor in successors:
            if successor.node not in visited:
                new_actions = actions + [successor.action]
                new_cost = cost + successor.cost
                fringe.push((successor.node, new_actions, new_cost))
    
    # Didn't find a goal
    raise pacai.core.search.SolutionNotFoundError()

def uniform_cost_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements uniform cost search (UCS).
    This means that it will search nodes with a lower total cost first.
    See: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Practical_optimizations_and_infinite_graphs .
    """

    # Graph search version of UCS - uses priority queue with cost as priority. Track best cost to each node
    fringe = pacai.util.containers.PriorityQueue()

    # Track which nodes already seen to avoid infinite loops# Track which nodes we've already explored
    visited = set()
    
    # Track best (cheapest) cost found to reach each node
    best_costs = {}
    start_node = problem.get_starting_node()
    actions_start = []
    cost_start = 0.0
    
    # Lower priority = explored first, use cost as priority
    fringe.push((start_node, actions_start, cost_start), cost_start)
    best_costs[start_node] = cost_start
    
    while not fringe.is_empty():
        node, actions, cost = fringe.pop()
        if problem.is_goal_node(node):
            return pacai.core.search.SearchSolution(actions, cost, node)
        if node in visited:
            continue
        visited.add(node)
        successors = problem.get_successor_nodes(node)

        for successor in successors:
            new_cost = cost + successor.cost
            if successor.node not in visited:
                if successor.node not in best_costs or new_cost < best_costs[successor.node]:
                    best_costs[successor.node] = new_cost
                    new_actions = actions + [successor.action]
                    fringe.push((successor.node, new_actions, new_cost), new_cost)
    
    # Didn't find a goal
    raise pacai.core.search.SolutionNotFoundError()

def astar_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements A* search (pronounced "A Star search").
    This means that it will search nodes with a lower combined cost and heuristic first.
    See: https://en.wikipedia.org/wiki/A*_search_algorithm .
    """
    fringe = pacai.util.containers.PriorityQueue()

    # Track which nodes already seen to avoid infinite loops
    visited = set()
    
    # Track best (cheapest) cost found to reach each node
    best_costs = {}
    
    start_node = problem.get_starting_node()
    start_cost = 0.0
    h_start = heuristic(start_node, problem)
    fringe.push((start_node, [], start_cost), start_cost + h_start)
    best_costs[start_node] = start_cost
    
    while not fringe.is_empty():
        node, actions, cost = fringe.pop()
        if problem.is_goal_node(node):
            return pacai.core.search.SearchSolution(actions, cost, node)
        if node in visited:
            continue
        visited.add(node)
        successors = problem.get_successor_nodes(node)

        for successor in successors:
            new_cost = cost + successor.cost
            if successor.node not in visited:
                if successor.node not in best_costs or new_cost < best_costs[successor.node]:
                    best_costs[successor.node] = new_cost
                    new_actions = actions + [successor.action]
                    new_h = heuristic(successor.node, problem)
                    f_n = new_cost + new_h
                    fringe.push((successor.node, new_actions, new_cost), f_n)
    
    # Didn't find a goal
    raise pacai.core.search.SolutionNotFoundError()

class CornersSearchNode(pacai.core.search.SearchNode):
    """
    A search node the can be used to represent the corners search problem.

    You get to implement this search node however you want.
    """
    def __init__(self, position: pacai.core.board.Position, visited_corners: tuple[pacai.core.board.Position, ...]) -> None:
        """ Construct a search node to help search for corners. """

        self.position: pacai.core.board.Position = position
        self.visited_corners: tuple[pacai.core.board.Position, ...] = tuple(sorted(visited_corners))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CornersSearchNode):
            return False
        return (self.position == other.position and self.visited_corners == other.visited_corners)
    
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CornersSearchNode):
            return False
        return ((self.position, self.visited_corners) < (other.position, other.visited_corners))
    
    def __hash__(self) -> int:
        return hash((self.position, self.visited_corners))

class CornersSearchProblem(pacai.core.search.SearchProblem[CornersSearchNode]):
    """
    A search problem for touching the four different corners in a board.

    You may assume that very board is surrounded by walls (e.g., (0, 0) is a wall),
    and that the position diagonally inside from the walled corner is the location we are looking for.
    For example, if we had a square board that was 10x10, then we would be looking for the following corners:
     - (1, 1) -- North-West / Upper Left
     - (1, 8) -- North-East / Upper Right
     - (8, 1) -- South-West / Lower Left
     - (8, 8) -- South-East / Lower Right
    """
    def __init__(self,
            game_state: pacai.core.gamestate.GameState,
            **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        self.board: pacai.core.board.Board = game_state.board
        
        corners = self.board.get_corners(offset=1)
        self.corners: tuple[pacai.core.board.Position, ...] = corners
        
        start_position = game_state.get_agent_position()
        if start_position is None:
            raise ValueError("Cannot find starting position.")
        self.start_position: pacai.core.board.Position = start_position

    def get_starting_node(self) -> CornersSearchNode:
        visited_corners = ()
        if self.start_position in self.corners:
            visited_corners = (self.start_position,)
        return CornersSearchNode(self.start_position, visited_corners)

    def is_goal_node(self, node: CornersSearchNode) -> bool:
        all_corners_visited = len(node.visited_corners)
        return all_corners_visited == 4

    def get_successor_nodes(self, node: CornersSearchNode) -> list[pacai.core.search.SuccessorInfo]:
        successors = []
        current_position = node.position
        current_visited_corners = node.visited_corners
        
        for (action, neighbor_position) in self.board.get_neighbors(current_position):
            new_visited_corners = list(current_visited_corners)
            
            if neighbor_position in self.corners:
                if neighbor_position not in new_visited_corners:
                    new_visited_corners.append(neighbor_position)
            
            next_node = CornersSearchNode(neighbor_position, tuple(new_visited_corners))
            successors.append(pacai.core.search.SuccessorInfo(next_node, action, 1.0))
        
        self.expanded_node_count += 1
        
        if node not in self.visited_nodes:
            self.position_history.append(node.position)
        
        return successors

def corners_heuristic(node: CornersSearchNode, problem: CornersSearchProblem, **kwargs: typing.Any) -> float:
    """
    A heuristic for CornersSearchProblem.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """
    # Get unvisited corners
    unvisited = [c for c in problem.corners if c not in node.visited_corners]
    
    # If no corners left, we're done
    if not unvisited:
        return 0.0
    
    # Return distance to FARTHEST unvisited corner (admissible because you must visit ALL)
    return float(max(
        abs(node.position.row - c.row) + abs(node.position.col - c.col)
        for c in unvisited
    ))

def food_heuristic(node: pacai.search.food.FoodSearchNode, problem: pacai.search.food.FoodSearchProblem, **kwargs: typing.Any) -> float:
    """
    A heuristic for the FoodSearchProblem.
    """
    # If no food left, we're done
    if not node.remaining_food:
        return 0.0
    
    food_list = list(node.remaining_food)
    n = len(food_list)
    
    # Distance to nearest food
    min_dist = min(
        abs(node.position.row - f.row) + abs(node.position.col - f.col)
        for f in food_list
    )
    
    if n == 1:
        return float(min_dist)
    
    # MST cost using Prim's algorithm
    in_mst = [False] * n
    min_edge = [float('inf')] * n
    min_edge[0] = 0
    mst_cost = 0
    
    for _ in range(n):
        u = -1
        for i in range(n):
            if not in_mst[i] and (u == -1 or min_edge[i] < min_edge[u]):
                u = i
        in_mst[u] = True
        mst_cost += min_edge[u]
        for v in range(n):
            if not in_mst[v]:
                dist = abs(food_list[u].row - food_list[v].row) + abs(food_list[u].col - food_list[v].col)
                if dist < min_edge[v]:
                    min_edge[v] = dist
    
    return float(min_dist + mst_cost)

class ClosestDotSearchAgent(pacai.agents.searchproblem.GreedySubproblemSearchAgent):
    """
    Search for a path to all the food by greedily searching for the next closest food again and again
    (util we have reached all the food).

    This agent is left to you to fill out.
    But make sure to take your time and think.
    The final solution is quite simple if you take your time to understand everything up until this point and leverage
    pacai.agents.searchproblem.GreedySubproblemSearchAgent and pacai.student.problem.AnyMarkerSearchProblem.
    pacai.agents.searchproblem.GreedySubproblemSearchAgent is already implemented,
    but you should take some time to understand it.
    pacai.student.problem.AnyMarkerSearchProblem (below in this file) has not yet been implemented,
    but is the quickest and easiest way to implement this class.

    Hint:
    Remember that you can call a parent class' `__init__()` method from a child class' `__init__()` method.
    (See pacai.student.problem.AnyMarkerSearchProblem for an example.)
    Child classes will generally always call their parent's `__init__()` method
    (if a child class does not implement `__init__()`, then the parent's `__init__()` is automatically called).
    This call does not need to be the first line in the method,
    and you can pass whatever you want to the parent's `__init__()`.
    """
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(
            problem = AnyMarkerSearchProblem,
            solver = pacai.util.alias.SEARCH_SOLVER_BFS.long,
            **kwargs
        )

class AnyMarkerSearchProblem(pacai.search.position.PositionSearchProblem):
    """
    A search problem for finding a path to any instance of the specified board marker (e.g., food, wall, power capsule).

    This search problem is just like the pacai.search.position.PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    You may modify the `__init__()` if you want, the other methods should be fine as-is.
    """
    def __init__(self,
            game_state: pacai.core.gamestate.GameState,
            target_marker: pacai.core.board.Marker = pacai.pacman.board.MARKER_PELLET,
            **kwargs: typing.Any) -> None:
        super().__init__(game_state, **kwargs)

        self.target_marker: pacai.core.board.Marker = target_marker

    def is_goal_node(self, node: pacai.search.position.PositionSearchNode) -> bool:
        markers_at_position = self.board.get(node.position)
        return self.target_marker in markers_at_position

class ApproximateSearchAgent(pacai.agents.searchproblem.GreedySubproblemSearchAgent):
    """
    A search agent that tries to perform an approximate search instead of an exact one.
    In other words, this agent is okay with a solution that is "good enough" and not necessarily optimal.
    """
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(
            problem = AnyMarkerSearchProblem,
            solver = pacai.util.alias.SEARCH_SOLVER_ASTAR.long,
            heuristic = pacai.util.alias.HEURISTIC_MANHATTAN.long,
            **kwargs
        )
