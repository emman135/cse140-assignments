import math
import typing

import pacai.agents.greedy
import pacai.agents.minimax
import pacai.core.action
import pacai.core.gamestate
import pacai.pacman.board
import pacai.search.distance


class ReflexAgent(pacai.agents.greedy.GreedyAgent):
    """
    A simple agent based on pacai.agents.greedy.GreedyAgent.

    You job is to make this agent better (it is pretty bad right now).
    You can change whatever you want about it,
    but it should still be a child of pacai.agents.greedy.GreedyAgent
    and be a "reflex" agent.
    This means that it shouldn't do any formal planning or searching,
    instead it should just look at the state of the game and try to make a good choice in the moment.
    You can make a great agent just by implementing a custom evaluate_state() method
    (and maybe add to the constructor if you want).
    """

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

    def evaluate_state(self,
            state: pacai.core.gamestate.GameState,
            action: pacai.core.action.Action | None = None,
            **kwargs: typing.Any) -> float:
        
        score = state.score
        pac_position = state.get_agent_position(0)

        if pac_position is None:
            return -math.inf

        # distance to nearest food
        food_positions = state.get_food()

        if len(food_positions) > 0:
            min_food_dist = float('inf')
            for food in food_positions:
                dist = pacai.search.distance.manhattan_distance(pac_position, food)
                if dist < min_food_dist:
                    min_food_dist = dist

            score += 10.0 / (min_food_dist + 1.0)
        else:
            score += 10.0

        # penalty for remaining food
        score -= 4.0 * len(food_positions)

        # ghost proximity
        nonscared_ghosts = state.get_nonscared_ghost_positions()
        for ghost_pos in nonscared_ghosts.values():
            dist = pacai.search.distance.manhattan_distance(pac_position, ghost_pos)
            if dist <= 1:
                score -= 300.0
            elif dist <= 3:
                score -= 30.0 / dist

        # chase scared ghosts
        scared_ghosts = state.get_scared_ghost_positions()
        for ghost_pos in scared_ghosts.values():
            dist = pacai.search.distance.manhattan_distance(pac_position, ghost_pos)
            score += 50.0 / (dist + 1.0)

        # capsule bonus
        capsule_positions = state.board.get_marker_positions(
            pacai.pacman.board.MARKER_CAPSULE
        )
        if len(capsule_positions) > 0:
            min_cap_dist = float('inf')
            for cap in capsule_positions:
                dist = pacai.search.distance.manhattan_distance(pac_position, cap)
                if dist < min_cap_dist:
                    min_cap_dist = dist

            score += 5.0 / (min_cap_dist + 1.0)

        # penalize revisiting recent positions
        if pac_position in self.last_positions[-4:]:
            score -= 15.0

        return score


class MyMinimaxLikeAgent(pacai.agents.minimax.MinimaxLikeAgent):
    """
    An agent that implements all the required methods for the minimax family of algorithms.
    Default implementations are supplied, so the agent should run right away,
    but it will not be very good.

    To implement minimax, minimax_step_max() and minimax_step_min() are required
    (you can ignore alpha and beta).

    To implement minimax with alpha-beta pruning,
    minimax_step_max() and minimax_step_min() with alpha and beta are required.

    To implement expectimax, minimax_step_max() and minimax_step_expected_min() are required.

    You are free to implement/override any methods you need to.
    """

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

    def minimax_step_max(self,
            state: pacai.core.gamestate.GameState,
            ply_count: int,
            legal_actions: list[pacai.core.action.Action],
            alpha: float,
            beta: float,
            ) -> tuple[list[pacai.core.action.Action], float]:
        best_score = -math.inf
        best_actions = []

        for action in legal_actions:
            successor = state.generate_successor(action, self.rng)
            _, score = self.minimax_step(successor, ply_count, alpha, beta)

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

            if self.alphabeta_prune:
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break

        return best_actions, best_score

    def minimax_step_min(self,
            state: pacai.core.gamestate.GameState,
            ply_count: int,
            legal_actions: list[pacai.core.action.Action],
            alpha: float,
            beta: float,
            ) -> tuple[list[pacai.core.action.Action], float]:
        best_score = math.inf
        best_actions = []

        for action in legal_actions:
            successor = state.generate_successor(action, self.rng)
            _, score = self.minimax_step(successor, ply_count, alpha, beta)

            if score < best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

            if self.alphabeta_prune:
                beta = min(beta, best_score)
                if beta <= alpha:
                    break

        return best_actions, best_score

    def minimax_step_expected_min(self,
            state: pacai.core.gamestate.GameState,
            ply_count: int,
            legal_actions: list[pacai.core.action.Action],
            alpha: float,
            beta: float,
            ) -> float:
        total_score = 0.0

        for action in legal_actions:
            successor = state.generate_successor(action, self.rng)
            _, score = self.minimax_step(successor, ply_count, alpha, beta)
            total_score += score

        return total_score / len(legal_actions)


def better_state_eval(
        state: pacai.core.gamestate.GameState,
        agent: typing.Any | None = None,
        action: pacai.core.action.Action | None = None,
        **kwargs: typing.Any) -> float:
    """
    Create a better state evaluation function for your MyMinimaxLikeAgent agent!

    In my new state evaluation function, I consider the following factors:
        - Current game score
        - Distance to the nearest food pellet
        - Number of remaining food pellets
        - Distance to non-scared ghosts
        - Distance to scared ghosts
        - Remaining capsules
    """
    score = state.score
    pac_position = state.get_agent_position(0)
    
    if pac_position is None:
        return -math.inf

    # distance to nearest food
    food_positions = state.get_food()
    
    if len(food_positions) > 0:
        min_food_dist = float('inf')
        for food in food_positions:
            dist = pacai.search.distance.manhattan_distance(pac_position, food)
            if dist < min_food_dist:
                min_food_dist = dist

        score += 10.0 / (min_food_dist + 1.0)
    else:
        score += 10.0

    # penalty for remaining food
    score -= 4.0 * len(food_positions)

    # non-scared ghost avoidance
    nonscared_ghosts = state.get_nonscared_ghost_positions()
    for ghost_position in nonscared_ghosts.values():
        dist = pacai.search.distance.manhattan_distance(pac_position, ghost_position)
        if dist <= 1:
            score -= 300.0
        elif dist <= 3:
            score -= 30.0 / dist

    # chase scared ghosts
    scared_ghosts = state.get_scared_ghost_positions()
    for ghost_position in scared_ghosts.values():
        dist = pacai.search.distance.manhattan_distance(pac_position, ghost_position)
        score += 50.0 / (dist + 1.0)

    # capsule proximity bonus and penalty for remaining capsules
    capsule_positions = state.board.get_marker_positions(
        pacai.pacman.board.MARKER_CAPSULE
    )

    score -= 10.0 * len(capsule_positions)

    if len(capsule_positions) > 0:
        min_cap_dist = float('inf')
        for cap in capsule_positions:
            dist = pacai.search.distance.manhattan_distance(pac_position, cap)
            if dist < min_cap_dist:
                min_cap_dist = dist

        score += 5.0 / (min_cap_dist + 1.0)

    return score
