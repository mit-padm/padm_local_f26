
from typing import (Callable, Iterable, List, Sequence, Tuple, Dict, Optional,
                    Any, Union, Set, ClassVar, Type, TypeVar)
import scipy
# Setup matplotlib animation
import matplotlib
from IPython.display import HTML
matplotlib.rc('animation', html='jshtml')
from abc import abstractmethod, ABC
import collections
import textwrap
import math
import functools
import itertools
import random
import dataclasses
import numpy as np
import heapq as hq

from functools import cached_property

import scipy.signal


def heatmap(data, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    import matplotlib.pylab as plt

    if ax is None:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if cbar_kw is not None:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=-30,
             ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for spine in ax.spines.values():
      spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im,
                     data=None,
                     valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None,
                     **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    import matplotlib.ticker

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


State = Any
Action = Any

class Problem(ABC):
    """The abstract base class for either a path cost problem or a reward problem."""

    @property
    @abstractmethod
    def initial(self) -> State:
        ...

    @abstractmethod
    def actions(self, state: State) -> Iterable[Action]:
        """Returns the allowed actions in a given state.

        The result would typically be a list. But if there are many
        actions, consider yielding them one at a time in an iterator,
        rather than building them all at once.
        """
        ...

    @abstractmethod
    def step(self, state: State, action: Action) -> State:
        """Returns the next state when executing a given action in a given
        state.

        The action must be one of self.actions(state).
        """
        ...


class PathCostProblem(Problem):
    """An abstract class for a path cost problem, based on AIMA.

    To formalize a path cost problem, you should subclass from this and
    implement the abstract methods. Then you will create instances of
    your subclass and solve them with the various search functions.
    """

    @abstractmethod
    def goal_test(self, state: State) -> bool:
        """Checks if the state is a goal."""
        ...

    @abstractmethod
    def step_cost(self, state1: State, action: Action, state2: State) -> float:
        """Returns the cost incurred at state2 from state1 via action."""
        ...

    def h(self, state: State) -> float:
        """Returns the heuristic value, a lower bound on the distance to goal."""
        return 0


class MDP(Problem):
    """A generative-model-based MDP."""

    @property
    def discount(self) -> float:
        """The discount factor."""
        return 1.

    @property
    def horizon(self) -> int:
        """The planning horizon."""
        return np.inf

    @abstractmethod
    def terminal(self, state: State) -> bool:
        """If this state is terminating (absorbing state)."""
        return False

    @abstractmethod
    def reward(self, state1: State, action: Action, state2: State) -> float:
        """Returns the reward given at state2 from state1 via action."""
        ...

    def get_next_state(self, state: State) -> State:
        """An MDP is fully observable."""
        return state


class SearchFailed(ValueError):
    """Raise this exception whenever a search must fail."""
    pass


# A useful data structure for best-first search
BFSNode = collections.namedtuple("BFSNode",
                                 ["state", "parent", "action", "cost", "g"])


def run_best_first_search(
    problem: PathCostProblem,
    get_priority: Callable[[BFSNode], float],
    step_budget: int = 10000
) -> Tuple[List[State], List[Action], List[float], int]:
    """A generic heuristic search implementation.

    Depending on `get_priority`, can implement A*, GBFS, or UCS.

    The `get_priority` function here should determine the order
    in which nodes are expanded. For example, if you want to
    use path cost as part of this determination, then the
    path cost (node.g) should appear inside of get_priority,
    rather than in this implementation of `run_best_first_search`.

    Important: for determinism (and to make sure our tests pass),
    please break ties using the state itself. For example,
    if you would've otherwise sorted by `get_priority(node)`, you
    should now sort by `(get_priority(node), node.state)`.

    Args:
      problem: a path cost problem.
      get_priority: a callable taking in a search Node and returns the priority
      step_budget: maximum number of `problem.step` before giving up.

    Returns:
      state_sequence: A list of states.
      action_sequence: A list of actions.
      cost_sequence: A list of costs.
      num_steps: number of taken `problem.step`s. Must be less than or equal to `step_budget`.

    Raises:
      error: SearchFailed, if no plan is found.
    """
    
    
    num_steps = 0
    frontier = []
    reached = {}

    root_node = BFSNode(state=problem.initial,
                        parent=None,
                        action=None,
                        cost=None,
                        g=0)
    hq.heappush(frontier, (get_priority(root_node), problem.initial, root_node))
    reached[problem.initial] = root_node
    num_expansions = 0

    while frontier:
        pri, s, node = hq.heappop(frontier)
        # If reached the goal, return
        if problem.goal_test(node.state):
            return (*finish_plan(node), num_steps)

        num_expansions += 1
        # Generate successors
        for action in problem.actions(node.state):
            if num_steps >= step_budget:
                raise SearchFailed(
                    f"Failed to find a plan in {step_budget} steps")
            child_state = problem.step(node.state, action)
            num_steps += 1
            cost = problem.step_cost(node.state, action, child_state)
            path_cost = node.g + cost
            # If the state is already in explored or reached, don't bother
            if not child_state in reached or path_cost < reached[child_state].g:
                # Add new node
                child_node = BFSNode(state=child_state,
                                     parent=node,
                                     action=action,
                                     cost=cost,
                                     g=path_cost)
                priority = get_priority(child_node)
                hq.heappush(frontier, (priority, child_state, child_node))
                reached[child_state] = child_node
    raise SearchFailed(f"Frontier exhausted after {num_steps} steps")


def finish_plan(node: BFSNode):
    """Helper for run_best_first_search."""
    state_sequence = []
    action_sequence = []
    cost_sequence = []

    while node.parent is not None:
        action_sequence.insert(0, node.action)
        state_sequence.insert(0, node.state)
        cost_sequence.insert(0, node.cost)
        node = node.parent
    state_sequence.insert(0, node.state)

    return state_sequence, action_sequence, cost_sequence


@dataclasses.dataclass(frozen=False, eq=False)
class MCTStateNode:
    """Node in the Monte Carlo search tree, keeps track of the children states."""
    state: State
    N: int
    horizon: int
    parent: Optional['MCTChanceNode']
    children: Dict['MCTChanceNode',
                   Action] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=False, eq=False)
class MCTChanceNode:
    U: float
    N: int
    parent: MCTStateNode
    children: Dict[State,
                   MCTStateNode] = dataclasses.field(default_factory=dict)


def ucb(n: MCTStateNode, C: float = 1.4) -> float:
    """UCB for a node, note the C argument"""
    return (np.inf if n.N == 0 else
            (n.U / n.N + C * np.sqrt(np.log(n.parent.N) / n.N)))


RolloutPolicy = Callable[[State], Action]


def random_rollout_policy(problem: MDP, state: State) -> Action:
    return random.choice(list(problem.actions(state)))


def run_mcts_search(problem: MDP,
                    state: Optional[State] = None,
                    state_sampler: Iterable[State] = None,
                    horizon: Optional[int] = None,
                    C: float = np.sqrt(2),
                    iteration_budget: int = 100,
                    n_simulations: int = 10,
                    max_backup: bool = True,
                    rollout_policy: RolloutPolicy = None,
                    verbose: bool = False) -> Action:
    """A generic MCTS search implementation for MDPs.

    This is a standard MCTS implementation based on description in AIMA
    (with some additional features).
    

    Args:
        problem: an MDP.
        state: the initial state. If None, then `state_sampler` must be provided.
        state_sampler: an iterable of states. If None, then `state` must be provided. This is used to sample the initial state for POMCP.
        horizon: the horizon of the search. If None, then the horizon is set to the problem's horizon.
        C: the C parameter for UCB.
        iteration_budget: the maximum number of iterations.
        n_simulations: the number of simulations to run at each MCTS iteration. In AIMA,
            this is set 1, but in general, we run multiple simulations to reduce variance.
        max_backup: whether to use max backup or sum backup.
        rollout_policy: the rollout policy. If None, then the random rollout policy is used.
        verbose: whether to print debug information.

    Returns:
        action: the best action to take at the initial state according to the search.
    """
    if state is None:
        state = problem.initial

    if state_sampler is None:
        state_sampler = itertools.repeat(state)

    if horizon is None:
        horizon = problem.horizon

    if rollout_policy is None:
        rollout_policy = functools.partial(random_rollout_policy, problem)

    ucb_fixed_C = functools.partial(ucb, C=C)

    rewards = []

    def select(n: MCTStateNode, state: State) -> Tuple[MCTStateNode, State]:
        """select a leaf node in the tree."""
        if n.children:
            # Select the best child, break ties randomly
            children = list(n.children.keys())
            random.shuffle(children)
            ucb_pick: MCTChanceNode = max(children, key=ucb_fixed_C)
            act = n.children[ucb_pick]
            next_state = problem.step(state, act)
            rewards.append(problem.reward(state, act, next_state))
            if next_state not in ucb_pick.children:
                new_leaf = MCTStateNode(next_state,
                                        horizon=n.horizon - 1,
                                        parent=ucb_pick,
                                        N=0)
                ucb_pick.children[next_state] = new_leaf
                return new_leaf, next_state
            return select(ucb_pick.children[next_state], next_state)
        return n, state

    def expand(n: MCTStateNode, state: State) -> Tuple[MCTStateNode, State]:
        """expand the leaf node by adding all its children actions."""
        assert not n.children
        if n.horizon == 0 or problem.terminal(state):
            return n, state
        for action in problem.actions(state):
            new_chance_node = MCTChanceNode(parent=n, U=0, N=0)
            n.children[new_chance_node] = action
        chance_node, action = random.choice(list(n.children.items()))
        next_state = problem.step(state, action)
        rewards.append(problem.reward(state, action, next_state))
        new_node = MCTStateNode(next_state,
                                N=0,
                                horizon=n.horizon - 1,
                                parent=chance_node)
        chance_node.children[next_state] = new_node
        return new_node, next_state

    def simulate(node: MCTStateNode, state: State) -> float:
        """simulate the utility of current state by taking a rollout policy."""
        total_reward = 0
        disc = 1
        h = node.horizon
        while h > 0 and not problem.terminal(state):
            action = rollout_policy(state)
            next_state = problem.step(state, action)
            reward = problem.reward(state, action, next_state)
            total_reward += disc * reward
            state = next_state
            disc = disc * problem.discount
            h -= 1
        return total_reward

    def backup(state_node: MCTStateNode, value: float) -> None:
        """passing the utility back to all parent nodes."""
        state_node.N += 1
        if state_node.parent:
            # Need to include the reward on the action *into* n
            parent_chance_node = state_node.parent
            parent_state_node = parent_chance_node.parent
            r = rewards.pop()
            future_val = r + problem.discount * value
            parent_chance_node.U += future_val
            parent_chance_node.N += 1
            if max_backup:
                bk_val = max(0 if n.N == 0 else n.U / n.N
                             for n in parent_state_node.children)
            else:
                bk_val = future_val
            backup(parent_state_node, bk_val)

    state = next(state_sampler)
    root = MCTStateNode(state=problem.get_next_state(state),
                        horizon=horizon,
                        parent=None,
                        N=0)

    i = 0
    while i < iteration_budget:
        state = next(state_sampler)
        assert len(rewards) == 0
        leaf, state = select(root, state)
        child, state = expand(leaf, state)
        value = np.mean([simulate(child, state) for _ in range(n_simulations)])
        backup(child, value)
        i += 1

    children = list(root.children.keys())
    random.shuffle(children)
    act = root.children[max(children, key=lambda p: p.U / p.N)]
    if verbose:
        print(
            {
                act: (c.U / c.N if c.N > 0 else 0, c.N)
                for c, act in root.children.items()
            }, act)
    return act





class Agent:
    """An agent that can act in an MDP.

    A derived agent must keep track of its own internal state.
    """

    def reset(self):
        """Reset the agent's internal state."""
        pass

    @abstractmethod
    def act(self, next_state: State) -> Action:
        """Return the agent's action given the state."""
        ...


@dataclasses.dataclass
class OpenLoopAgent(Agent):
    """Agent that just follows a fixed sequence of actions."""

    actions: Sequence[Action]

    t: int = dataclasses.field(default=0, init=False)

    def reset(self):
        self.t = 0

    def act(self, next_state) -> Action:
        del next_state  # state is not used
        assert self.t < len(self.actions)
        a = self.actions[self.t]
        self.t += 1
        return a


@dataclasses.dataclass
class RolloutLookaheadAgent(Agent):
    """MDP Agent that uses a rollout lookahead to decide what to do."""

    problem: MDP
    n_rollout_per_action: int = 10

    receding_horizon: int = None
    t: int = 0

    def reset(self):
        self.t = 0

    @property
    def planning_horizon(self):
        if self.receding_horizon is None:
            return self.problem.horizon - self.t
        return self.receding_horizon

    def act(self, state: State) -> Action:
        """Return the action that maximizes the expected reward."""
        self.t += 1
        actions = list(self.problem.actions(state))
        random.shuffle(actions)
        return max(actions, key=lambda a: self._rollout(state, a))

    def _rollout(self, state: State, action: Action) -> float:
        """Return the expected reward of taking action in state."""
        return sum(
            self._rollout_single(state, action) for _ in range(
                self.n_rollout_per_action)) / self.n_rollout_per_action

    def _rollout_single(self, state: State, action: Action) -> float:
        """simulate the utility of current state by taking a rollout policy."""
        total_reward = 0
        disc = 1
        t = 0
        planning_horizon = self.planning_horizon
        while t < planning_horizon and not self.problem.terminal(state):
            if t > 0:
                action = self.rollout_policy(state)
            next_state = self.problem.step(state, action)
            reward = self.problem.reward(state, action, next_state)
            total_reward += disc * reward
            state = next_state
            disc = disc * self.problem.discount
            t += 1
        return total_reward

    def rollout_policy(self, state: State) -> Action:
        """Return the action to take in state during rollout.

        Subclass may override to implement rollout policy with preferred actions."""

        return random.choice(list(self.problem.actions(state)))


def benchmark_agent(problem: MDP,
                    agent: Agent,
                    n_repeats: int = 100,
                    verbose: bool = False) -> List[float]:
    """Bencmark an agent on a problem by performing repeated experiments."""
    import tqdm
    total_rewards = []
    for _ in tqdm.tqdm(range(n_repeats)):
        *_, total_reward = run_agent_on_problem(problem, agent, verbose=verbose)
        total_rewards.append(total_reward)
    return total_rewards


def compare_agents(problem: MDP,
                   agents: Dict[str, Agent],
                   n_repeats: int = 30,
                   verbose: bool = False):
    """Compare the performance of multiple agents on a problem."""
    for agent in agents:
        if isinstance(agent, tuple):
            agent, name = agent
        else:
            name = agent.__class__.__name__
        print(f"Running {name}...")
        rewards = benchmark_agent(problem,
                                  agent,
                                  n_repeats=n_repeats,
                                  verbose=verbose)
        print(f"Mean reward: {np.mean(rewards):.2f} +- {np.std(rewards):.2f}")
        print(f"Median reward: {np.median(rewards):.2f}")
        print(f"Min reward: {np.min(rewards):.2f}")
        print(f"Max reward: {np.max(rewards):.2f}")


def run_agent_on_problem(problem: MDP,
                         agent: Agent,
                         verbose: bool = True
                        ) -> Tuple[Sequence[State], Sequence[Action], float]:
    """Runs the agent on the problem and returns the trajectory."""
    agent.reset()
    state = problem.initial
    state_sequence = [state]
    action_sequence = []
    total_reward = 0
    while not problem.terminal(state) and len(state_sequence) < problem.horizon:
        action = agent.act(state)
        next_state = problem.step(state, action)
        reward = problem.reward(state, action, next_state)
        total_reward += reward * problem.discount**len(state_sequence)
        if verbose:
            print(
                f"Action={action} reward={reward} total_reward={total_reward}")
        state = next_state
        action_sequence.append(action)
        state_sequence.append(state)
    return state_sequence, action_sequence, total_reward


def animate_trajectory(problem: MDP,
                       trajectory: Tuple[Sequence[State], Sequence[Action]]):
    """Visualizes a trajectory.

    Args:
        problem: The problem.
        trajectory: A tuple of state and action sequences.

    Returns:
        A matplotlib animation.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation

    state_sequence, action_sequence, *_ = trajectory

    fig, ax = plt.subplots()

    total_reward = 0.

    def animate(i):
        ax.clear()
        ax.set_aspect('equal')
        nonlocal total_reward
        if i == 0:
            total_reward = 0
            ax.set_title(f"Step {i}: begin, total_reward={total_reward:.2f}")
        elif i < len(state_sequence):
            action = action_sequence[i - 1]
            reward = problem.reward(state_sequence[i - 1], action,
                                    state_sequence[i])
            total_reward += reward * problem.discount**i
            ax.set_title(
                f"Step {i}: action={action}, "
                f"reward={reward:.2f}, total_reward={total_reward:.2f}")

        problem.render(state_sequence[i], ax=ax)

    anim = matplotlib.animation.FuncAnimation(fig,
                                              animate,
                                              frames=len(state_sequence),
                                              interval=500)
    return anim
