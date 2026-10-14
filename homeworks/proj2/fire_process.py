import numpy as np
import dataclasses
import functools
import scipy.signal
from functools import cached_property
from typing import Type, Optional, Any, Dict, Iterable, TypeVar
import textwrap
from utils import State, MDP, Action, heatmap
from pickup_problem import PickupProblem, OneWayBlock, PickupProblemState
FireGridT = np.ndarray  # boolean array

conv2D = functools.partial(scipy.signal.convolve2d,
                           mode='same',
                           boundary='fill',
                           fillvalue=0)


@dataclasses.dataclass(frozen=True)
class FireProcess:
    '''A probabilistic model for the evolution of fire in a grid.

      At time step $t$, the probability of a cell being on fire is weighted probability 
      of the neighboring cell being on fire at time step $t-1$.
      '''

    initial_fire_grid: np.ndarray

    fire_weights: np.ndarray = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])
    attenuation: float = 1.0

    rng: np.random.Generator = dataclasses.field(
        default_factory=np.random.default_rng)

    @cached_property
    def normalized_fire_weights(self) -> np.ndarray:
        return self.attenuation * self.fire_weights / np.sum(self.fire_weights)

    def dist(self, fire_grid: FireGridT) -> np.ndarray:
        '''Given the fire grid at time t, return a new grid with marginal distributions of fire for t + 1.'''
        next_fire_dist = conv2D(fire_grid, self.normalized_fire_weights)
        return np.clip(next_fire_dist, 0, 1)  # clip for numerical stability

    def sample(self, fire_grid: FireGridT) -> FireGridT:
        '''Given the fire grid at time t, return a new grid that's a sample from the distribution.'''
        return self.rng.binomial(1, self.dist(fire_grid)).astype(bool)

    def render(self, fire_grid: FireGridT, ax=None):
        heatmap(fire_grid, ax=ax, cmap="YlOrRd",
                vmin=0, vmax=1, origin="upper")


@dataclasses.dataclass(frozen=True, eq=True)
class FireMDPState(PickupProblemState):
    '''A state in the fire MDP extends the pickup problem state with a fire grid.'''

    fire_grid: np.ndarray

    # Below we implement __eq__ and __hash__ to make FireMDPState hashable.
    def __eq__(self, other):
        if not isinstance(other, FireMDPState):
            return False
        return (self.robot_loc == other.robot_loc and
                self.carried_patient == other.carried_patient and
                np.all(self.fire_grid == other.fire_grid))

    def __hash__(self):
        return hash((super().__hash__(), self.fire_grid.tobytes()))


T = TypeVar('T', bound='FireProblemCommon')


@dataclasses.dataclass(frozen=True)
class FireProblemCommon:
    '''Common code for the fire MDP and POMDP problems.'''

    pickup_problem: PickupProblem
    fire_process: FireProcess

    _horizon: int = np.inf
    _discount: float = 0.999

    step_reward: float = 0.

    burn_reward: float = -1
    goal_reward: float = 1.

    @property
    def horizon(self):
        return self._horizon

    @property
    def discount(self):
        return self._discount

    @property
    def initial_robot_loc(self):
        return self.pickup_problem.initial_robot_loc

    def robot_burned(self, state) -> bool:
        return bool(state.fire_grid[state.robot_loc])

    def succeeded(self, state) -> bool:
        return self.pickup_problem.goal_test(state)

    def reward(self, state1, action, state2) -> float:
        if self.robot_burned(state2):
            return self.burn_reward
        if self.succeeded(state2):
            return self.goal_reward
        return self.step_reward

    def terminal(self, state) -> bool:
        return (self.robot_burned(state) or self.succeeded(state))

    @property
    def grid_shape(self):
        return self.pickup_problem.grid_shape

    @classmethod
    def from_str(cls: Type[T],
                 env_s: str,
                 fire_process_kargs: Optional[Dict[str, Any]] = None,
                 **kwargs) -> T:
        '''Create a problem from a grid string.

            Legend:
                . = empty
                F = fire
                < = one way block (can go left)
                > = one way block (can go right)
                ^ = one way block (can go up)
                v = one way block (can go down)
                X = wall (two way block)
                R = robot
                P = patient
                H = hospital
            Each line must be the same length, and starts and ends with a `|`.
            Each character is separated by a space.

            Warning:
                The user needs to make sure that no cell location is a dead end (due to roadblocks)
                since our `terminal` condition does not check for empty available actions.
            '''
        if fire_process_kargs is None:
            fire_process_kargs = {}

        lines = env_s.splitlines()
        assert all(len(line) == len(lines[0]) for line in lines)
        assert all(line[0] == '|' and line[-1] == '|' for line in lines)
        # remove the first and last character of each line
        lines = [line[1:-1] for line in lines]
        # split each line into a list of characters
        lines = [line[::2] for line in lines]
        w = len(lines[0]) // 2 + 1
        h = len(lines) // 2 + 1
        robot_loc = None
        patient_loc = None
        hospital_loc = None
        fire_grid = np.zeros((h, w), dtype=bool)
        one_ways = []
        for i, line in enumerate(lines):
            i2 = i // 2
            if i % 2 == 0:
                for j, c in enumerate(line):
                    j2 = j // 2
                    if j % 2 == 0:
                        if c == 'F':
                            fire_grid[i2, j2] = True
                        elif c == 'R':
                            robot_loc = (i2, j2)
                        elif c == 'P':
                            patient_loc = (i2, j2)
                        elif c == 'H':
                            hospital_loc = (i2, j2)
                        else:
                            assert c == '.'
                    else:
                        if c in '<X':
                            one_ways.append(OneWayBlock(
                                (i2, j2), (i2, j2 + 1)))
                        if c in '>X':
                            one_ways.append(OneWayBlock(
                                (i2, j2 + 1), (i2, j2)))
            if i % 2 == 1:
                for j, c in enumerate(line[::2]):
                    if c in 'vX':
                        one_ways.append(OneWayBlock(
                            loc=(i2 + 1, j), dest=(i2, j)))
                    if c in '^X':
                        one_ways.append(OneWayBlock(
                            loc=(i2, j), dest=(i2 + 1, j)))

        if robot_loc is None:
            raise ValueError("No robot location specified")
        if patient_loc is None:
            raise ValueError("No patient location specified")
        if hospital_loc is None:
            raise ValueError("No hospital location specified")

        pickup_problem = PickupProblem(fire_grid.shape, robot_loc, patient_loc,
                                       hospital_loc, one_ways)
        fire_process = FireProcess(fire_grid, **fire_process_kargs)
        return cls(pickup_problem, fire_process, **kwargs)


@dataclasses.dataclass(frozen=True)
class FireMDP(FireProblemCommon, MDP):
    '''The completely observable fire problem.'''

    @property
    def initial(self) -> FireMDPState:
        return FireMDPState(*dataclasses.astuple(self.pickup_problem.initial),
                            self.fire_process.initial_fire_grid)

    def actions(self, state: State) -> Iterable[Action]:
        return self.pickup_problem.actions(state)

    def step(self, state: FireMDPState, action: Action) -> FireMDPState:
        return FireMDPState(
            *dataclasses.astuple(self.pickup_problem.step(state, action)),
            self.fire_process.sample(state.fire_grid))

    def render(self, state: FireMDPState, ax=None):
        self.pickup_problem.render(state, ax=ax)
        self.fire_process.render(state.fire_grid, ax=ax)


def get_problem(name: str) -> MDP:
    '''Return a problem instance by name.'''

    params = {
        "maze":
            dict(env_s='''\
                |R < . X . > H|
                |            X|
                |. X .   .   .|
                |        X   X|
                |. X .   . X .|
                |    X   ^    |
                |. X .   . X .|
                |    X   ^   ^|
                |P   . > . > .|
                ''',
                 fire_process_kargs=dict(fire_weights=np.array([
                     [0, 1, 0],
                     [1, 10, 1],
                     [0, 1, 0],
                 ])),
                 _horizon=20),
        "just_wait":
            dict(env_s='''\
                |.   R   .   H|
                |X   v   X   ^|
                |. X . X .   .|
                |    v       ^|
                |. X . X .   .|
                |    v       ^|
                |. X . X .   .|
                |    v       ^|
                |F X . X F   .|
                |    v       ^|
                |. X . X .   F|
                |    v       ^|
                |. X . X .   .|
                |X   v   X   ^|
                |P   .   .   .|
                ''',
                 fire_process_kargs=dict(fire_weights=np.array([
                     [0, 1, 0],
                     [1, 20, 1],
                     [0, 1, 0],
                 ]))),
        "the_circle":
            dict(env_s='''\
                |R   .   .   H|
                |    X   X   v|
                |. X .   . X .|
                |            v|
                |. X F   . X .|
                |            v|
                |. X F   . X .|
                |    X   X   v|
                |P   . < . < .|
                ''',
                 fire_process_kargs=dict(fire_weights=np.array([
                     [0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0],
                 ]))),
        "the_choice":
            dict(
                env_s='''\
                |.   .   F   F   F   F|
                |.   X   X   X   X   X|
                |. X .   .   .   .   .|
                |X                    |
                |R > .   .   F   .   .|
                |v                    |
                |. X .   .   .   .   .|
                |v   X   X   X   X   v|
                |. X .   F   F   . X .|
                |v                   v|
                |. X F   .   .   . X .|
                |v                   v|
                |. X F   F   .   . X .|
                |v                   v|
                |. X .   F   .   . X .|
                |v   X   X   X   X   v|
                |. > . > H   P < . < .|
                ''',
                fire_process_kargs=dict(fire_weights=np.array([
                    [0, 1, 0],
                    [1, 4, 1],
                    [0, 1, 0],
                ]),),
            )
    }

    if name not in params:
        raise ValueError(f"Unknown problem name: {name}")

    params[name]["env_s"] = textwrap.dedent(params[name]["env_s"])
    return FireMDP.from_str(**params[name])
