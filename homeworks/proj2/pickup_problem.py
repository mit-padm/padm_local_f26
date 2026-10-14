from typing import Tuple, List, ClassVar, Set, Iterable
from utils import PathCostProblem, State, Action, heatmap
import dataclasses
import collections
import numpy as np 
from functools import cached_property

@dataclasses.dataclass(frozen=True, eq=True, order=True)
class PickupProblemState:
    robot_loc: Tuple[int, int]
    carried_patient: bool


OneWayBlock = collections.namedtuple('OneWayBlock', ['loc', 'dest'])

@dataclasses.dataclass(frozen=True)
class PickupProblem(PathCostProblem):
    """A simple deterministic pickup problem in a grid.

    The robot starts at some location and can move up, down, left, or right.
    There is a patient at some location.
    The goal is to pick up the patient and drop them off at the hospital.
    The robot can only move to a location if there is a path from the robot's
    current location to the destination that does not pass through any roadblocks.
    """
    grid_shape: Tuple[int, int] = (8, 4) 

    initial_robot_loc: Tuple[int, int] = (0, 1)

    patient_loc: Tuple[int, int] = (7, 0)  # initial location of the patient
    hospital_loc: Tuple[int, int] = (0, 3)

    one_ways = []
    for i in range(7):
        one_ways.append(OneWayBlock((i+1, 1), (i,1)))
        one_ways.append(OneWayBlock((i, 3), (i+1,3)))

    for i in range(1,6):
        one_ways.append(OneWayBlock((i, 1), (i, 0)))
        one_ways.append(OneWayBlock((i, 1), (i, 2)))
    one_ways.append(OneWayBlock((0, 0), (1, 0)))
    one_ways.append(OneWayBlock((7, 0), (6, 0)))
    one_ways.append(OneWayBlock((0, 2), (1, 2)))
    one_ways.append(OneWayBlock((7, 2), (6, 2)))
    one_ways.append(OneWayBlock((1, 2), (0, 2)))
    one_ways.append(OneWayBlock((6, 2), (7, 2)))

    grid_shape: Tuple[int, int]

    initial_robot_loc: Tuple[int, int] = (0, 0)

    patient_loc: Tuple[int, int] = (4, 0)  # initial location of the patient
    hospital_loc: Tuple[int, int] = (0, 3)

    one_ways: List[OneWayBlock] = dataclasses.field(default_factory=list)
    
    grid_act_to_delta: ClassVar = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }
    all_grid_actions: ClassVar = tuple(grid_act_to_delta.keys())

    @property
    def initial(self) -> State:
        return PickupProblemState(self.initial_robot_loc,
                                  self.initial_robot_loc == self.patient_loc)

    @cached_property
    def _one_way_set(self) -> Set[OneWayBlock]:
        """Set of one-way blocks. Helps with fast lookup."""
        return set(self.one_ways)

    def actions(self, state: PickupProblemState) -> Iterable[Action]:
        """Actions from the current state: move up, down, left, or right unless blocked."""
        (r, c) = state.robot_loc
        actions = []
        for act in self.all_grid_actions:
            dr, dc = self.grid_act_to_delta[act]
            new_r, new_c = r + dr, c + dc
            if (new_r in range(self.grid_shape[0]) and
                    new_c in range(self.grid_shape[1]) and
                    OneWayBlock(state.robot_loc,
                                (new_r, new_c)) not in self._one_way_set):
                actions.append(act)
        return actions

    def step(self, state: PickupProblemState,
             action: Action) -> PickupProblemState:
        """We automatically pick up patient if we're on that square."""
        (r, c) = state.robot_loc
        dr, dc = self.grid_act_to_delta[action]
        return PickupProblemState(
            (r + dr, c + dc),
            state.carried_patient or self.patient_loc == (r + dr, c + dc),
        )

    def step_cost(self, state1, action, state2) -> float:
        """Cost of taking an action in a state.

        """
        return 1.

    def goal_test(self, state: PickupProblemState) -> bool:
        """True if at hospital and holding patient."""
        return state.robot_loc == self.hospital_loc and state.carried_patient

    def render(self, state: PickupProblemState, ax=None):
        """Render the state as a grid."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()

        heatmap(np.zeros(self.grid_shape),
                ax=ax,
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
                origin="upper")

        # Render the robot
        robot = plt.Circle(state.robot_loc[::-1], 0.5, color='blue')
        ax.add_patch(robot)

        # Render the patient
        patient_loc = (state.robot_loc
                       if state.carried_patient else self.patient_loc)
        patient = plt.Circle(patient_loc[::-1], 0.3, color='orange')
        ax.add_patch(patient)

        # Render the hospital
        plt.plot([self.hospital_loc[1]], [self.hospital_loc[0]],
                 marker='P',
                 color='r',
                 markersize=15)

        # Render the walls and one-way doors
        one_ways = set(self._one_way_set)
        while one_ways:
            one_way = one_ways.pop()
            rev = OneWayBlock(one_way.dest, one_way.loc)
            if rev in one_ways:
                one_ways.remove(rev)
                src, dst = one_way.loc, one_way.dest
                if src < dst:
                    dst, src = src, dst
                src, dst = np.array(src), np.array(dst)
                mid_pt = (src + dst) / 2
                delta = dst - src
                wall = plt.Rectangle(mid_pt[::-1] - delta / 2 +
                                     delta[::-1] * 0.05,
                                     delta[0] if delta[0] != 0 else 0.1,
                                     delta[1] if delta[1] != 0 else 0.1,
                                     color='black')
                ax.add_patch(wall)
            else:
                src, dst = np.array(one_way.loc), np.array(one_way.dest)
                mid_pt = (src + dst) / 2
                delta = dst - src
                base = mid_pt + delta * 0.07
                length = -delta * 0.2
                arrow = plt.arrow(base[1],
                                  base[0],
                                  length[1],
                                  length[0],
                                  width=0.2,
                                  length_includes_head=True,
                                  head_length=0.2,
                                  color='black')
                ax.add_patch(arrow)