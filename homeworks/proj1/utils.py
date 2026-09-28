import os
import copy
import uuid
import tempfile
import numpy as np
from pyperplan.pddl.parser import Parser
from pyperplan import grounding, planner
from IPython.display import display, HTML
from itertools import zip_longest

########################################
### Interactive Slider Visualization ###
########################################
# To make it easier to step through the execution of the plans you create, we’ve added an interactive HTML slider. You can
# drag the slider to move through the sequence of states without having lengthy print-outs.

# If the slider does not display properly in your environment, you can disable it by setting `USE_HTML = False` below and
# restarting your kernel before re-running the notebook. With `USE_HTML = False`, the code will fall back to printing each
# rendering one after another, so you can still follow the execution without the slider.

USE_HTML = True   # Change to False if the slider visualization doesn't work, make sure you RESTART your kernel


class State:
    """States have the following attributes:

    "robot": A (row, col) representing the robot's loc.
    "hospital": A (row, col) representing the hospital's loc.
    "carrying": The str name of a person being carried,
      or None, if no person is being carried.
    "people": A dict mapping str people names to (row, col)
      locs. If a person is being carried, they do not
      appear in this dict.
    "state_map": A numpy array of str 'C', 'F', 'S', and 'W',
      where 'C' represents free space, 'F' represents fire,
      'S' represents smoke, and 'W' represents an obstacle(wall).
      The robot may safely enter any cell that is clear (‘C’)
      or contains smoke (‘S’).
    """

    def __init__(self,
                 robot=None,
                 hospital=None,
                 carrying=None,
                 people=None,
                 state_map=None):
        default_state_map = np.array([['C', 'C', 'C', 'C', 'C', 'C', 'C'],
                                      ['C', 'W', 'W', 'C', 'C', 'W', 'W'],
                                      ['C', 'C', 'C', 'C', 'C', 'C', 'C'],
                                      ['C', 'C', 'W', 'C', 'C', 'C', 'C'],
                                      ['C', 'C', 'W', 'C', 'W', 'C', 'C'],
                                      ['C', 'C', 'C', 'C', 'C', 'W', 'C'],
                                      ['C', 'W', 'C', 'C', 'W', 'C', 'C']],
                                     dtype=str)
        default_robot = (0, 0)  # top left corner
        default_hospital = (6, 6)  # bottom right corner
        default_carrying = None
        default_people = {
            "p1": (4, 0),
            "p2": (6, 0),
            "p3": (0, 6),
            "p4": (3, 3)
        }
        self.state_map = state_map if state_map is not None else default_state_map
        self.robot = robot if robot is not None else default_robot
        self.hospital = hospital if hospital is not None else default_hospital
        self.carrying = carrying if carrying is not None else default_carrying
        self.people = people if people is not None else default_people

    def get_safe_grid(self):
        """
        "safe_grid": A grid map of boolean values where `True`
        indicate the locations where the robot are allowed to move into.

        Clear and Smoke grid cells are safe to enter
        """
        safe_grid = np.logical_or(self.state_map == "C", self.state_map == "S")
        return safe_grid

    def render(self, msg=None):
        height, width = self.state_map.shape
        state_arr = np.full((height, width), "  ", dtype=object)
        state_arr[self.state_map == 'W'] = "##"
        state_arr[self.state_map == 'F'] = "XX"
        state_arr[self.state_map == 'S'] = "||"
        state_arr[self.state_map == 'U'] = "??"
        state_arr[self.hospital] = "Ho"
        state_arr[self.robot] = "Ro"
        # Draw the people not at the hospital
        for person, loc in self.people.items():
            if loc == self.hospital:
                continue
            elif loc == self.robot:
                person = "R" + person[-1]
            state_arr[loc] = person
        # Add padding
        padded_state_arr = np.full((height + 2, width + 2), "##", dtype=object)
        padded_state_arr[1:-1, 1:-1] = state_arr
        state_arr = padded_state_arr
        carrying_str = f"Carrying: {self.carrying}"
        # Print
        full_str = ""
        if msg:
            full_str += msg + "\n"
        for row in state_arr:
            full_str += ''.join(row) + "\n"
        full_str += carrying_str + "\n"
        return full_str

    def copy(self):
        state_copy = copy.copy(self)
        state_copy.state_map = self.state_map.copy()  # copy the numpy array
        state_copy.people = self.people.copy()
        return state_copy


class SearchAndRescueProblem:
    """Defines a search and rescue (SAR) problem.

    In search and rescue, a robot must navigate to, pick up, and
    drop off people that are in need of help.

    Actions are strs. The following actions are defined:
      "up" / "down" / "left" / "right" : Moves the robot. The
        robot cannot move into obstacles or off the map.
      "pickup-{person}": If the robot is at the person, and if
        the robot is not already carrying someone, picks them up.
      "dropoff": If the robot is carrying a person, they are
        dropped off at the robot's current location.
      "look...": later we'll allow these actions, but they
        have no effect on the state.

    This structure serves as a container for a transition model
    "get_next_state(state, action)", an observaton model "get_observation(state)"
    and an action model "get_legal_actions(state)"

    Example usage:
      problem = SearchAndRescueProblem()
      state = State()
      state.render()
      action = "down"
      next_state = problem.get_next_state(state, action)
      next_state.render()
    """

    def __init__(self):
        self.action_deltas = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

    @staticmethod
    def is_valid_location(loc_r, loc_c, state, verbose=False):
        if not (0 <= loc_r < state.state_map.shape[0] and
                0 <= loc_c < state.state_map.shape[1]):
            if verbose:
                print(
                    "WARNING: attempted to move out of bounds, action has no effect."
                )
            return False
        if not state.get_safe_grid()[loc_r, loc_c]:
            if verbose:
                print(
                    "WARNING: attempted to move into an obstacle/unsafe region, action has no effect."
                )
            return False
        return True

    @staticmethod
    def get_legal_actions(state):
        legal_actions = ["up", "down", "left", "right", "dropoff"]
        for person in state.people:
            legal_actions.append(f"pickup-{person}")
        return legal_actions

    def get_next_state(self, state, action, verbose=False):
        legal_actions = self.get_legal_actions(state)
        if action not in legal_actions and not action.startswith('look'):
            raise ValueError(
                f"Unrecognized action {action}. Actions must be one of: {legal_actions}"
            )

        if action in ["up", "down", "left", "right"]:
            dr, dc = self.action_deltas[action]
            r, c = state.robot
            if not self.is_valid_location(
                    r + dr, c + dc, state, verbose=verbose):
                if verbose:
                    print(f"Action {action} is invalid in {state}.")
                return state, False
            new_state = state.copy()
            new_state.robot = (r + dr, c + dc)
            return new_state, True

        elif action.startswith("pickup"):
            person = action.split("-")[1]
            if state.carrying is not None:
                if verbose:
                    print(
                        "WARNING: attempted to pick up a person while already carrying someone, action has no effect."
                    )
                return state, False
            if person not in state.people or (state.people[person] !=
                                              state.robot):
                if verbose:
                    print(
                        "WARNING: attempted to pick up a person not at the robot location, action has no effect."
                    )
                return state, False
            new_state = state.copy()
            del new_state.people[person]
            new_state.carrying = person
            return new_state, True

        elif action == "dropoff":
            if state.carrying is None:
                if verbose:
                    print(
                        "WARNING: attempted to dropoff while not carrying anyone, action has no effect."
                    )
                return state, False
            person = state.carrying
            new_state = state.copy()
            new_state.carrying = None
            new_state.people[person] = state.robot
            return new_state, True

        elif action.startswith('look'):
            return state, True

        else:
            raise KeyError

    def get_observation(self, state):
        """Return the states of the adjacent (non-wall) grid squares."""
        height, width = state.state_map.shape
        deltas = self.action_deltas
        r, c = state.robot
        observation = {(r, c): state.state_map[r, c]}
        for direction, (dr, dc) in deltas.items():
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if state.state_map[nr, nc] == "W":
                continue
            observation[(nr, nc)] = state.state_map[nr, nc]
        return observation

def display_rendered_states(rendered_states_list):
    if USE_HTML:
        for rendered_states in rendered_states_list:
            slider_id = str(uuid.uuid4()).replace("-", "_")
            
            # def escape(text):
            #     return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            # states_js_array = [escape(s) for s in rendered_states]
            js_states = "[" + ",".join(f"`{s}`" for s in rendered_states) + "]"
            n_states = len(rendered_states)

            html = f"""
            <div style="margin-bottom:16px; border:1px solid #ddd; padding:8px;">
                <input type="range" id="{slider_id}_slider" min="0" max="{n_states-1}" value="0" style="width:100%">
                <div id="{slider_id}_label" style="font-family: sans-serif; margin-top:4px; font-size: 14px;">
                    Render 1 of {n_states}
                </div>
                <pre id="{slider_id}_container" style="font-family: monospace; white-space: pre; border:1px solid #ccc; padding:10px; margin-top:8px;"></pre>
            </div>
            <script>
                const states_{slider_id} = {js_states};
                const slider_{slider_id} = document.getElementById("{slider_id}_slider");
                const container_{slider_id} = document.getElementById("{slider_id}_container");
                const label_{slider_id} = document.getElementById("{slider_id}_label");

                function update_{slider_id}() {{
                    const i = parseInt(slider_{slider_id}.value, 10);
                    container_{slider_id}.textContent = states_{slider_id}[i];
                    label_{slider_id}.textContent = "Render " + (i+1) + " of {n_states}";
                }}

                slider_{slider_id}.addEventListener("input", update_{slider_id});
                update_{slider_id}();  // initialize
            </script>
            """
            display(HTML(html))
    else:
        for renderings in zip_longest(*rendered_states_list):
            for render in renderings:
                if render is not None:
                    print(render)

def execute_plan(problem, plan, state, visualize=True):
    rendered_states = []
    for action in plan:
        if visualize:
            rendered_states.append(state.render(msg=f'execute_plan: {action}'))
        # Resulting state
        state, valid = problem.get_next_state(state, action)
        assert valid, ('Attempted to execute invalid action '+ state + ' ' + action)
    rendered_states.append(state.render(msg=f'execute_plan: Final state'))

    if visualize:
        display_rendered_states([rendered_states])
    return state

def agent_loop(problem, initial_state, policy, initial_belief, max_steps=200):
    """See MP01 introduction."""
    rendered_states = []
    rendered_beliefs = []
    state = initial_state
    rendered_states.append(state.render(msg='Initial state'))
    belief = initial_belief
    rendered_beliefs.append(belief.render(msg='Initial belief'))

    # An initial observation
    observation = problem.get_observation(state)

    # Update the belief, first with transition, then with observation
    belief = belief.update(problem, observation)
    rendered_beliefs.append(belief.render(msg=f'Initial observation {observation} \nNew belief'))
    
    for step in range(max_steps):
        action = policy(belief)
        if action in ('*Success*', '*Failure*'):
            display_rendered_states([rendered_states, rendered_beliefs])
            print('Terminate with', action)
            return action, state, belief
        # Resulting state
        state, valid = problem.get_next_state(state, action)
        assert valid, f'Attempted to execute invalid action: {action}'
        # Get observation of grid squares around the robot
        observation = problem.get_observation(state)
        # Update the belief, first with transition, then with observation
        belief = belief.update(problem, observation, action)
        step_label = f'agent_loop: step {step}, action {action}\nObservation {observation}'
        rendered_states.append(state.render(msg=f'{step_label}\nNew state'))
        rendered_beliefs.append(belief.render(msg=f'{step_label}\nNew belief'))

    display_rendered_states([rendered_states, rendered_beliefs])
    return '*Failure*', state, belief


def get_num_delivered(state):
    """Returns the number of people located in the hospital."""
    num_delivered = 0
    for loc in state.people.values():
        if loc == state.hospital:
            num_delivered += 1
    return num_delivered


def execute_count_num_delivered(problem, state, plan, visualize=True):
    """Execute a plan for search and rescue and count the number of people
    delivered.

    Args:
      problem: A SearchAndRescueProblem
      plan: A list of action strs, see SearchAndRescueProblem.

    Returns:
      num_delivered: int
    """
    state = execute_plan(problem=problem, plan=plan, state=state, visualize=visualize)
    return get_num_delivered(state)


def run_planning(domain_pddl_str,
                 problem_pddl_str,
                 search_alg_name,
                 heuristic=None):
    """Plan a sequence of actions to solve the given PDDL problem.

    This function is a lightweight wrapper around pyperplan.

    Args:
      domain_pddl_str: A str, the contents of a domain.pddl file.
      problem_pddl_str: A str, the contents of a problem.pddl file.
      search_alg_name: A str, the name of a search algorithm in
        pyperplan. Options: astar, wastar, gbf, bfs, ehs, ids, sat.
      heuristic: A str or a pyperplan `Heuristic` class.
        A str, the name of a heuristic in pyperplan.
          Options: blind, hadd, hmax, hsa, hff, lmcut, landmark.
        A pyperplan `Heuristic` class.
          See: https://github.com/aibasel/pyperplan/blob/main/doc/documentation.md#implementing-new-heuristics

    Returns:
      plan: A list of actions; each action is a pyperplan Operator.
    """
    # Parsing the PDDL
    domain_file = tempfile.NamedTemporaryFile(delete=False, dir='.')
    problem_file = tempfile.NamedTemporaryFile(delete=False, dir='.')
    with open(domain_file.name, 'w') as f:
        f.write(domain_pddl_str)
    with open(problem_file.name, 'w') as f:
        f.write(problem_pddl_str)
    parser = Parser(domain_file.name, problem_file.name)
    domain = parser.parse_domain()
    problem = parser.parse_problem(domain)
    os.remove(domain_file.name)
    os.remove(problem_file.name)

    # Ground the PDDL
    task = grounding.ground(problem)

    # Get the search alg
    search_alg = planner.SEARCHES[search_alg_name]

    if heuristic is None:
        return search_alg(task)

    if isinstance(heuristic, str):
        # Get the heuristic from pyperplan
        heuristic_initialized = planner.HEURISTICS[heuristic](task)
    else:
        # Use customized heuristic
        heuristic_initialized = heuristic(task)

    # Run planning
    return search_alg(task, heuristic_initialized)


# Test Cases

# First problem
P1_B0 = np.array([["U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U"],
                  ["U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U"],
                  ["U", "U", "U", "U", "U"], ["U", "U", "U", "U", "U"]])

P1_B1 = np.array([["C", "S", "C", "C", "C"], ["S", "U", "U", "U", "U"],
                  ["S", "U", "U", "U", "U"], ["S", "U", "U", "U", "U"],
                  ["C", "U", "U", "U", "U"], ["C", "C", "C", "C", "C"]])

P1_G0 = np.array([["C", "S", "C", "C", "C"], ["S", "F", "S", "C", "C"],
                  ["S", "F", "S", "S", "S"], ["S", "F", "F", "F", "F"],
                  ["C", "S", "S", "S", "S"], ["C", "C", "C", "C", "C"]])

# Second problem
P2_B1 = np.array([["C", "S", "C", "C", "C"], ["S", "U", "U", "C", "U"],
                  ["S", "U", "U", "C", "U"], ["S", "U", "U", "U", "U"],
                  ["C", "U", "U", "C", "U"], ["C", "C", "C", "C", "C"]])

P2_G0 = np.array([["C", "S", "C", "C", "C"], ["S", "F", "S", "C", "C"],
                  ["S", "F", "S", "C", "S"], ["S", "F", "F", "S", "F"],
                  ["C", "S", "S", "C", "S"], ["C", "C", "C", "C", "C"]])