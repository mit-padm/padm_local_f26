import numpy as np

class MDP(object):
    """Simple MDP class."""
    def __init__(self, S, A, T, R, gamma):
        """Define MDP"""
        self.S = S # Set of states
        self.A = A # Set of actions
        self.T = T # Transition probabilities: T[s][a][s']
        self.R = R # Rewards: R[s][a][s']
        self.gamma = gamma # Discount factor
    def print_mdp(self):
        # Code to print MDP
        print("MDP: \n  States (%d): %s\n  Actions (%d): %s" % (len(self.S),
                                                           ", ".join(["'%s'" % str(s) for s in self.S])
                                                                ,len(self.A)                                                                
                                                                , self.A))
        print("  Transitions:")
        for sij in sorted(self.S):
            print("   + State '%s'" % (str(sij)))
            for a in self.T[sij]:
                print("     -action '%s'" % str(a))
                for sdest,pdest in self.T[sij][a].items():
                    print("       to '%s' with p=%.2f" %(str(sdest), pdest))
        print("  Rewards:") 
        for sij in sorted(self.S):
            if sij in self.R:
                print("    + from state '%s'" % str(sij))
                for a in self.A:
                    if a in self.R[sij]:
                        for sdest,rdest in self.R[sij][a].items():
                            if not rdest == 0:
                                print("      - with '%s' to state '%s', r=%.2f" % (a, sdest, rdest))

def build_mdp(n, p, obstacles, goal, gamma, goal_reward=100, obstacle_reward=-1000):
    S = set()
    T = dict()
    R = dict()
    actions = ['up','down','right','left']
    vertical = [(0,1), (0,-1)]
    horizontal = [(1,0), (-1,0)]
    directions =  vertical + horizontal
    right_angles = [horizontal, horizontal, vertical, vertical]
    action_dest_dirs = [[directions[i]] + right_angles[i] for i in range(len(actions))]
    
    def xy_to_i(x,y):
        return n*x + y
    def apply_direction(x,y,direction):
        return (x+direction[0], y+direction[1])
    def valid_state(x,y):
        return 0<=x<n and 0<=y<n
    def neighbors(x,y):
        candidates = [(x+dx, y+dy) for dx,dy in directions]
        coords = (x,y)
        return list(filter(lambda coords: valid_state(coords[0],coords[1]), candidates))    
    def action_state(x, y, a):
        dx, dy = directions[actions.index(a)]
        return (x+dx, y+dy)
    def action_dest_states(x, y, a):
        dx, dy = directions[actions.index(a)]
        candidate_states = [apply_direction(x,y,d) for d in action_dest_dirs[actions.index(a)]]
        # If hitting obstacle, bounce back to same state
        return [cs if valid_state(*cs) else (x,y) for cs in candidate_states ]
        
    # Add states and transitions
    for i in range(n):
        for j in range(n):            
            sij = (i, j)
            S.add(sij)
            T[sij] = dict()
            ij_neighbors = neighbors(i,j)
            for ai, a in enumerate(actions):
                T[sij][a] = dict()
                dest_states = action_dest_states(i,j, a)
                T[sij][a][dest_states[0]] = p # main outcome of action
                remaining_p = (1-p)/2.0
                for other_s in dest_states[1:]:
                    T[sij][a][other_s] = T[sij][a].get(other_s, 0.0) + remaining_p
                assert np.allclose(sum([p_s_sdest for p_s_sdest in T[sij][a].values()]),1.0), "The sum of p for state %s should be 1.0 but it's %.2f" %(str(sij),
                                                                                         sum([p_s_sdest for p_s_sdest in T[sij][a].values()]))

            # Reward function
            # R(s, s')
            # Reward is 0 for all neighbor nodes by default
            R[sij] = dict()
            for a in actions:
                R[sij][a] = dict()
                for sdest in ij_neighbors:
                    R[sij][a][sdest] = 0.0
    

    # Add reward for goal    
    for nn in neighbors(*goal):
        for a in actions:
            if goal in T[nn][a]:
                R[nn][a][goal] = goal_reward
    
    # Negative rewards for obstacles
    for obs in obstacles:
        for a in actions:
            for nn in neighbors(*obs):
                if obs in T[nn][a]: 
                    R[nn][a][obs] = obstacle_reward
    
    # Make goal and obstacles sink states
    for sink_s in [goal]+obstacles:
        for a in actions:
            T[sink_s][a] = {sink_s: 1.0}
                    
                    
    mdp = MDP(S, actions, T, R, gamma)
    return mdp


def build_graph_mdp(G, goal, p, gamma, goal_reward=100):
    """Build an MDP on the nodes of a networkx graph G.

    Each edge of G is an action. Playing an edge that touches the current state
    moves the agent to the edge's other endpoint with probability p, and leaves
    it in place with probability 1-p. Playing any other edge leaves the agent in
    place. The agent receives goal_reward when it arrives at the goal, which is
    a sink state.
    """
    S = set(G.nodes)
    actions = list(G.edges)
    T = dict()
    R = dict()
    for s in S:
        T[s] = dict()
        R[s] = dict()
        for a in actions:
            u, v = a
            if s == goal or s not in (u, v):
                T[s][a] = {s: 1.0}
                R[s][a] = {s: 0.0}
                continue
            dest = v if s == u else u
            T[s][a] = {dest: p}
            R[s][a] = {dest: goal_reward if dest == goal else 0.0}
            if p < 1.0:
                T[s][a][s] = 1.0 - p
                R[s][a][s] = 0.0

    mdp = MDP(S, actions, T, R, gamma)
    return mdp





