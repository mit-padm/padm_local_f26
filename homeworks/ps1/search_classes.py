

class SearchNode(object):
    def __init__(self, state, parent_node=None):
        self._parent = parent_node
        self._state = state

    def __repr__(self):
        return "<SearchNode: state: %s, parent: %s>" % (repr(self.state),repr(self.parent))

    def expand(self, graph):
        """Returns new search nodes pointing to each children state of the state represented by this node."""
        return [SearchNode(state, self) for state in graph.children_of(self.state)]

    @property
    def state(self):
        """Get the state represented by this SearchNode"""
        return self._state

    @property
    def parent(self):
        """Get the parent search node that we are coming from."""
        return self._parent
    
    def __eq__(self, other):
        return isinstance(other, SearchNode) and self._state == other._state

    def __hash__(self):
        return hash(self._state)
    
class Path(object):
    """This class computes the path from the starting state until the state specified by the search_node
    parameter by iterating backwards."""
    def __init__(self, search_node):
        self.path = []
        node = search_node
        while node is not None:
            self.path.append(node.state)
            node = node.parent
        self.path.reverse()
    def __repr__(self):
        return "Path of length %d: %s" % (len(self.path), self.path)

