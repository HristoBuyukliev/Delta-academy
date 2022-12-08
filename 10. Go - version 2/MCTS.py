def UCT_search(game_state, num_reads):
    root = UCTNode(game_state)
    for _ in range(num_reads):
        leaf = root.select_leaf()
        child_priors, value_estimate = NeuralNet.evaluate(leaf.game_state)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
    return max(root.children.items(),
               key=lambda item: item[1].number_visits)


class UCTNode():
    def __init__(self, game_state,
                 move, parent=None):
        self.game_state = game_state
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = {}  # Dict[move, UCTNode]
        self.child_priors = np.zeros(
            [81], dtype=np.float32)
        self.child_total_value = np.zeros(
            [81], dtype=np.float32)
        self.child_number_visits = np.zeros(
            [81], dtype=np.float32)
        
    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]
    
    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits = value
        
    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value = value
        
    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visit)
    
    def child_U(self):
        return math.sqrt(self.number_visits) * (
            self.child_priors / (1 + self.child_number_visits))
    
    def best_child(self):
        return np.argmax(self.child_U() + self.child_Q())