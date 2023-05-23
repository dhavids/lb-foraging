#tester with terminal
class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None
        self.id= ""

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller
    
    def set_id(self, id):
        self.id = id

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return self.id

pp= [Player() for i in range(5)]
for idx, player in enumerate(pp):
    player.set_id("player_"+str(idx))
for p in pp:
    print(p.name)