prompts = {
    "env_description": """\
You are responsible for critiquing a decision policy to solve the following task: 
The agent has to pick up a box which is placed in another room, behind a locked door. This environment can be solved without relying on language.

The observation space is defined formally as: 
You can see a (7, 7) square of tiles in the direction you are facing and your inventory of item. Formally - observation['agent']['direction'] with 0: right, 1: down, 2: left, 3: up
- observation['agent']['image'] array with shape (7, 7, 3) with each tile in the (7, 7) grid encoded as the triple (object: int, color: int, state: int) where
    - object with 0: unseen, 1: empty, 2: wall, 3: floor, 4: door, 5: key, 6: ball, 7: box, 8: goal, 9: lava
    - color with 0: red, 1: green, 2: blue, 3: purple, 4: yellow, 5: grey
    - state with 0: door open, 1: door closed, 2: door locked
- observation['inv']: list[int] contains any object being held and is empty otherwise 
Note, the agent is always located at observation['image'][3][6] with observation['image'][2] to the left and observation['image'][4] to the right and observation['image'][3][5] forward.


The action space is defined formally as:
action: int such that
- 0: turn left
- 1: turn right
- 2: move forward, Precondition: Forward tile must be empty
- 3: pickup item, Precondition: must be standing in tile adjacent to object and facing it (item must be on observation['image'][3][5]). Cannot be holding another object
- 4: drop item, Precondition: Must be holding item. Tile you are facing must be empty
- 5: toggle key to open door, Precondition: Must have key and be facing the door (door is on observation['image'][3][5])


The rewards are defined formally as:
A reward of ‘1 - 0.9 * (step_count / 300)’ is given for success  for picking up the box
+0.1 for picking up the key for the first time. 
+0.2 for opening the door, and +0.1 for going through the door, 
+0.1 for putting down the key after opening the door.
""",

    "example_policy": """\
import numpy as np

class Policy:
    def __init__(self):
        self.memory = {
            'key_picked_up': False,
            'door_opened': False,
            'door_opening': False,
            'door_unlocked': False,
            'box_picked_up': False
        }

        self.steps = 0
        self.actions = [0, 1, 2, 3, 4, 5]
        self.explore_count = 0
        self.explored_tiles = set()

    def act(self, observation):
        direction = observation['agent']['direction']
        image = observation['agent']['image']
        inv = observation['inv']

        current_tile = image[3][6]
        forward_tile = image[3][5]

        if forward_tile[0] == 5 and 3 not in inv:
            return 3  # Pickup key

        if forward_tile[0] == 4 and forward_tile[2] == 2 and 5 in inv and not self.memory['door_opening']:
            return 5  # Toggle key to open door

        if forward_tile[0] == 7 and 0 in inv:
            return 3  # Pickup box if available

        if forward_tile[0] not in [2, 9] and tuple(forward_tile) not in self.explored_tiles:
            self.explored_tiles.add(tuple(forward_tile))
            return 2  # Move forward if facing an unexplored tile

        # If holding key, door is open, and facing the box, drop the key before picking up the box
        if 5 in inv and self.memory['door_opened'] and forward_tile[0] == 7:
            return 4  # Drop the key

        # Exploration strategy
        self.explore_count += 1

        if self.explore_count % 3 == 0:
            return np.random.choice([0, 1])  # Randomly turn left or right every 3rd step

        if self.explore_count % 2 == 0:
            return 2  # Move forward every 2nd step

        return 2  # Move forward by default

    def update(self, observation, action, reward, next_observation):
        key_picked_up = self.memory['key_picked_up']
        door_opened = self.memory['door_opened']
        door_unlocked = self.memory['door_unlocked']
        door_opening = self.memory['door_opening']
        box_picked_up = self.memory['box_picked_up']

        self.steps += 1

        if not key_picked_up and reward == 0.1:
            self.memory['key_picked_up'] = True

        if not door_unlocked and reward == 0.3:
            self.memory['door_unlocked'] = True

        if not door_opening and reward == 0.2:
            self.memory['door_opening'] = True

        if not door_opened and reward == 0.1:
            self.memory['door_opened'] = True

        if not box_picked_up and reward == 0.9:
            self.memory['box_picked_up'] = True
""",

    "report_explanation": """\
The above policy has been rolled out in the environment and a dataset of trajectories has been collected.
You must now analyze this dataset to better understand the policy's behavior and environment dynamics.
This is done by using reporting functions which capture parts of the policy's behavior and the environmental dynamics.
Note: len(trajectory["observations"]) == len(trajectory["actions"]) + 1 because the observation after 
the last action is also saved.
""",

}