from abc import ABC, abstractmethod

import Decider.action as action


#
# Abstract World Class modeling a hot wire game capable of performing actions and switching states
#
class World(ABC):

    #
    # World encoding:
    # Unix like world encoding to enable fast bitwise operation on the world states
    #
    backdrop = 0
    wire = 1
    rotor_top = 1 << 1
    rotor_bot = 1 << 2
    goal = 1 << 3
    out_of_world = 1 << 4

    def __init__(self):
        pass

    def perform_action(self, action_to_perform):
        if not isinstance(action_to_perform, action.Action):
            print("Error: Action is not in the allowed set of Actions")
            return

        if action_to_perform == action.Action.LEFT:
            self.move_left()
        elif action_to_perform == action.Action.RIGHT:
            self.move_right()
        elif action_to_perform == action.Action.UP:
            self.move_up()
        elif action_to_perform == action.Action.DOWN:
            self.move_down()
        elif action_to_perform == action.Action.ROTATE_CLOCKWISE:
            self.turn_clockwise()
        elif action_to_perform == action.Action.ROTATE_COUNTER_CLOCKWISE:
            self.turn_counter_clockwise()

    @abstractmethod
    def move_left(self):
        pass

    @abstractmethod
    def move_right(self):
        pass

    @abstractmethod
    def move_up(self):
        pass

    @abstractmethod
    def move_down(self):
        pass

    @abstractmethod
    def turn_clockwise(self):
        pass

    @abstractmethod
    def turn_counter_clockwise(self):
        pass

    @abstractmethod
    def get_state(self, coordinates):
        pass
