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
    wire = 1
    rotor_top = 1 << 1
    rotor_bot = 1 << 2
    goal = 1 << 3

    def __init__(self):
        pass

    def perform_action(self, action_to_perform):
        if not isinstance(action_to_perform, action.Action):
            print("Error: Action is not in the allowed set of Actions")
            return

        if action_to_perform == action.Action.LEFT:
            self.__move_left()
        elif action_to_perform == action.Action.RIGHT:
            self.__move_right()
        elif action_to_perform == action.Action.UP:
            self.__move_up()
        elif action_to_perform == action.Action.DOWN:
            self.__move_down()
        elif action_to_perform == action.Action.ROTATE_CLOCKWISE:
            self.__turn_clockwise()
        elif action_to_perform == action.Action.ROTATE_COUNTER_CLOCKWISE:
            self.__turn_counter_clockwise()

    @abstractmethod
    def __move_left(self):
        pass

    @abstractmethod
    def __move_right(self):
        pass

    @abstractmethod
    def __move_up(self):
        pass

    @abstractmethod
    def __move_down(self):
        pass

    @abstractmethod
    def __turn_clockwise(self):
        pass

    @abstractmethod
    def __turn_counter_clockwise(self):
        pass

    @abstractmethod
    def get_state(self):
        pass
