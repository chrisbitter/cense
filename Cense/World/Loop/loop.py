from pyfirmata import Arduino, util
import time
import threading

class Loop:
    def __init__(self, timeout=.1):

        self.touched_wire = False

        thread = threading.Thread(target=self.check_connection)
        thread.daemon = True

        #give board 5 seconds to establish a connection
        time.sleep(5)

        thread.start()

    def check_connection(self):
        board = Arduino('com3')
        it = util.Iterator(board)
        it.setDaemon(True)
        it.start()
        board.analog[0].enable_reporting()
        board.analog[1].enable_reporting()
        
        
        analog_0 = board.get_pin('a:0:i')
        analog_1 = board.get_pin('a:1:i')

        while True:
            value_a0 = analog_0.read()
            value_a1 = analog_1.read()

            if value_a0 is not None and value_a1 is not None:
                if value_a0 > value_a1:
                    self.touched_wire = True

            time.sleep(.1)

    def has_touched_wire(self):
        time.sleep(.2)
        response = self.touched_wire
        self.touched_wire = False
        return response

    def is_touching_wire(self):
        self.touched_wire = False

        now = time.time()
        while time.time() - now < .5:
            if self.touched_wire:
                return True

        return self.touched_wire


if __name__ == "__main__":
    loop = Loop()

    while True:
        print(loop.has_touched_wire())
        input()
