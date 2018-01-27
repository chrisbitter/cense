from pyfirmata import Arduino, util
import time
import threading


class Loop:
    def __init__(self):

        print("Setup Loop")

        self.timestamp_touched = 0
        self.timestamp_not_touched = 0

        thread = threading.Thread(target=self.check_connection)
        thread.daemon = True

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

        tic = time.time()

        while True:
            value_a0 = analog_0.read()
            value_a1 = analog_1.read()

            if value_a0 is not None and value_a1 is not None:
                if value_a0 > value_a1:
                    self.timestamp_touched = time.time()
                else:
                    self.timestamp_not_touched = time.time()
                tic = time.time()

            if time.time() - tic > 1:
                print("Loop Thread slow")

            time.sleep(.01)

    def has_touched_wire(self, timestamp=0):
        while True:
            if self.timestamp_touched > timestamp:
                return True
            # second condition in case timestep_touched was updated
            elif self.timestamp_not_touched > timestamp and self.timestamp_not_touched > self.timestamp_touched:
                return False

    def is_touching_wire(self):

        return self.has_touched_wire(time.time())

if __name__ == "__main__":
    loop = Loop()

    while True:
        print(loop.has_touched_wire(time.time()))
        input()
