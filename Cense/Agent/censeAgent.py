import World.world

world = None

if __name__ == '__main__':
    print("test")


if __name__ == '_init':
    print("init")


def init(image_path):
    global world
    world = World(image_path)
