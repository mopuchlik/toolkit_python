class Bird:
    def fly(self):
        print("Flying")


class Penguin(Bird):
    def fly(self):
        raise Exception("Penguins can't fly")


def make_it_fly(bird: Bird):
    bird.fly()


# b = Bird()
# make_it_fly(b)  # Works

p = Penguin()
make_it_fly(p)  # Raises exception → violates LSP
