from logger.log import log_debug


class Animal:
    static_value: str = 'codebusters'

    def __init__(self, name: str, age: int):
        self.name = name
        self.__age = age

    def show_name(self):
        log_debug(self.name)

    def __show_age(self):
        log_debug(self.__age)

    def set_name(self, value):
        self.name = value

    def get_age_multiply(self):
        return self.__age*2


class Cat(Animal):
    def __init__(self, name: str, age: int, color: str):
        super().__init__(name, age)
        self.color = color

    def set_color(self, value):
        self.color = value


def main():
    log_debug(Animal.static_value)
    animal = Animal(name='bug', age=1)
    animal.show_name()
    log_debug(animal.name)
    animal.set_name('dog')
    animal.show_name()
    log_debug(animal.get_age_multiply())

    cat = Cat(name='bug2', color='blue', age=2)
    log_debug(cat.color)
    cat.set_color('red')
    log_debug(cat.color)
    cat.show_name()
    log_debug(cat.static_value)


if __name__ == "__main__":
    main()
