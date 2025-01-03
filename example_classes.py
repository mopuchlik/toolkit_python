# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:00:57 2023

example of a class with some tests

@author: User
"""


class Pet:

    # class variable
    global_var = 1

    def __init__(self, name, age):
        self.name = name
        self.age = age

    #  standard method working with an instance
    def get_name(self):
        return self.name

    @classmethod
    def add_1(cls):
        cls.global_var += 1

    @staticmethod
    def x_double(x):
        return 2 * x


class Dog(Pet):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color


a = Pet("reksio", 8)
b = Pet("lulek", 8)
print(a.name)
print(a.global_var)

# dodaje dla obu
a.add_1()
print(a.global_var)
print(b.global_var)

c = Dog("reksio", 8, "czarny")
print(c.color)

# statyczna funkcja dziala jak biblioteka
a.x_double(2)

# podklasa ma dostep do funkcji nadklasy
c.x_double(2)


# %% z kursu, zlicza loginy i logouty
# i listuje kto jest zalogowany na danej maszynie


def get_event_date(event):
    return event.date


def current_users(events):
    events.sort(key=get_event_date)
    machines = {}
    for event in events:
        if event.machine not in machines:
            machines[event.machine] = set()
        if event.type == "login":
            machines[event.machine].add(event.user)
        elif event.type == "logout":
            machines[event.machine].remove(event.user)
    return machines


def generate_report(machines):
    for machine, users in machines.items():
        if len(users) > 0:
            user_list = ", ".join(users)
            print("{}: {}".format(machine, user_list))


class Event:
    def __init__(self, event_date, event_type, machine_name, user):
        self.date = event_date
        self.type = event_type
        self.machine = machine_name
        self.user = user


# %%

events = [
    Event("2020-01-21 12:45:46", "login", "myworkstation.local", "jordan"),
    Event("2020-01-22 15:53:42", "logout", "webserver.local", "jordan"),
    Event("2020-01-21 18:53:21", "login", "webserver.local", "lane"),
    Event("2020-01-22 10:25:34", "logout", "myworkstation.local", "jordan"),
    Event("2020-01-21 08:20:01", "login", "webserver.local", "jordan"),
    Event("2020-01-23 11:24:35", "login", "mailserver.local", "chris"),
]

users = current_users(events)
print(users)

generate_report(users)

# %%
