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








