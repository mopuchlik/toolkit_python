#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 22:23:55 2023

https://www.tutorialspoint.com/python_data_structure/python_linked_lists.htm

@author: michal
"""


# %%
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

# %%


class LinkedList:
    def __init__(self):
        self.head = None

    # print list
    def listprint(self):
        printval = self.head
        while printval is not None:
            print(printval.val)
            printval = printval.next

    # add at the beginning
    def add_begin(self, newval):
        new = Node(newval)
        new.next = self.head
        self.head = new

    # add at the end
    def add_end(self, newval):
        new = Node(newval)

        if self.head is None:
            self.head = new

        lastnode = self.head
        while lastnode.next is not None:
            lastnode = lastnode.next
        lastnode.next = new

    # add after given node
    def add_after(self, afternode, newval):

        if afternode is None:
            print("Nothing to add after")
            return

        new = Node(newval)
        new.next = afternode.next
        afternode.next = new


# %%
llist = LinkedList()

n1 = Node(1)
n2 = Node(2)
n3 = Node(3)

llist.head = n1
n1.next = n2
n2.next = n3
# llist.listprint()

llist.add_begin(0)
# llist.listprint()

llist.add_end(4)
# llist.listprint()

llist.add_after(n1, 11)
llist.listprint()
