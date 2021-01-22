##############################################################
# Rutgers CS 520 Spring 2020
# Assignment 1: Fast Trajectory Replanning
# Fatima AlSaadeh (fya7)
# Ashley Dunn (apd109)


class Node:
    def __init__(self, is_blocked=False, x=0, y=0):
        self.is_blocked = is_blocked
        self.parent = None
        self.next = None
        self.is_seen = False
        self.x = x
        self.y = y
        self.h = 0
        self.g = None
        self.f = 0
        self.search = 0
        self.finished = False

    def __lt__(self, other):
        return self.f < other.f

    def add_child(self, node):
        temp = self.next
        self.next = node
        node.next = temp
