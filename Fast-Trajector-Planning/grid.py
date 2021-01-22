##############################################################
# Rutgers CS 520 Spring 2020
# Assignment 1: Fast Trajectory Replanning
# Fatima AlSaadeh (fya7)
# Ashley Dunn (apd109)


import numpy as np
import random
import tkinter as tk
import sys
from utils import *

BLACK = '#000000'
DARK_GRAY = '#00008B'
LIGHT_GRAY = '#979899'
WHITE = '#ADD8E6'
sys.setrecursionlimit(10 ** 6)


class Grid:
    # Metadata
    gridArr = [[]]  # numpy array
    start = None
    goal = None

    # GUI variables
    numRows = 0
    numCols = 0
    sideLength = 15
    canvas = 0
    width = 0
    height = 0

    def __init__(self):

        self.numCols = 101
        self.numRows = 101

        # self.gridArr = np.array((self.numCols, self.numRows))
        self.gridArr = [[[] for i in range(self.numCols)] for j in range(self.numRows)]
        for x in range(self.numRows):
            for y in range(self.numCols):
                self.gridArr[y][x] = Node(x=x, y=y)

        self._generate_maze()

    # dfs to pick blocked/unblocked cells
    def _dfs(self, x, y, unvisited):

        if len(unvisited) == 0:
            return

        unvisited.remove((x, y))

        points = [(x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)]

        random.shuffle(points)

        for i, j in points:
            if i < 0 or i >= self.numRows or j < 0 or j >= self.numCols:
                continue

            if (i, j) not in unvisited:
                continue

            chance = random.random()

            if chance < .3:
                unvisited.remove((i, j))
                self.gridArr[i][j].is_blocked = True

            else:
                self._dfs(i, j, unvisited)

    # generate points near perimeter for the start or goal
    def _gen_start_goal(self):
        # start point
        x = random.randint(0, 39)
        y = random.randint(0, 39)

        # make sure the start point is near the perimeter
        if x >= 20:
            x -= 20
            x += self.numCols - 20

        if y >= 20:
            y -= 20
            y += self.numRows - 20

        return (x, y)

    # pick valid start/goal points
    def _pick_start_goal(self):

        start = self._gen_start_goal()
        while self.gridArr[start[1]][start[0]].is_blocked:
            start = self._gen_start_goal()

        goal = self._gen_start_goal()
        while self.gridArr[goal[1]][goal[0]].is_blocked:
            goal = self._gen_start_goal()

        a = start[0] - goal[0]
        b = start[1] - goal[1]
        csq = a ** 2 + b ** 2

        while csq < 10000:
            goal = self._gen_start_goal()
            while self.gridArr[goal[1]][goal[0]].is_blocked:
                goal = self._gen_start_goal()
            a = start[0] - goal[0]
            b = start[1] - goal[1]
            csq = a ** 2 + b ** 2


        self.start = self.gridArr[start[1]][start[0]]
        self.goal = self.gridArr[goal[1]][goal[0]]

    # create the overall maze
    def _generate_maze(self):
        unvisited = []

        for a in range(self.numRows):
            for b in range(self.numCols):
                unvisited.append((a, b))

        while len(unvisited) is not 0:
            random.shuffle(unvisited)
            x, y = unvisited[0]

            self._dfs(x, y, unvisited)

        self._pick_start_goal()

    def _draw_cell(self, node, tag, r, c, border):
        if node.is_blocked:
            color = DARK_GRAY
        else:
            color = WHITE

        self.canvas.create_rectangle(
            r * self.sideLength, c * self.sideLength,
            (r + 1) * self.sideLength, (c + 1) * self.sideLength,
            outline=border, fill=color, tag=tag)

        cpt_x = (r * self.sideLength + (r + 1) * self.sideLength) / 2
        cpt_y = (c * self.sideLength + (c + 1) * self.sideLength) / 2

        # Mark Start and Goal Nodes
        if r == self.goal.x and c == self.goal.y:
            self.canvas.create_oval(
                r * self.sideLength + 2, c * self.sideLength + 2,
                (r + 1) * self.sideLength - 2, (c + 1) * self.sideLength - 2,
                fill='red')
        if r == self.start.x and c == self.start.y:
            self.canvas.create_oval(
                r * self.sideLength + 2, c * self.sideLength + 2,
                (r + 1) * self.sideLength - 2, (c + 1) * self.sideLength - 2,
                fill='green')

    # create GUI of maze on its own
    def create_maze(self, root):
        if self.numCols > 50 or self.numRows > 50:
            self.sideLength = 7
        self.width = self.numCols * self.sideLength
        self.height = self.numRows * self.sideLength
        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        root.wm_title("Maze\n")
        for c in range(self.numRows):
            for r in range(self.numCols):
                node = self.gridArr[c][r]
                tag = str(r) + " " + str(c)
                self._draw_cell(node, tag, r, c, BLACK)
        self.canvas.pack()

    def _draw_path(self, root, point):
        x = point.x
        y = point.y

        color ='red'

        if x == self.start.x and y == self.start.y:
            color = "green"
        if x == self.goal.x and y == self.goal.y:
            color = "red"
        f = tk.Frame(root, height=self.sideLength + 1, width=self.sideLength + 1)
        f.pack_propagate(0)
        b = tk.Button(f, bg=color, bd=1, command=lambda: self._print(point))
        f.pack()
        b.pack()
        f.place(x=x * self.sideLength, y=y * self.sideLength)

    # prints information about cell on path
    def _print(self, info):
        x, y = info.x, info.y
        node = self.gridArr[y][x]
        temp = ""
        if x == self.start.x and y == self.start.y:
            temp = temp + "Start:	"
        if x == self.goal.x and y == self.goal.y:
            temp = temp + "Goal:	"
        temp = temp + "x=" + str(x) + " y=" + str(y)
        print(temp)
        '''
        if node.is_blocked:
            temp = "Space Type: Blocked"
        else:
            temp = "Space Type: Unblocked"
        print(temp)
        '''
        print("h=%f\tg=%f\tf=%f\tisb=%f" % (info.h, info.g, info.f, info.is_blocked))
        print

    # create GUI of maze with a path displayed on it
    def display_path(self, root, pathInfo):
        self.create_maze(root)
        for i in range(len(pathInfo)):
            self._draw_path(root, pathInfo[i])
        self.canvas.pack()

    # print the grid; probably not useful
    def print_grid(self, f):
        for y in range(self.h):
            for x in range(self.w):
                f.write(self.gridArr[y][x] + ' ')
            f.write('\n')


def main():
    g = Grid()
    root = tk.Tk()
    g.create_maze(root)

    root.mainloop()


if __name__ == "__main__":
    main()
