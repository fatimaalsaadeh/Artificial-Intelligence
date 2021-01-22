##############################################################
# Rutgers CS 520 Spring 2020
# Assignment 1: Fast Trajectory Replanning
# Fatima AlSaadeh (fya7)
# Ashley Dunn (apd109)


import heapq
import time
from grid import *


def manhattan_distance(start, goal):
    return abs(start.x - goal.x) + abs(start.y - goal.y)


class Algorithm:
    def __init__(self, grid):
        self.grid = None
        self.grid_info = grid.gridArr
        self.counter = 0
        self.delta = []
        self.goal = grid.goal
        self.start = grid.start
        self.reached = None

    def neighbors(self, start, seen=False):
        neighborslist = []
        (x, y) = start
        points = [(x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)]
        i = 0
        k = 4
        while i < k:
            if points[i][0] < 0 or points[i][0] >= 101 or points[i][1] < 0 or points[i][1] >= 101:
                points.remove(points[i])
                i -= 1
                k -= 1
            else:
                position = self.grid_info[points[i][1]][points[i][0]]
                neighborslist.append(position)
                if (seen and position.is_blocked):
                    position.is_seen = True
            i += 1
        return points

    def initialize_node(self, cur_node, delta, counter):
        cur_node_o = self.grid_info[cur_node[1]][cur_node[0]]
        if (cur_node_o.search != self.counter and cur_node_o.search != 0):
            if (cur_node_o.g + cur_node_o.h < cur_node_o.g):
                cur_node_o.h = self.goal.g - cur_node_o.g
            cur_node_o.h = max(manhattan_distance(cur_node_o, self.goal), cur_node_o.h)
            cur_node_o.g = sys.maxsize
        elif (cur_node_o.search == 0):
            cur_node_o.h = manhattan_distance(cur_node_o, self.goal)
            cur_node_o.g = sys.maxsize
        cur_node_o.search = counter
        self.grid_info[cur_node[1]][cur_node[0]] = cur_node_o

    def adaptive_astar(self, steps, grid_o, tie_break, show):

        open_set = []
        closed_set = []
        path = {}
        cost = {}
        counter = 0
        delta = []
        delta.append(counter)
        reached = False
        final_path = {}

        start = (self.start.x, self.start.y)
        goal = (self.goal.x, self.goal.y)
        heapq.heappush(open_set, (0, start))
        path[start] = None
        cost[start] = 0
        snode = self.grid_info[start[1]][start[0]]
        gnode = self.grid_info[goal[1]][goal[0]]
        snode.g = 0
        snode.h = manhattan_distance(snode, gnode)
        snode.f = snode.g + snode.h
        numexpanded = 0
        current = heapq.heappop(open_set)[1]
        numexpanded = numexpanded + 1
        start_time = time.time()

        goal_g = 0

        while current != goal:
            path = {}
            counter += 1
            cnode = self.grid_info[current[1]][current[0]]
            cnode.search = counter
            if goal_g > 0 and cnode.g:
                cnode.h = goal_g - cnode.g
            else:
                cnode.h = manhattan_distance(cnode, gnode)
            cnode.g = 0
            cnode.f = cnode.h + cnode.g

            gnode.g = sys.maxsize
            gnode.search = counter
            open_set = []
            closed_set = [final_path.keys()]
            heapq.heappush(open_set, (cnode.f, current))
            while open_set and gnode.g > open_set[0][0]:
                numexpanded = numexpanded + 1
                current = heapq.heappop(open_set)[1]
                cnode = self.grid_info[current[1]][current[0]]
                closed_set.append(current)
                for next in self.neighbors(current):
                    if next not in closed_set:
                        nextnode = self.grid_info[next[1]][next[0]]
                        if nextnode.search < counter:
                            if goal_g > 0 and nextnode.g and nextnode.search == counter - 1:
                                nextnode.h = goal_g - nextnode.g
                            nextnode.g = sys.maxsize
                            nextnode.search = counter

                        costCur = cnode.g + (sys.maxsize if nextnode.is_seen else 1)
                        if nextnode.g is None or costCur < nextnode.g:
                            if (nextnode.f, next) in open_set:
                                open_set.remove((nextnode.f, next))
                                heapq.heapify(open_set)

                            if nextnode.h == 0:
                                nextnode.h = manhattan_distance(gnode, nextnode)
                            nextnode.g = costCur
                            nextnode.f = nextnode.g + nextnode.h
                            nextnode.f = (tie_break * nextnode.f) - nextnode.g
                            heapq.heappush(open_set, (nextnode.f, next))
                            path[next] = current
            if not open_set:
                print("blocked target")
                r = tk.Tk()
                grid_o.create_maze(r)
                reached = False
                break
            elif goal in path:
                reached = True
                goal_g = gnode.g
            else:
                goal_g = 0

            cur_p = goal
            pathOrder = []
            path = {**path, **final_path}

            while cur_p is not None and reached:
                x, y = cur_p
                if any(r.x == x and r.y == y for r in pathOrder):
                    break
                pathOrder.insert(0, self.grid_info[y][x])
                if (cur_p in path):
                    cur_p = path[cur_p]
                else:
                    cur_p = None

            # move
            located = pathOrder[0]
            invalid = False
            for i in range(1, len(pathOrder)):
                n = self.neighbors((located.x, located.y), True)
                potential_next = pathOrder[i]
                if potential_next.is_seen:
                    if steps:
                        r = tk.Tk()
                        grid_o.display_path(r, pathOrder[:i][::-1])
                        r.title("intermediate representation")
                        r.mainloop()

                    break
                else:
                    final_path[(potential_next.x, potential_next.y)] = (located.x, located.y)
                    located = potential_next

            current = (located.x, located.y)

            if current == goal:
                cur_p = goal
                pathOrder = []
                while cur_p is not None and reached:
                    x, y = cur_p
                    if any(r.x == x and r.y == y for r in pathOrder):
                        break
                    pathOrder.insert(0, self.grid_info[y][x])
                    if (cur_p in final_path):
                        cur_p = final_path[cur_p]
                    else:
                        cur_p = None
                reached = True
            else:
                reached = False

        if reached:
            end_time = time.time()
            total_time = end_time - start_time
            if show:
                r = tk.Tk()
                grid_o.display_path(r, pathOrder[::-1])
                r.title("final representation")
                r.mainloop()
            # return reached, pathOrder[::-1], self.grid_info[goal[1]][goal[0]].g, len(cost)
            return total_time, numexpanded
        return 0, 0

    def repeated_astar(self, steps, grid_o, is_forward, is_backward, tie_break, show):

        open_set = []
        closed_set = []
        path = {}
        cost = {}
        counter = 0
        delta = []
        delta.append(counter)
        reached = False
        prev = None
        final_path = {}

        if is_forward:
            start = (self.start.x, self.start.y)
            goal = (self.goal.x, self.goal.y)
            heapq.heappush(open_set, (0, start))
            path[start] = None
            cost[start] = 0
            snode = self.grid_info[start[1]][start[0]]
            gnode = self.grid_info[goal[1]][goal[0]]
            snode.g = 0
            snode.h = manhattan_distance(snode, gnode)
            snode.f = snode.g + snode.h
            numexpanded = 0
            current = heapq.heappop(open_set)[1]
            numexpanded = numexpanded + 1
            start_time = time.time()

            while current != goal:
                path = {}
                counter += 1
                cnode = self.grid_info[current[1]][current[0]]
                cnode.search = counter
                cnode.g = 0
                cnode.h = manhattan_distance(cnode, gnode)
                cnode.f = cnode.h + cnode.g

                gnode.g = sys.maxsize
                gnode.search = counter
                open_set = []
                closed_set = [final_path.keys()]
                heapq.heappush(open_set, (cnode.f, current))
                while open_set and gnode.g > open_set[0][0]:
                    numexpanded = numexpanded + 1
                    current = heapq.heappop(open_set)[1]
                    cnode = self.grid_info[current[1]][current[0]]
                    closed_set.append(current)
                    for next in self.neighbors(current):
                        if next not in closed_set:
                            nextnode = self.grid_info[next[1]][next[0]]
                            if nextnode.search < counter:
                                nextnode.g = sys.maxsize
                                nextnode.search = counter
                            costCur = cnode.g + (sys.maxsize if nextnode.is_seen else 1)
                            if nextnode.g is None or costCur < nextnode.g:
                                if (nextnode.f, next) in open_set:
                                    open_set.remove((nextnode.f, next))
                                    heapq.heapify(open_set)

                                nextnode.h = manhattan_distance(gnode, nextnode)
                                nextnode.g = costCur
                                nextnode.f = nextnode.g + nextnode.h
                                nextnode.f = (tie_break * nextnode.f) - nextnode.g
                                heapq.heappush(open_set, (nextnode.f, next))
                                path[next] = current
                if not open_set:
                    print("blocked target")
                    r = tk.Tk()
                    grid_o.create_maze(r)
                    reached = False
                    break
                elif goal in path:
                    reached = True

                cur_p = goal
                pathOrder = []
                path = {**path, **final_path}

                while cur_p is not None and reached:
                    x, y = cur_p
                    if any(r.x == x and r.y == y for r in pathOrder):
                        break
                    pathOrder.insert(0, self.grid_info[y][x])
                    if (cur_p in path):
                        cur_p = path[cur_p]
                    else:
                        cur_p = None

                # move
                located = pathOrder[0]
                invalid = False
                for i in range(1, len(pathOrder)):
                    n = self.neighbors((located.x, located.y), True)
                    potential_next = pathOrder[i]
                    if potential_next.is_seen:
                        if steps:
                            r = tk.Tk()
                            grid_o.display_path(r, pathOrder[:i][::-1])
                            r.title("intermediate representation")
                            r.mainloop()

                        break
                    else:
                        final_path[(potential_next.x, potential_next.y)] = (located.x, located.y)
                        located = potential_next

                current = (located.x, located.y)

                if current == goal:
                    cur_p = goal
                    pathOrder = []
                    while cur_p is not None and reached:
                        x, y = cur_p
                        if any(r.x == x and r.y == y for r in pathOrder):
                            break
                        pathOrder.insert(0, self.grid_info[y][x])
                        if (cur_p in final_path):
                            cur_p = final_path[cur_p]
                        else:
                            cur_p = None
                    reached = True
                else:
                    reached = False

            if reached:
                end_time = time.time()
                total_time = end_time - start_time
                if show:
                    r = tk.Tk()
                    grid_o.display_path(r, pathOrder[::-1])
                    r.title("final representation")
                    r.mainloop()
                # return reached, pathOrder[::-1], self.grid_info[goal[1]][goal[0]].g, len(cost)
                return total_time, numexpanded
            return 0, 0

        if is_backward:
            start = (self.goal.x, self.goal.y)
            goal = (self.start.x, self.start.y)
            heapq.heappush(open_set, (0, start))
            path[start] = None
            cost[start] = 0
            snode = self.grid_info[start[1]][start[0]]
            gnode = self.grid_info[goal[1]][goal[0]]
            snode.g = 0
            snode.h = manhattan_distance(snode, gnode)
            snode.f = snode.g + snode.h
            numexpanded = 0
            current = heapq.heappop(open_set)[1]
            numexpanded = numexpanded + 1
            start_time = time.time()
            while current != goal:
                path = {}
                counter += 1
                cnode = self.grid_info[current[1]][current[0]]
                cnode.search = counter
                cnode.g = 0
                cnode.h = manhattan_distance(cnode, gnode)
                cnode.f = cnode.h + cnode.g

                gnode.g = sys.maxsize
                gnode.search = counter
                open_set = []
                closed_set = [final_path.keys()]
                heapq.heappush(open_set, (cnode.f, current))
                while open_set and gnode.g > open_set[0][0]:
                    numexpanded = numexpanded + 1
                    current = heapq.heappop(open_set)[1]
                    cnode = self.grid_info[current[1]][current[0]]
                    closed_set.append(current)
                    for next in self.neighbors(current):
                        if next not in closed_set:
                            nextnode = self.grid_info[next[1]][next[0]]
                            if nextnode.search < counter:
                                nextnode.g = sys.maxsize
                                nextnode.search = counter
                            costCur = cnode.g + (sys.maxsize if nextnode.is_seen else 1)
                            if nextnode.g is None or costCur < nextnode.g:
                                if (nextnode.f, next) in open_set:
                                    open_set.remove((nextnode.f, next))
                                    heapq.heapify(open_set)
                                nextnode.h = manhattan_distance(gnode, nextnode)
                                nextnode.g = costCur
                                nextnode.f = nextnode.g + nextnode.h
                                nextnode.f = (tie_break * nextnode.f) - nextnode.g
                                heapq.heappush(open_set, (nextnode.f, next))
                                path[next] = current
                if not open_set:
                    print("blocked target")
                    r = tk.Tk()
                    grid_o.create_maze(r)
                    reached = False
                    break
                elif goal in path:
                    reached = True

                cur_p = goal
                pathOrder = []
                path = {**path, **final_path}

                while cur_p is not None and reached:
                    x, y = cur_p
                    if any(r.x == x and r.y == y for r in pathOrder):
                        break
                    pathOrder.insert(0, self.grid_info[y][x])
                    if cur_p in path:
                        cur_p = path[cur_p]
                    else:
                        cur_p = None

                # move
                located = pathOrder[0]
                invalid = False
                for i in range(1, len(pathOrder)):
                    n = self.neighbors((located.x, located.y), True)
                    potential_next = pathOrder[i]
                    if potential_next.is_seen:
                        if steps:
                            r = tk.Tk()
                            grid_o.display_path(r, pathOrder[:i][::-1])
                            r.title("intermediate representation")
                            r.mainloop()

                        break
                    else:
                        final_path[(potential_next.x, potential_next.y)] = (located.x, located.y)
                        located = potential_next

                current = (located.x, located.y)

                if current == goal:
                    cur_p = goal
                    pathOrder = []
                    while cur_p is not None and reached:
                        x, y = cur_p
                        if any(r.x == x and r.y == y for r in pathOrder):
                            break
                        pathOrder.insert(0, self.grid_info[y][x])
                        if cur_p in final_path:
                            cur_p = final_path[cur_p]
                        else:
                            cur_p = None
                    reached = True
                else:
                    reached = False

            if reached:
                if show:
                    r = tk.Tk()
                    grid_o.display_path(r, pathOrder[::-1])
                    r.title("final representation")
                    r.mainloop()
                end_time = time.time()
                total_time = end_time - start_time

                #     return reached, pathOrder[::-1], self.grid_info[goal[1]][goal[0]].g, len(cost)
                # return reached, pathOrder[::-1], sys.maxsize, len(cost)
                return total_time, numexpanded
            return 0, 0


def main():
    steps = False
    f = False
    b = True
    show = True

    avg_time_repeated_f = 0;
    avg_expanded_repeated_f = 0;
    count = 50
    for i in range(count):
        grid_o = Grid()
        alg = Algorithm(grid_o)
        total_time, num_expanded = alg.repeated_astar(steps, grid_o, f, b, 200, show)
        #total_time, num_expanded = alg.adaptive_astar(steps, grid_o, 15, show)
        if total_time == 0:
            count = count - 1
            continue
        print("expanded", num_expanded)
        print("time", total_time)
        avg_expanded_repeated_f += num_expanded
        avg_time_repeated_f += total_time

    avg_time_repeated_f = avg_time_repeated_f / count
    avg_expanded_repeated_f = avg_expanded_repeated_f / count

    print(avg_expanded_repeated_f)
    print(avg_time_repeated_f)


if __name__ == "__main__":
    main()
