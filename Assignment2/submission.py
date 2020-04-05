# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from Assignment2.explorable_graph import ExplorableGraph
import heapq
import math


class PriorityQueue(object):
    def __init__(self, queue=[]):
        self.queue = queue
        self.idx = 0

    def pop(self):
        return heapq.heappop(self.queue)[-1]

    def remove(self, node_id):
        self.queue.pop(node_id)
        heapq.heapify(self.queue)

    def append(self, node: tuple):
        heapq.heappush(self.queue, (node[0], self.idx, node))
        self.idx += 1

    def size(self):
        return len(self.queue)

    def clear(self):
        self.queue = []

    def top(self):
        return self.queue[0] if self.queue else [0]

    def __iter__(self):
        return iter(sorted(self.queue))

    def __str__(self):
        return 'PQ:%s' % self.queue

    def __contains__(self, key):
        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        return self.queue == other.queue


def breadth_first_search(graph, start, goal):
    """
    Returns: The best path as a list from the start and goal nodes (including both).
    """
    if start == goal or start not in graph.explored_nodes.keys() or goal not in graph.explored_nodes.keys():
        return []

    frontier = [start]
    path = {start: [start]}

    while frontier:
        start = frontier.pop(0)
        for node in sorted(graph.neighbors(start)):
            if not graph.explored_nodes[node] and node not in frontier:
                path[node] = path[start] + [node]
                frontier.append(node)
            if node == goal:
                return path[goal]

    return []


def uniform_cost_search(graph, start, goal):
    """
    Returns: The best path as a list from the start and goal nodes (including both).
    """
    if start == goal or start not in graph.explored_nodes.keys() or goal not in graph.explored_nodes.keys():
        return []

    pq = PriorityQueue([])
    pq.append((0, start))
    path = {start: [start]}
    node_weight = {start: 0}

    while pq.size():
        weight, start = pq.pop()
        if start == goal:
            return path[goal]
        for node in graph.neighbors(start):
            if not graph.explored_nodes[node]:
                cost = graph.get_edge_weight(start, node)
                if node_weight.get(node, float('inf')) > weight + cost:
                    node_weight[node] = weight + cost
                    path[node] = path[start] + [node]
                    pq.append((weight + cost, node))

    return []


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.
    Returns: 0
    """
    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Returns: Euclidean distance between `v` node and `goal` node
    """
    x1, y1 = graph.nodes[v]['pos']
    x2, y2 = graph.nodes[goal]['pos']
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    if start == goal or start not in graph.explored_nodes.keys() or goal not in graph.explored_nodes.keys():
        return []

    pq = PriorityQueue([])
    pq.append((0, 0, start))
    path = {start: [start]}
    weights = {start: 0}

    while pq.size():
        f, dist, start = pq.pop()
        if start == goal:
            return path[goal]
        for node in graph.neighbors(start):
            if not graph.explored_nodes[node]:
                g = dist + graph.get_edge_weight(start, node)
                h = heuristic(graph, node, goal)
                if weights.get(node, float('inf')) > g + h:
                    weights[node] = g + h
                    path[node] = path[start] + [node]
                    pq.append((g + h, g, node))

    return []


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.
    """
    if start == goal or start not in graph.explored_nodes.keys() or goal not in graph.explored_nodes.keys():
        return []

    pq1 = PriorityQueue([])
    pq2 = PriorityQueue([])
    pq1.append((0, start))
    pq2.append((0, goal))

    path1 = {start: []}
    path2 = {goal: []}
    weight1 = {start: 0}
    weight2 = {goal: 0}

    explored1 = set()
    explored2 = set()

    res = []
    min_cost = float('inf')

    while pq1.size() and pq2.size():
        w1, n1 = pq1.pop()
        explored1.add(n1)
        if n1 in path2.keys() and weight1[n1] + weight2[n1] < min_cost:
            res = path1[n1] + [n1] + path2[n1]
            min_cost = weight1[n1] + weight2[n1]
        for node in graph.neighbors(n1):
            if node not in explored1:
                cost = graph.get_edge_weight(n1, node) + w1
                if weight1.get(node, float('inf')) > cost:
                    weight1[node] = cost
                    pq1.append((cost, node))
                    path1[node] = path1[n1] + [n1]

        w2, n2 = pq2.pop()
        explored2.add(n2)
        if n2 in path1.keys() and weight1[n2] + weight2[n2] < min_cost:
            res = path1[n2] + [n2] + path2[n2]
            min_cost = weight1[n2] + weight2[n2]
        for node in graph.neighbors(n2):
            if node not in explored2:
                cost = graph.get_edge_weight(node, n2) + w2
                if weight2.get(node, float('inf')) > cost:
                    weight2[node] = cost
                    pq2.append((cost, node))
                    path2[node] = [n2] + path2[n2]

        if pq1.top()[0] + pq2.top()[0] >= min_cost:
            return res

    return []


def bidirectional_a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.
    """
    if start == goal or start not in graph.explored_nodes.keys() or goal not in graph.explored_nodes.keys():
        return []

    pq1, pq2 = PriorityQueue([]), PriorityQueue([])
    pq1.append((0, 0, start))
    pq2.append((0, 0, goal))

    path1, path2 = {start: []}, {goal: []}
    weight1, weight2 = {start: 0}, {goal: 0}

    explored1, explored2 = set(), set()

    res = []
    min_cost = float('inf')
    n0 = ''

    while pq1.size() and pq2.size():
        f1, g1, n1 = pq1.pop()
        explored1.add(n1)
        if n1 in path2.keys() and weight1[n1] + weight2[n1] < min_cost:
            n0 = n1
            min_cost = weight1[n1] + weight2[n1]
            res = path1[n1] + [n1] + path2[n1]
        for node in graph.neighbors(n1):
            if node not in explored1:
                g = graph.get_edge_weight(n1, node) + g1
                f = heuristic(graph, node, goal) + g
                if weight1.get(node, float('inf')) > g:
                    weight1[node] = g
                    pq1.append((f, g, node))
                    path1[node] = path1[n1] + [n1]

        f1, g1, n1 = pq1.top()[-1]
        f2, g2, n2 = pq2.top()[-1]
        if n0 and weight1[n1] + weight2[n2] >= min_cost - 0.5 * heuristic(graph, n1, n2):
            return res

        f2, g2, n2 = pq2.pop()
        explored2.add(n2)
        if n2 in path1.keys() and weight1[n2] + weight2[n2] < min_cost:
            n0 = n2
            min_cost = weight1[n2] + weight2[n2]
            res = path1[n2] + [n2] + path2[n2]
        for node in graph.neighbors(n2):
            if node not in explored2:
                g = graph.get_edge_weight(node, n2) + g2
                f = heuristic(graph, start, node) + g
                if weight2.get(node, float('inf')) > g:
                    weight2[node] = g
                    pq2.append((f, g, node))
                    path2[node] = [n2] + path2[n2]

        f1, g1, n1 = pq1.top()[-1]
        f2, g2, n2 = pq2.top()[-1]
        if n0 and weight1[n1] + weight2[n2] >= min_cost - 0.5 * heuristic(graph, n1, n2):
            return res

    return []


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search
    """

    def helper(start, goal):
        if start == goal or start not in graph.explored_nodes.keys() or goal not in graph.explored_nodes.keys():
            return []

        pq1, pq2 = PriorityQueue([]), PriorityQueue([])
        pq1.append((0, start))
        pq2.append((0, goal))

        path1, path2 = {start: []}, {goal: []}
        weight1, weight2 = {start: 0}, {goal: 0}

        explored1, explored2 = set(), set()

        res = []
        min_cost = float('inf')

        while pq1.size() and pq2.size():
            w1, n1 = pq1.pop()
            explored1.add(n1)
            if n1 in path2.keys() and weight1[n1] + weight2[n1] < min_cost:
                res = path1[n1] + [n1] + path2[n1]
                min_cost = weight1[n1] + weight2[n1]
            for node in graph.neighbors(n1):
                if node not in explored1:
                    cost = graph.get_edge_weight(n1, node) + w1
                    if weight1.get(node, float('inf')) > cost:
                        weight1[node] = cost
                        pq1.append((cost, node))
                        path1[node] = path1[n1] + [n1]

            w2, n2 = pq2.pop()
            explored2.add(n2)
            if n2 in path1.keys() and weight1[n2] + weight2[n2] < min_cost:
                res = path1[n2] + [n2] + path2[n2]
                min_cost = weight1[n2] + weight2[n2]
            for node in graph.neighbors(n2):
                if node not in explored2:
                    cost = graph.get_edge_weight(node, n2) + w2
                    if weight2.get(node, float('inf')) > cost:
                        weight2[node] = cost
                        pq2.append((cost, node))
                        path2[node] = [n2] + path2[n2]

            if pq1.top()[0] + pq2.top()[0] >= min_cost:
                return res

        return []

    v = set(goals)
    if len(v) <= 1:
        return []
    if len(v) == 2:
        v1, v2 = v
        return helper(v1, v2)
    for vi in v:
        if vi not in graph.explored_nodes.keys():
            return []

    v1, v2, v3 = goals

    pq1, pq2, pq3 = PriorityQueue([]), PriorityQueue([]), PriorityQueue([])
    pq1.append((0, v1))
    pq2.append((0, v2))
    pq3.append((0, v3))

    path1, path2, path3 = {v1: []}, {v2: []}, {v3: []}
    weight1, weight2, weight3 = {v1: 0}, {v2: 0}, {v3: 0}

    explored1, explored2, explored3 = set(), set(), set()

    min_cost12 = min_cost13 = min_cost23 = float('inf')
    res12, res13, res23 = [], [], []

    def res():
        ab, bc, ac = set(res12), set(res23), set(res13)
        if min_cost12 < min_cost23 and min_cost13 < min_cost23:
            if ac.issubset(ab):
                return res12
            elif ab.issubset(ac):
                return res13
            else:
                return res12[::-1] + res13[1:]

        if min_cost12 < min_cost13 and min_cost23 < min_cost13:
            if ab.issubset(bc):
                return res23
            elif bc.issubset(ab):
                return res12
            else:
                return res12 + res23[1:]

        if min_cost23 < min_cost12 and min_cost13 < min_cost12:
            if ac.issubset(bc):
                return res23
            elif bc.issubset(ac):
                return res13
            else:
                return res13[:-1] + res23[::-1]

        return []

    while pq1.size() and pq2.size() and pq3.size():
        w1, n1 = pq1.pop()
        explored1.add(w1)
        if w1 < min_cost12 or w1 < min_cost13:
            if n1 in path2.keys() and weight1[n1] + weight2[n1] < min_cost12:
                min_cost12 = weight1[n1] + weight2[n1]
                res12 = path1[n1] + [n1] + path2[n1][::-1]
            if n1 in path3.keys() and weight1[n1] + weight3[n1] < min_cost13:
                min_cost13 = weight1[n1] + weight3[n1]
                res13 = path1[n1] + [n1] + path3[n1][::-1]
            for node in graph.neighbors(n1):
                if node not in explored1:
                    dist = graph.get_edge_weight(n1, node) + w1
                    if weight1.get(node, float('inf')) > dist:
                        weight1[node] = dist
                        pq1.append((dist, node))
                        path1[node] = path1[n1] + [n1]
        a, b, c = pq1.top()[0], pq2.top()[0], pq3.top()[0]
        if min(a + b, b + c) >= min_cost12 and min(a + c, b + c) >= min_cost13:
            return res()

        w2, n2 = pq2.pop()
        explored2.add(n2)
        if w2 < min_cost12 or w2 < min_cost23:
            if n2 in path1.keys() and weight1[n2] + weight2[n2] < min_cost12:
                min_cost12 = weight1[n2] + weight2[n2]
                res12 = path1[n2] + [n2] + path2[n2][::-1]
            if n2 in path3.keys() and weight2[n2] + weight3[n2] < min_cost23:
                min_cost23 = weight2[n2] + weight3[n2]
                res23 = path2[n2] + [n2] + path3[n2][::-1]
            for node in graph.neighbors(n2):
                if node not in explored2:
                    dist = graph.get_edge_weight(n2, node) + w2
                    if weight2.get(node, float('inf')) > dist:
                        weight2[node] = dist
                        pq2.append((dist, node))
                        path2[node] = path2[n2] + [n2]
        a, b, c = pq1.top()[0], pq2.top()[0], pq3.top()[0]
        if min(a + c, a + b) >= min_cost12 and min(a + c, b + c) >= min_cost23:
            return res()

        w3, n3 = pq3.pop()
        explored3.add(n3)
        if w3 < min_cost13 or w3 < min_cost23:
            if n3 in path1.keys() and weight1[n3] + weight3[n3] < min_cost13:
                min_cost13 = weight1[n3] + weight3[n3]
                res13 = path1[n3] + [n3] + path3[n3][::-1]
            if n3 in path2.keys() and weight2[n3] + weight3[n3] < min_cost23:
                min_cost23 = weight2[n3] + weight3[n3]
                res23 = path2[n3] + [n3] + path3[n3][::-1]
            for node in graph.neighbors(n3):
                if node not in explored3:
                    dist = graph.get_edge_weight(n3, node) + w3
                    if weight3.get(node, float('inf')) > dist:
                        weight3[node] = dist
                        pq3.append((dist, node))
                        path3[node] = path3[n3] + [n3]
        a, b, c = pq1.top()[0], pq2.top()[0], pq3.top()[0]
        if min(a + b, a + c) >= min_cost13 and min(a + b, b + c) >= min_cost23:
            return res()

    return []


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 4: Upgraded Tridirectional Search
    """

    def helper(start, goal):
        if start == goal or start not in graph.explored_nodes.keys() or goal not in graph.explored_nodes.keys():
            return []

        pq1, pq2 = PriorityQueue([]), PriorityQueue([])
        pq1.append((0, 0, start))
        pq2.append((0, 0, goal))

        path1, path2 = {start: []}, {goal: []}
        weight1, weight2 = {start: 0}, {goal: 0}

        explored1, explored2 = set(), set()

        res = []
        min_cost = float('inf')
        n0 = ''

        while pq1.size() and pq2.size():
            f1, g1, n1 = pq1.pop()
            explored1.add(n1)
            if n1 in path2.keys() and weight1[n1] + weight2[n1] < min_cost:
                n0 = n1
                min_cost = weight1[n1] + weight2[n1]
                res = path1[n1] + [n1] + path2[n1]
            for node in graph.neighbors(n1):
                if node not in explored1:
                    g = graph.get_edge_weight(n1, node) + g1
                    f = heuristic(graph, node, goal) + g
                    if weight1.get(node, float('inf')) > g:
                        weight1[node] = g
                        pq1.append((f, g, node))
                        path1[node] = path1[n1] + [n1]

            f1, g1, n1 = pq1.top()[-1]
            f2, g2, n2 = pq2.top()[-1]
            if n0 and weight1[n1] + weight2[n2] >= min_cost - 0.5 * heuristic(graph, n1, n2):
                return res

            f2, g2, n2 = pq2.pop()
            explored2.add(n2)
            if n2 in path1.keys() and weight1[n2] + weight2[n2] < min_cost:
                n0 = n2
                min_cost = weight1[n2] + weight2[n2]
                res = path1[n2] + [n2] + path2[n2]
            for node in graph.neighbors(n2):
                if node not in explored2:
                    g = graph.get_edge_weight(node, n2) + g2
                    f = heuristic(graph, start, node) + g
                    if weight2.get(node, float('inf')) > g:
                        weight2[node] = g
                        pq2.append((f, g, node))
                        path2[node] = [n2] + path2[n2]

            f1, g1, n1 = pq1.top()[-1]
            f2, g2, n2 = pq2.top()[-1]
            if n0 and weight1[n1] + weight2[n2] >= min_cost - 0.5 * heuristic(graph, n1, n2):
                return res

        return []

    v = set(goals)
    if len(v) <= 1:
        return []
    if len(v) == 2:
        v1, v2 = v
        return helper(v1, v2)
    for vi in v:
        if vi not in graph.explored_nodes.keys():
            return []

    v1, v2, v3 = goals

    pq1, pq2, pq3 = PriorityQueue([]), PriorityQueue([]), PriorityQueue([])
    pq1.append((0, 0, v1))
    pq2.append((0, 0, v2))
    pq3.append((0, 0, v3))

    path1, path2, path3 = {v1: []}, {v2: []}, {v3: []}
    weight1, weight2, weight3 = {v1: 0}, {v2: 0}, {v3: 0}

    explored1, explored2, explored3 = set(), set(), set()

    min_cost12 = min_cost13 = min_cost23 = float('inf')
    res12, res13, res23 = [], [], []

    def res():
        ab, bc, ac = set(res12), set(res23), set(res13)
        if min_cost12 < min_cost23 and min_cost13 < min_cost23:
            if ac.issubset(ab):
                return res12
            elif ab.issubset(ac):
                return res13
            else:
                return res12[::-1] + res13[1:]

        if min_cost12 < min_cost13 and min_cost23 < min_cost13:
            if ab.issubset(bc):
                return res23
            elif bc.issubset(ab):
                return res12
            else:
                return res12 + res23[1:]

        if min_cost23 < min_cost12 and min_cost13 < min_cost12:
            if ac.issubset(bc):
                return res23
            elif bc.issubset(ac):
                return res13
            else:
                return res13[:-1] + res23[::-1]

        return []

    while pq1.size() and pq2.size() and pq3.size():
        f1, g1, n1 = pq1.pop()
        explored1.add(n1)
        if g1 < min_cost12:
            if n1 in path2.keys() and weight1[n1] + weight2[n1] < min_cost12:
                min_cost12 = weight1[n1] + weight2[n1]
                res12 = path1[n1] + [n1] + path2[n1][::-1]
            if n1 in path3.keys() and weight1[n1] + weight3[n1] < min_cost13:
                min_cost13 = weight1[n1] + weight3[n1]
                res13 = path1[n1] + [n1] + path3[n1][::-1]
            for node in graph.neighbors(n1):
                reach = min(heuristic(graph,node,v1), heuristic(graph,node,v2))
                if reach > heuristic(graph,n1,v1)+graph.get_edge_weight(n1, node):
                    continue
                if node not in explored1:
                    g = graph.get_edge_weight(n1, node) + g1
                    f = heuristic(graph, node, v2) + g
                    if weight1.get(node, float('inf')) > g:
                        weight1[node] = g
                        pq1.append((f, g, node))
                        path1[node] = path1[n1] + [n1]
        else:
            pq1.append((f1, g1, n1))
        f1, g1, n1 = pq1.top()[-1]
        f2, g2, n2 = pq2.top()[-1]
        f3, g3, n3 = pq3.top()[-1]
        if g1 >= min_cost12 - heuristic(graph, n1, v2) and \
                g3 >= min_cost13 - heuristic(graph, n3, v1) and \
                g2 >= min_cost23 - heuristic(graph, n2, v3):
            return res()

        f2, g2, n2 = pq2.pop()
        explored2.add(n2)
        if g2 < min_cost23:
            if n2 in path1.keys() and weight1[n2] + weight2[n2] < min_cost12:
                min_cost12 = weight1[n2] + weight2[n2]
                res12 = path1[n2] + [n2] + path2[n2][::-1]
            if n2 in path3.keys() and weight2[n2] + weight3[n2] < min_cost23:
                min_cost23 = weight2[n2] + weight3[n2]
                res23 = path2[n2] + [n2] + path3[n2][::-1]
            for node in graph.neighbors(n2):
                reach = min(heuristic(graph,node,v2), heuristic(graph,node, v3))
                if reach > heuristic(graph,n2,v2) + graph.get_edge_weight(node, n2):
                    continue
                if node not in explored2:
                    g = graph.get_edge_weight(node, n2) + g2
                    f = heuristic(graph, node, v3) + g
                    if weight2.get(node, float('inf')) > g:
                        weight2[node] = g
                        pq2.append((f, g, node))
                        path2[node] = path2[n2] + [n2]
        else:
            pq2.append((f2, g2, n2))
        f1, g1, n1 = pq1.top()[-1]
        f2, g2, n2 = pq2.top()[-1]
        f3, g3, n3 = pq3.top()[-1]
        # if g1 >= min_cost12 and g3 >= min_cost13 and g2 >= min_cost23:
        if g1 >= min_cost12 - heuristic(graph, n1, v2) and \
                g3 >= min_cost13 - heuristic(graph, n3, v1) and \
                g2 >= min_cost23 - heuristic(graph, n2, v3):
            return res()

        f3, g3, n3 = pq3.pop()
        explored3.add(n3)
        if g3 < min_cost13:
            if n3 in path1.keys() and weight1[n3] + weight3[n3] < min_cost13:
                min_cost13 = weight1[n3] + weight3[n3]
                res13 = path1[n3] + [n3] + path3[n3][::-1]
            if n3 in path2.keys() and weight2[n3] + weight3[n3] < min_cost23:
                min_cost23 = weight2[n3] + weight3[n3]
                res23 = path2[n3] + [n3] + path3[n3][::-1]
            for node in graph.neighbors(n3):
                reach = min(heuristic(graph,node,v1), heuristic(graph,node, v3))
                if reach > heuristic(graph,n3,v3) + graph.get_edge_weight(node, n3):
                    continue
                if node not in explored3:
                    g = graph.get_edge_weight(node, n3) + g3
                    f = heuristic(graph, node, v1) + g
                    if weight3.get(node, float('inf')) > g:
                        weight3[node] = g
                        pq3.append((f, g, node))
                        path3[node] = path3[n3] + [n3]
        else:
            pq3.append((f3, g3, n3))
        f1, g1, n1 = pq1.top()[-1]
        f2, g2, n2 = pq2.top()[-1]
        f3, g3, n3 = pq3.top()[-1]
        # if g1 >= min_cost12 and g3 >= min_cost13 and g2 >= min_cost23:
        if g1 >= min_cost12 - heuristic(graph, n1, v2) and \
                g3 >= min_cost13 - heuristic(graph, n3, v1) and \
                g2 >= min_cost23 - heuristic(graph, n2, v3):
            return res()


def return_your_name():
    """Return your name from this function"""
    return 'Xingbo Song'


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
    """


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to bonnie, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None


def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    # Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    # Now we want to execute portions of the formula:
    constOutFront = 2 * 6371  # Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0] - vLatLong[0]) / 2)) ** 2  # First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0]) * math.cos(goalLatLong[0]) * (
            (math.sin((goalLatLong[1] - vLatLong[1]) / 2)) ** 2)  # Second term
    return constOutFront * math.asin(math.sqrt(term1InSqrt + term2InSqrt))  # Straight application of formula
