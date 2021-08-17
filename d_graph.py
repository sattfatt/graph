# Course: CS261 - Data Structures
# Author: Satyam patel
# Assignment: 6
# Description: Implements a directed graph

from collections import deque
import heapq

class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #
    def _is_valid_index(self, i:int) -> bool:
            """
            returns true if i is a valid index for the adj matrix
            """
            if 0 <= i < len(self.adj_matrix):
                return True
            return False

    def _is_valid_edge(self, row, col) -> bool:
        """
        Returns true if row, col is a valid entry in adj matrix
        """
        if self._is_valid_index(row) and self._is_valid_index(col) and row != col:
            return True
        return False

    def add_vertex(self) -> int:
        """
        Adds vertex to directed graph
        """
        size = len(self.adj_matrix)
        # append a zero for each row
        for row in self.adj_matrix:
            row.append(0)
        # append a row of zeros
        self.adj_matrix.append([0 for _ in range(size + 1)])
        self.v_count += 1
        return self.v_count
        
    
    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        Adds an edge to the directed graph
        """
        if self._is_valid_edge(src, dst) and weight > 0:
            self.adj_matrix[src][dst] = weight

    def remove_edge(self, src: int, dst: int) -> None:
        """
        Removes edge from graph (This means vertices stay where they are.)
        """
        if self._is_valid_edge(src, dst):
            self.adj_matrix[src][dst] = 0

    def get_vertices(self) -> list:
        """
        Returns all the vertices in the graph
        """
        return [i for i in range(len(self.adj_matrix))]

    def get_edges(self) -> list:
        """
        Returns all the edges in the graph as a list
        """
        return [(src, dst, weight) for src, row in enumerate(self.adj_matrix) for dst, weight in enumerate(row) if weight > 0]

    def is_valid_path(self, path: list) -> bool:
        """
        Checks if path is valid in the directe graph
        """
        return all([self.adj_matrix[src][dst] > 0 for src, dst in zip(path[0:len(path)-1], path[1:len(path)])])

    def _get_neighbors(self, vertex:int, weights=False) -> list:
        """
        Returns list of all the neighbors of vertex in the adj matrix
        """
        if weights:
            return [(dst, weight) for dst, weight in enumerate(self.adj_matrix[vertex]) if weight > 0]

        return [dst for dst, weight in enumerate(self.adj_matrix[vertex]) if weight > 0]

    def dfs(self, v_start, v_end=None) -> list:
        """
        Navigates graph using depth first search.
        """
        visited = []
        stack = deque()
        stack.append(v_start)

        while len(stack) > 0:
            vertex = stack.pop()
            if vertex == v_end:
                visited.append(vertex)
                return visited
            elif vertex not in visited:
                # get all the neighbors of 
                neighbors = self._get_neighbors(vertex)
                # push these to the stack
                for v in reversed(neighbors):
                    stack.append(v)
                # add the vertex to visited
                visited.append(vertex)
        return visited        

    def bfs(self, v_start, v_end=None) -> list:
        """
        Navigates graph using breadth first search
        """
        visited = []
        queue = deque()
        queue.appendleft(v_start)

        while len(queue) > 0:
            vertex = queue.pop()
            if vertex == v_end:
                visited.append(vertex)
                return visited
            elif vertex not in visited:
                # add to visited
                visited.append(vertex)
            neighbors = self._get_neighbors(vertex)
            for v in neighbors:
                # get all the non visited neighbors and add them to the queue
                if v not in visited:
                    queue.appendleft(v)
        return visited

    def has_cycle_in_component(self, start):
        """
        Checks for cycles in component using dfs and keeping track of entering and exiting each node.
        """
        # state of the nodes (for the state dict)
        UNVISITED = 0
        ENTERED = 1
        EXITED = 2

        # maneuver taken on the node (for the stack)
        ENTERING = 0
        EXITING = 1

        state = {src: UNVISITED for src, row in enumerate(self.adj_matrix)}
        stack = deque()
        stack.append((start, ENTERING))

        while len(stack) > 0:
            vertex, maneuver = stack.pop()
            if maneuver == EXITING:
                # mark this node as successfully processed. This node is in no loops.
                state[vertex] = EXITED
            else:
                state[vertex] = ENTERED
                # we are going to exit this vertex so push that to the stack
                stack.append((vertex, EXITING))
                for neighbor in self._get_neighbors(vertex):
                    # if we encounter a node that we entered but never exited, we have a loop.
                    if state[neighbor] == ENTERED:
                        visited = [v for v, state in state.items() if state!=UNVISITED]
                        return True, visited
                    elif state[neighbor] == UNVISITED:
                        # if we have an unvisited node push that we are entering it  
                        stack.append((neighbor, ENTERING))
        visited = [v for v, state in state.items() if state!=UNVISITED]           
        return False, visited 

    def has_cycle(self):
        """
        Checks all disjoint components for cycles
        """
        visited = set()
        start = 0
        while len(visited) != self.v_count:
            if start in visited:
                start += 1
                continue

            cyclic, verts = self.has_cycle_in_component(start)

            if cyclic:
                return True

            visited = visited.union(set(verts))

        return False


    def dijkstra(self, src: int) -> list:
        """
        Implements djikstra's algorithm to find the smallest pathlengths to each node from the src node.
        """

        visited = {src:float("inf") for src, row in enumerate(self.adj_matrix)}
        prioq = []
        heapq.heappush(prioq, (0, src))

        while len(prioq) > 0:
            dist, v = heapq.heappop(prioq)
            if visited[v] == float("inf"):
                visited[v] = dist
                for dst, weight in self._get_neighbors(v,True):
                    d_i = weight
                    heapq.heappush(prioq, (dist + d_i, dst))

        return [visited[v] for v in visited]


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = DirectedGraph()
    print(g)
    for _ in range(5):
        g.add_vertex()
    print(g)

    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    for src, dst, weight in edges:
        g.add_edge(src, dst, weight)
    print(g)


    print("\nPDF - method get_edges() example 1")
    print("----------------------------------")
    g = DirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, g.is_valid_path(path))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        g.remove_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    for src, dst in edges_to_add:
        g.add_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')
    print('\n', g)


    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    g.remove_edge(4, 3)
    print('\n', g)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
