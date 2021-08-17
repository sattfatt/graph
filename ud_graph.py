# Course: CS261
# Author: Satyam Patel patelsat
# Assignment: 6
# Description: Implements an Undirected Graph data structure.

from collections import deque


class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        Add new vertex to the graph
        """
        if v not in self.adj_list:
            self.adj_list[v] = []
        
    def add_edge(self, u: str, v: str) -> None:
        """
        Add edge to the graph
        """
        if u == v:
            return

        self.add_vertex(u)
        self.add_vertex(v)

        if u not in self.adj_list[v]:
            self.adj_list[v].append(u)
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)

    def remove_edge(self, v: str, u: str) -> None:
        """
        Remove edge from the graph
        """
        if u not in self.adj_list or v not in self.adj_list:
            return
        if u not in self.adj_list[v] or v not in self.adj_list[u]:
            return
        
        self.adj_list[v].remove(u)
        self.adj_list[u].remove(v)
        

    def remove_vertex(self, v: str) -> None:
        """
        Remove vertex and all connected edges
        """
        if v not in self.adj_list:
            return
        # remove v from all the vertices connected to it.
        for vertex in self.adj_list[v]:
            try:
                self.adj_list[vertex].remove(v)
            except:
                continue
        # remove v from the dict
        self.adj_list.pop(v)
        
        

    def get_vertices(self) -> list:
        """
        Return list of vertices in the graph (any order)
        """
        return list(self.adj_list.keys())


    def get_edges(self) -> list:
        """
        Return list of edges in the graph (any order)
        """
        edges = set()

        for key, connected in self.adj_list.items():
            for vertex in connected:
                edge = None
                if vertex < key:
                    edge = (vertex, key)
                else:
                    edge = (key, vertex)
                edges.add(edge)

        return list(edges)
        

    def is_valid_path(self, path: []) -> bool:
        """
        Return true if provided path is valid, False otherwise
        """
        if any([v not in self.adj_list for v in path]):
            return False

        if len(path) < 2:
            return True

        for i in range(1, len(path)):
            if path[i-1] not in self.adj_list[path[i]]:
                return False

        return True
       
    def dfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during DFS search
        Vertices are picked in alphabetical order
        """
        return self.dfs_extended(v_start, v_end)[0]

    def dfs_extended(self, v_start, v_end=None, end_if_cycle=False) -> []:
        """
        Return list of vertices visited during DFS search
        Vertices are picked in alphabetical order
        Also checks for cycles and can end prematurely if cycle is found.
        """
        #pre checks
        if v_start not in self.adj_list:
            return [], False
        
        visited = []
        stack = deque()
        stack.append(v_start)
        cyclic = False
        while len(stack) > 0:
            vert = stack.pop()
            # if we found v_end then break
            if vert == v_end:
                visited.append(vert)
                break

            elif vert not in visited:
                sorted_neighbors = sorted(self.adj_list[vert], reverse=True)
                for neighbor in sorted_neighbors:                
                    stack.append(neighbor)
                visited.append(vert)

            if len(stack) > 0:
                if vert in stack:
                    cyclic = True

        return list(visited), cyclic
       
    def bfs(self, v_start, v_end=None, return_set=False) -> list:
        """
        Return list of vertices visited during BFS search
        """
        # same as dfs except we use a queue instead of stack.
        if v_start not in self.adj_list:
            if return_set:
                return set()
            return []

        visited = []
        queue = deque()
        queue.appendleft(v_start)

        while len(queue) > 0:
            vert = queue.pop()
            if vert == v_end:
                visited.append(vert)
                break
            if vert not in visited:
                visited.append(vert)  
            sorted_neighbors = sorted(self.adj_list[vert])
            for v in sorted_neighbors:
                if v not in visited:         
                    queue.appendleft(v)
        if return_set:
            return set(visited)
        return visited
        
    def count_connected_components(self):
        """
        Return number of connected componets in the graph
        """
        components = 0
        visited = set()
        vertices = iter(self.adj_list.keys())

        while len(visited) != len(self.adj_list):
            # choose the next vertex
            vertex = next(vertices)
            if vertex in visited:
                # if we already visited it go back to the beginning of the while loop
                continue
            # run bfs on it until it ends
            verts = self.bfs(vertex, return_set=True)
            # union visited
            visited = visited.union(verts)
            # increase the count since this is a component
            components += 1
        return components

    def has_cycle(self):
        """
        Return True if graph contains a cycle, False otherwise
        """
        # we need to check cyclic for each component

        visited = set()
        vertices = iter(self.adj_list.keys())

        while len(visited) != len(self.adj_list):
            # choose the next vertex
            vertex = next(vertices)
            if vertex in visited:
                # if we already visited it go back to the beginning of the while loop
                continue
            # run bfs on it until it ends
            verts, cyclic = self.dfs_extended(vertex)
            if cyclic:
                return True
            # union visited
            visited = visited.union(set(verts))

        return False


    

if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)


    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    g.remove_vertex('D')
    print(g)


    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')


    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())
