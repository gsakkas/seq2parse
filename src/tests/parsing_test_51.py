def reachable_helper(graph, node, found):
    children = set(graph[node])
    print('children', children)
    new_nodes = children - found
    found |= new_nodes
    for child_node in list(new_nodes):
        new_nodes |= reachable_helper(graph, child_node, found)
    print('new nodes', new_nodes)
    return new_nodes

def reachable(graph, node):
    return list(reachable_helper(graph, node, set()))


my_graph = { 'a': ['b', 'c'], 'b': ['d'], 'c': [], 'd': ['a'], 'e': ['d']  }
reachable(my_graph, 'e')
reachable({}, 'a not in graph')
