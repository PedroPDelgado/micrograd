def build_topo(v, visited=None):
    if visited is None:
        visited = set()

    if v in visited:
        return []

    visited.add(v)
    children_topo = []
    for c in v._prev:
        children_topo.extend(build_topo(c, visited))
    return children_topo + [v]