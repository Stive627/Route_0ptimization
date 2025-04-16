import streamlit as st
import heapq
import matplotlib.pyplot as plt
import networkx as nx
import time

st.set_page_config(page_title="Route Optimizer", layout="centered")

# Define graph
graph = {
    'A': {'B': 5, 'C': 2},
    'B': {'D': 4},
    'C': {'D': 7, 'E': 3},
    'D': {'F': 1},
    'E': {'F': 5},
    'F': {}
}

heuristic = {
    'A': 7,
    'B': 6,
    'C': 4,
    'D': 2,
    'E': 3,
    'F': 0
}

# Algorithms
def bfs(start, goal):
    queue = [(start, [start])]
    visited = set()
    while queue:
        node, path = queue.pop(0)
        if node == goal:
            return path
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited and neighbor not in [p[0] for p in queue]:
                queue.append((neighbor, path + [neighbor]))
    return None

def dfs(start, goal):
    stack = [(start, [start])]
    visited = set()
    while stack:
        node, path = stack.pop()
        if node == goal:
            return path
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None

def astar(start, goal):
    open_set = [(heuristic[start], 0, start, [start])]
    visited = set()
    while open_set:
        _, cost, node, path = heapq.heappop(open_set)
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                total_cost = cost + graph[node][neighbor]
                heapq.heappush(open_set, (total_cost + heuristic[neighbor], total_cost, neighbor, path + [neighbor]))
    return None

def hill_climbing(start, goal):
    current = start
    path = [start]
    while current != goal:
        neighbors = list(graph[current].keys())
        if not neighbors:
            return None
        current = min(neighbors, key=lambda n: heuristic[n])
        if current in path:
            return None
        path.append(current)
    return path

def minimax(node, depth, maximizingPlayer):
    if depth == 0 or node == 'F':
        return heuristic[node], [node]
    paths = []
    for child in graph[node]:
        score, path = minimax(child, depth - 1, not maximizingPlayer)
        paths.append((score, [node] + path))
    return max(paths) if maximizingPlayer else min(paths)

def alpha_beta(node, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or node == 'F':
        return heuristic[node], [node]
    if maximizingPlayer:
        maxEval = float('-inf')
        best_path = []
        for child in graph[node]:
            eval, path = alpha_beta(child, depth-1, alpha, beta, False)
            if eval > maxEval:
                maxEval = eval
                best_path = [node] + path
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval, best_path
    else:
        minEval = float('inf')
        best_path = []
        for child in graph[node]:
            eval, path = alpha_beta(child, depth-1, alpha, beta, True)
            if eval < minEval:
                minEval = eval
                best_path = [node] + path
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval, best_path

def ao_star(node):
    if node == 'F':
        return 0, [node]
    best_cost = float('inf')
    best_path = []
    for child in graph[node]:
        cost, path = ao_star(child)
        total = graph[node][child] + cost
        if total < best_cost:
            best_cost = total
            best_path = [node] + path
    return best_cost, best_path

# Helper
def compute_cost(path):
    return sum(graph[path[i]][path[i+1]] for i in range(len(path)-1)) if path else float('inf')

# Draw graph
def draw_graph(graph, path):
    G = nx.DiGraph()
    for node in graph:
        for neighbor, cost in graph[node].items():
            G.add_edge(node, neighbor, weight=cost)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)})
    if path:
        edge_list = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color='red', width=3)
    st.pyplot(plt)

# UI
st.title("🚦 AI Route Optimization Interface")
st.markdown("Choose a search algorithm and visualize the optimized path on the graph!")

algo = st.selectbox("Choose Algorithm", ["BFS", "DFS", "A*", "Hill Climbing", "Minimax", "Alpha-Beta", "AO*"])
start = st.selectbox("Start Node", list(graph.keys()))
end = st.selectbox("Goal Node", list(graph.keys()))

if st.button("Run Search"):
    with st.spinner("Searching..."):
        t0 = time.time()

        if algo == "BFS":
            path = bfs(start, end)
        elif algo == "DFS":
            path = dfs(start, end)
        elif algo == "A*":
            path = astar(start, end)
        elif algo == "Hill Climbing":
            path = hill_climbing(start, end)
        elif algo == "Minimax":
            path = minimax(start, 4, True)[1]
        elif algo == "Alpha-Beta":
            path = alpha_beta(start, 4, float('-inf'), float('inf'), True)[1]
        elif algo == "AO*":
            path = ao_star(start)[1]

        duration = (time.time() - t0) * 1000
        cost = compute_cost(path)

        st.success("Path found!" if path else "No path found.")
        st.write("**Path:**", path)
        st.write("**Cost:**", cost)
        st.write("**Time:** %.2f ms" % duration)
        draw_graph(graph, path)
