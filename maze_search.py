import random
import time
import copy
import tracemalloc
from collections import deque
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ============================================================
# MAZE GENERATION
# ============================================================

def create_maze(size=25, obstacle_pct=0.30):
    # fill grid with zeros (empty cells)
    maze = [[0]*size for _ in range(size)]

    # place obstacles randomly (~30%)
    for r in range(size):
        for c in range(size):
            if random.random() < obstacle_pct:
                maze[r][c] = 1

    # pick start and goal positions that are far enough apart
    while True:
        sr, sc = random.randint(0, size-1), random.randint(0, size-1)
        gr, gc = random.randint(0, size-1), random.randint(0, size-1)
        dist = abs(sr-gr) + abs(sc-gc)
        if dist >= 15 and maze[sr][sc] == 0 and maze[gr][gc] == 0:
            break

    # clear a path using BFS so we know one exists
    maze[sr][sc] = 0
    maze[gr][gc] = 0
    clear_path_bfs(maze, (sr, sc), (gr, gc), size)

    return maze, (sr, sc), (gr, gc)


def clear_path_bfs(maze, start, goal, size):
    # use BFS to find a path, then clear obstacles along it
    visited = set()
    visited.add(start)
    queue = deque()
    queue.append((start, [start]))

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            # clear all cells on this path
            for pr, pc in path:
                maze[pr][pc] = 0
            return
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))

    # fallback: carve a straight-ish path
    r, c = start
    gr, gc = goal
    path_cells = []
    while r != gr:
        r += 1 if gr > r else -1
        path_cells.append((r, c))
    while c != gc:
        c += 1 if gc > c else -1
        path_cells.append((r, c))
    for pr, pc in path_cells:
        maze[pr][pc] = 0


# ============================================================
# DYNAMIC OBSTACLES
# ============================================================

def add_obstacle(maze, size, start, goal, current_pos):
    # add a new obstacle at a random empty cell
    empty = []
    for r in range(size):
        for c in range(size):
            if maze[r][c] == 0 and (r,c) != start and (r,c) != goal and (r,c) != current_pos:
                empty.append((r, c))
    if empty:
        r, c = random.choice(empty)
        maze[r][c] = 1
        return (r, c)
    return None


def move_obstacle(maze, size, start, goal, current_pos):
    # pick a random obstacle and move it to an adjacent empty cell
    obstacles = []
    for r in range(size):
        for c in range(size):
            if maze[r][c] == 1:
                obstacles.append((r, c))
    if not obstacles:
        return None, None

    random.shuffle(obstacles)
    for r, c in obstacles:
        neighbors = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < size and 0 <= nc < size and maze[nr][nc] == 0:
                if (nr, nc) != start and (nr, nc) != goal and (nr, nc) != current_pos:
                    neighbors.append((nr, nc))
        if neighbors:
            nr, nc = random.choice(neighbors)
            maze[r][c] = 0
            maze[nr][nc] = 1
            return (r, c), (nr, nc)
    return None, None


def move_start_or_goal(maze, size, start, goal):
    # bonus: move start or goal to adjacent empty cell every 20 moves
    pick = random.choice(['start', 'goal'])
    pos = start if pick == 'start' else goal
    r, c = pos
    neighbors = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < size and 0 <= nc < size and maze[nr][nc] == 0:
            if (nr, nc) != start and (nr, nc) != goal:
                neighbors.append((nr, nc))
    if neighbors:
        new_pos = random.choice(neighbors)
        if pick == 'start':
            return new_pos, goal, pick
        else:
            return start, new_pos, pick
    return start, goal, None


# ============================================================
# HEURISTIC (Manhattan Distance)
# ============================================================

def manhattan(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


# ============================================================
# HELPER: get valid neighbors
# ============================================================

def get_neighbors(maze, pos, size):
    r, c = pos
    result = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < size and 0 <= nc < size and maze[nr][nc] == 0:
            result.append((nr, nc))
    return result


# ============================================================
# DYNAMIC OBSTACLE HANDLER (shared by all algorithms)
# ============================================================

def handle_dynamic_changes(maze, size, move_count, start, goal, current_pos, reroute_log):
    changed = False
    # every 10 moves add an obstacle
    if move_count > 0 and move_count % 10 == 0:
        added = add_obstacle(maze, size, start, goal, current_pos)
        if added:
            changed = True

    # every 15 moves move an obstacle
    if move_count > 0 and move_count % 15 == 0:
        old, new = move_obstacle(maze, size, start, goal, current_pos)
        if old:
            changed = True

    # every 20 moves move start or goal (bonus)
    if move_count > 0 and move_count % 20 == 0:
        start, goal, which = move_start_or_goal(maze, size, start, goal)

    return start, goal, changed


# ============================================================
# CHECK IF A PATH IS STILL VALID
# ============================================================

def path_still_valid(maze, path):
    for r, c in path:
        if maze[r][c] == 1:
            return False
    return True


def quick_bfs(maze, start, goal, size):
    # find shortest path between two points on current maze
    if start == goal:
        return [start]
    visited = set()
    visited.add(start)
    queue = deque([(start, [start])])
    while queue:
        cur, p = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            if 0 <= nr < size and 0 <= nc < size and (nr,nc) not in visited and maze[nr][nc] == 0:
                visited.add((nr, nc))
                new_p = p + [(nr, nc)]
                if (nr, nc) == goal:
                    return new_p
                queue.append(((nr, nc), new_p))
    return None


def fix_path(maze, path, start, goal, size):
    # ensure returned path starts at S and ends at G with no gaps
    if path is None:
        return None
    # keep only cells that are free in the final maze
    valid = [c for c in path if maze[c[0]][c[1]] == 0]
    if not valid:
        return quick_bfs(maze, start, goal, size)

    # rebuild a fully connected path through valid waypoints
    result = []
    # connect start to first valid cell
    if valid[0] != start:
        link = quick_bfs(maze, start, valid[0], size)
        if link:
            result = link
        else:
            result = [start]
    else:
        result = [valid[0]]

    # walk through valid cells, bridging any gaps
    for i in range(1, len(valid)):
        prev = result[-1]
        curr = valid[i]
        if curr == prev:
            continue
        if abs(prev[0] - curr[0]) + abs(prev[1] - curr[1]) == 1:
            result.append(curr)
        else:
            link = quick_bfs(maze, prev, curr, size)
            if link:
                result.extend(link[1:])
            # else skip this waypoint — unreachable from current position

    # connect last cell to goal
    if result[-1] != goal:
        link = quick_bfs(maze, result[-1], goal, size)
        if link:
            result.extend(link[1:])
        else:
            # can't reach goal from end of path, try full S→G
            full = quick_bfs(maze, start, goal, size)
            if full:
                return full
            result.append(goal)

    # final check: if path still has gaps, replace with clean S->G BFS
    has_gaps = any(abs(result[i][0]-result[i+1][0]) + abs(result[i][1]-result[i+1][1]) != 1
                   for i in range(len(result)-1))
    if has_gaps:
        full = quick_bfs(maze, start, goal, size)
        if full:
            return full
        return None  # S and G disconnected on final maze

    return result


# ============================================================
# SEARCH ALGORITHMS
# ============================================================

# --- BFS ---
def bfs_search(maze, start, goal, size):
    maze = copy.deepcopy(maze)
    visited = set()
    visited.add(start)
    queue = deque()
    queue.append((start, [start]))
    nodes_expanded = 0
    move_count = 0
    reroute_count = 0
    reroute_log = []

    while queue:
        current, path = queue.popleft()
        # skip if cell got blocked by dynamic obstacle
        if maze[current[0]][current[1]] == 1:
            continue
        nodes_expanded += 1
        move_count += 1

        # check goal before dynamic changes
        if current == goal:
            return path, nodes_expanded, move_count, reroute_count, reroute_log, maze

        start_new, goal_new, changed = handle_dynamic_changes(
            maze, size, move_count, start, goal, current, reroute_log)
        if start_new != start or goal_new != goal:
            start, goal = start_new, goal_new
        if changed and not path_still_valid(maze, path):
            reroute_count += 1
            reroute_log.append(current)
            visited = set()
            visited.add(current)
            queue = deque()
            queue.append((current, [current]))
            continue

        for nb in get_neighbors(maze, current, size):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))

    return None, nodes_expanded, move_count, reroute_count, reroute_log, maze


# --- DFS ---
def dfs_search(maze, start, goal, size):
    maze = copy.deepcopy(maze)
    visited = set()
    stack = [(start, [start])]
    nodes_expanded = 0
    move_count = 0
    reroute_count = 0
    reroute_log = []

    while stack:
        current, path = stack.pop()
        if current in visited:
            continue
        # skip if cell got blocked
        if maze[current[0]][current[1]] == 1:
            continue
        visited.add(current)
        nodes_expanded += 1
        move_count += 1

        if current == goal:
            return path, nodes_expanded, move_count, reroute_count, reroute_log, maze

        start_new, goal_new, changed = handle_dynamic_changes(
            maze, size, move_count, start, goal, current, reroute_log)
        if start_new != start or goal_new != goal:
            start, goal = start_new, goal_new
        if changed and not path_still_valid(maze, path):
            reroute_count += 1
            reroute_log.append(current)
            visited = set()
            visited.add(current)
            stack = [(current, [current])]
            continue

        for nb in sorted(get_neighbors(maze, current, size), key=lambda x: manhattan(x, goal), reverse=True):
            if nb not in visited:
                stack.append((nb, path + [nb]))

    return None, nodes_expanded, move_count, reroute_count, reroute_log, maze


# --- UCS ---
def ucs_search(maze, start, goal, size):
    maze = copy.deepcopy(maze)
    visited = set()
    # priority queue: (cost, counter, position, path)
    counter = 0
    pq = [(0, counter, start, [start])]
    nodes_expanded = 0
    move_count = 0
    reroute_count = 0
    reroute_log = []

    while pq:
        cost, _, current, path = heapq.heappop(pq)
        if current in visited:
            continue
        # skip blocked cells still in priority queue
        if maze[current[0]][current[1]] == 1:
            continue
        visited.add(current)
        nodes_expanded += 1
        move_count += 1

        if current == goal:
            return path, nodes_expanded, move_count, reroute_count, reroute_log, maze

        start_new, goal_new, changed = handle_dynamic_changes(
            maze, size, move_count, start, goal, current, reroute_log)
        if start_new != start or goal_new != goal:
            start, goal = start_new, goal_new
        if changed and not path_still_valid(maze, path):
            reroute_count += 1
            reroute_log.append(current)
            # keep valid frontier entries, just clear visited
            visited = set()
            visited.add(current)
            counter += 1
            new_pq = [(0, counter, current, [current])]
            for item in pq:
                r, c = item[2]
                if maze[r][c] == 0 and item[2] not in visited:
                    new_pq.append(item)
            pq = new_pq
            heapq.heapify(pq)
            continue

        for nb in get_neighbors(maze, current, size):
            if nb not in visited:
                counter += 1
                heapq.heappush(pq, (cost + 1, counter, nb, path + [nb]))

    return None, nodes_expanded, move_count, reroute_count, reroute_log, maze


# --- DLS (Depth Limited Search) ---
def dls_search(maze, start, goal, size, depth_limit=50):
    maze = copy.deepcopy(maze)
    nodes_expanded = 0
    move_count = 0
    reroute_count = 0
    reroute_log = []

    def dls_recursive(current, path, depth, visited):
        nonlocal nodes_expanded, move_count, reroute_count, start, goal, maze

        # skip if cell got blocked dynamically
        if maze[current[0]][current[1]] == 1:
            return None
        nodes_expanded += 1
        move_count += 1

        start_new, goal_new, changed = handle_dynamic_changes(
            maze, size, move_count, start, goal, current, reroute_log)
        if start_new != start or goal_new != goal:
            start, goal = start_new, goal_new
        if changed and not path_still_valid(maze, path):
            reroute_count += 1
            reroute_log.append(current)

        if current == goal:
            return path

        if depth >= depth_limit:
            return None  # hit depth limit, backtrack

        for nb in sorted(get_neighbors(maze, current, size), key=lambda x: manhattan(x, goal)):
            if nb not in visited:
                visited.add(nb)
                result = dls_recursive(nb, path + [nb], depth + 1, visited)
                if result is not None:
                    return result
                visited.discard(nb)  # backtrack

        return None

    visited = set()
    visited.add(start)
    result = dls_recursive(start, [start], 0, visited)
    return result, nodes_expanded, move_count, reroute_count, reroute_log, maze


# --- IDS (Iterative Deepening Search) ---
def ids_search(maze, start, goal, size, max_depth=100):
    original_maze = copy.deepcopy(maze)
    total_nodes = 0
    total_moves = 0
    total_reroutes = 0
    all_reroute_log = []

    for limit in range(0, max_depth + 1):
        # reset maze for each depth iteration
        maze_copy = copy.deepcopy(original_maze)
        path, nodes, moves, reroutes, rlog, final_maze = dls_search(
            maze_copy, start, goal, size, depth_limit=limit)
        total_nodes += nodes
        total_moves += moves
        total_reroutes += reroutes
        all_reroute_log.extend(rlog)
        if path is not None:
            return path, total_nodes, total_moves, total_reroutes, all_reroute_log, final_maze

    return None, total_nodes, total_moves, total_reroutes, all_reroute_log, maze


# --- Greedy Best-First Search ---
def greedy_search(maze, start, goal, size):
    maze = copy.deepcopy(maze)
    visited = set()
    counter = 0
    pq = [(manhattan(start, goal), counter, start, [start])]
    nodes_expanded = 0
    move_count = 0
    reroute_count = 0
    reroute_log = []

    while pq:
        _, _, current, path = heapq.heappop(pq)
        if current in visited:
            continue
        if maze[current[0]][current[1]] == 1:
            continue
        visited.add(current)
        nodes_expanded += 1
        move_count += 1

        if current == goal:
            return path, nodes_expanded, move_count, reroute_count, reroute_log, maze

        start_new, goal_new, changed = handle_dynamic_changes(
            maze, size, move_count, start, goal, current, reroute_log)
        if start_new != start or goal_new != goal:
            start, goal = start_new, goal_new
        if changed and not path_still_valid(maze, path):
            reroute_count += 1
            reroute_log.append(current)
            visited = set()
            visited.add(current)
            counter += 1
            pq = [(manhattan(current, goal), counter, current, [current])]
            continue

        for nb in get_neighbors(maze, current, size):
            if nb not in visited:
                counter += 1
                heapq.heappush(pq, (manhattan(nb, goal), counter, nb, path + [nb]))

    return None, nodes_expanded, move_count, reroute_count, reroute_log, maze


# --- A* Search ---
def astar_search(maze, start, goal, size):
    maze = copy.deepcopy(maze)
    visited = set()
    counter = 0
    g_cost = {start: 0}
    pq = [(manhattan(start, goal), counter, start, [start])]
    nodes_expanded = 0
    move_count = 0
    reroute_count = 0
    reroute_log = []

    while pq:
        f, _, current, path = heapq.heappop(pq)
        if current in visited:
            continue
        if maze[current[0]][current[1]] == 1:
            continue
        visited.add(current)
        nodes_expanded += 1
        move_count += 1

        if current == goal:
            return path, nodes_expanded, move_count, reroute_count, reroute_log, maze

        start_new, goal_new, changed = handle_dynamic_changes(
            maze, size, move_count, start, goal, current, reroute_log)
        if start_new != start or goal_new != goal:
            start, goal = start_new, goal_new
        if changed and not path_still_valid(maze, path):
            reroute_count += 1
            reroute_log.append(current)
            visited = set()
            visited.add(current)
            g_cost = {current: 0}
            counter += 1
            pq = [(manhattan(current, goal), counter, current, [current])]
            continue

        current_g = g_cost[current]
        for nb in get_neighbors(maze, current, size):
            new_g = current_g + 1
            if nb not in visited and (nb not in g_cost or new_g < g_cost[nb]):
                g_cost[nb] = new_g
                f_val = new_g + manhattan(nb, goal)
                counter += 1
                heapq.heappush(pq, (f_val, counter, nb, path + [nb]))

    return None, nodes_expanded, move_count, reroute_count, reroute_log, maze


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_maze(maze, size, path, start, goal, title="Maze"):
    # create a color grid
    grid = np.zeros((size, size))
    for r in range(size):
        for c in range(size):
            if maze[r][c] == 1:
                grid[r][c] = 1  # obstacle

    # validate and mark path (skip any cell that is an obstacle)
    if path:
        for r, c in path:
            if maze[r][c] == 1:
                print(f"WARNING: path cell ({r},{c}) is an obstacle, skipping")
                continue
            grid[r][c] = 2  # path cell

    # mark start and goal on top
    grid[start[0]][start[1]] = 3
    grid[goal[0]][goal[1]] = 4

    # custom colormap: white=empty, black=obstacle, blue=path, green=start, red=goal
    cmap = mcolors.ListedColormap(['white', 'black', '#4A90D9', '#2ECC71', '#E74C3C'])
    bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    # use extent so each cell maps exactly to one grid square
    ax.imshow(grid, cmap=cmap, norm=norm, origin='upper',
              extent=[0, size, size, 0], interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=14)

    # gridlines at cell boundaries (0, 1, 2, ... size)
    ax.set_xticks(range(size + 1))
    ax.set_yticks(range(size + 1))
    ax.set_xticklabels([str(i) if i < size else '' for i in range(size + 1)], fontsize=6)
    ax.set_yticklabels([str(i) if i < size else '' for i in range(size + 1)], fontsize=6)
    ax.grid(True, linewidth=0.5, color='gray')

    # label S and G at cell centers (col + 0.5, row + 0.5)
    ax.text(start[1] + 0.5, start[0] + 0.5, 'S', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    ax.text(goal[1] + 0.5, goal[0] + 0.5, 'G', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(title.replace(" ", "_").replace("*", "star") + ".png", dpi=100)
    plt.show()


# ============================================================
# RUN A SINGLE ALGORITHM AND COLLECT METRICS
# ============================================================

def run_algorithm(algo_func, maze, start, goal, size, name, **kwargs):
    maze_copy = copy.deepcopy(maze)

    # track memory
    tracemalloc.start()
    start_time = time.time()

    result = algo_func(maze_copy, start, goal, size, **kwargs)

    elapsed = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    path, nodes_expanded, moves, reroutes, reroute_log, final_maze = result
    # ensure S and G are free in the final maze (dynamic obstacles may have landed on them)
    final_maze[start[0]][start[1]] = 0
    final_maze[goal[0]][goal[1]] = 0
    # fix path so it always starts at S and ends at G
    if path is not None:
        path = fix_path(final_maze, path, start, goal, size)
    success = path is not None
    path_len = len(path) if path else 0

    # print results
    print(f"\n{'='*50}")
    print(f"Algorithm: {name}")
    print(f"{'='*50}")
    print(f"Success: {success}")
    print(f"Path Length: {path_len}")
    print(f"Nodes Expanded: {nodes_expanded}")
    print(f"Moves: {moves}")
    print(f"Reroutes: {reroutes}")
    print(f"Time: {elapsed:.4f} sec")
    print(f"Peak Memory: {peak_mem / 1024:.2f} KB")
    if path:
        print(f"Path: {path}")
    if reroute_log:
        print(f"Reroute points: {reroute_log}")

    # visualize the path on the original maze
    visualize_maze(final_maze, size, path, start, goal, title=f"{name} - Path")

    return {
        'name': name,
        'success': success,
        'path_length': path_len,
        'nodes_expanded': nodes_expanded,
        'moves': moves,
        'reroutes': reroutes,
        'time': elapsed,
        'peak_memory_kb': peak_mem / 1024
    }


# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

def compare_results(results):
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    header = f"{'Algorithm':<20} {'Success':<10} {'Path Len':<10} {'Nodes':<10} {'Moves':<10} {'Reroutes':<10} {'Time(s)':<10} {'Mem(KB)':<10}"
    print(header)
    print("-"*80)
    for r in results:
        row = f"{r['name']:<20} {str(r['success']):<10} {r['path_length']:<10} {r['nodes_expanded']:<10} {r['moves']:<10} {r['reroutes']:<10} {r['time']:<10.4f} {r['peak_memory_kb']:<10.2f}"
        print(row)

    # bar charts
    names = [r['name'] for r in results]
    x = range(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # nodes expanded
    axes[0][0].bar(x, [r['nodes_expanded'] for r in results], color='#3498DB')
    axes[0][0].set_title("Nodes Expanded")
    axes[0][0].set_xticks(x)
    axes[0][0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)

    # time taken
    axes[0][1].bar(x, [r['time'] for r in results], color='#E67E22')
    axes[0][1].set_title("Time (seconds)")
    axes[0][1].set_xticks(x)
    axes[0][1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)

    # memory
    axes[1][0].bar(x, [r['peak_memory_kb'] for r in results], color='#2ECC71')
    axes[1][0].set_title("Peak Memory (KB)")
    axes[1][0].set_xticks(x)
    axes[1][0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)

    # path length
    axes[1][1].bar(x, [r['path_length'] for r in results], color='#9B59B6')
    axes[1][1].set_title("Path Length")
    axes[1][1].set_xticks(x)
    axes[1][1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)

    plt.suptitle("Algorithm Performance Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig("comparison_chart.png", dpi=100)
    plt.show()


# ============================================================
# SUCCESS RATE EVALUATION (multiple runs)
# ============================================================

def evaluate_success_rate(num_runs=5, size=25):
    print("\n" + "="*80)
    print(f"SUCCESS RATE EVALUATION ({num_runs} runs)")
    print("="*80)

    algo_names = ["BFS", "DFS", "UCS", "DLS", "IDS", "Greedy", "A*"]
    algo_funcs = [bfs_search, dfs_search, ucs_search, dls_search, ids_search, greedy_search, astar_search]
    algo_kwargs = [{}, {}, {}, {'depth_limit': 80}, {'max_depth': 80}, {}, {}]

    success_counts = {name: 0 for name in algo_names}
    total_times = {name: 0.0 for name in algo_names}
    total_nodes = {name: 0 for name in algo_names}

    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        maze, start, goal = create_maze(size=size)

        for i, (func, name) in enumerate(zip(algo_funcs, algo_names)):
            maze_copy = copy.deepcopy(maze)
            t0 = time.time()
            result = func(maze_copy, start, goal, size, **algo_kwargs[i])
            elapsed = time.time() - t0

            path = result[0]
            nodes = result[1]
            if path:
                success_counts[name] += 1
            total_times[name] += elapsed
            total_nodes[name] += nodes

    print("\n" + "="*80)
    print("SUCCESS RATE SUMMARY")
    print("="*80)
    header = f"{'Algorithm':<15} {'Successes':<12} {'Rate':<12} {'Avg Time(s)':<15} {'Avg Nodes':<12}"
    print(header)
    print("-"*60)
    for name in algo_names:
        rate = success_counts[name] / num_runs * 100
        avg_time = total_times[name] / num_runs
        avg_nodes = total_nodes[name] / num_runs
        rate_str = f"{rate:.1f}%"
        print(f"{name:<15} {success_counts[name]:<12} {rate_str:<12} {avg_time:<15.4f} {avg_nodes:<12.0f}")

    # chart for success rate
    fig, ax = plt.subplots(figsize=(10, 5))
    rates = [success_counts[n] / num_runs * 100 for n in algo_names]
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']
    ax.bar(algo_names, rates, color=colors)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(f"Algorithm Success Rate over {num_runs} Runs")
    ax.set_ylim(0, 110)
    for i, v in enumerate(rates):
        ax.text(i, v + 2, f"{v:.0f}%", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("success_rate_chart.png", dpi=100)
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    # no fixed seed so S/G are randomized each run
    size = 25
    maze, start, goal = create_maze(size=size)

    print(f"Maze Size: {size}x{size}")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Manhattan distance: {manhattan(start, goal)}")

    # show initial maze
    visualize_maze(copy.deepcopy(maze), size, None, start, goal, title="Initial Maze")

    # run all algorithms on the same maze
    results = []

    results.append(run_algorithm(bfs_search, maze, start, goal, size, "BFS"))
    results.append(run_algorithm(dfs_search, maze, start, goal, size, "DFS"))
    results.append(run_algorithm(ucs_search, maze, start, goal, size, "UCS"))
    results.append(run_algorithm(dls_search, maze, start, goal, size, "DLS", depth_limit=80))
    results.append(run_algorithm(ids_search, maze, start, goal, size, "IDS", max_depth=80))
    results.append(run_algorithm(greedy_search, maze, start, goal, size, "Greedy"))
    results.append(run_algorithm(astar_search, maze, start, goal, size, "A*"))

    # compare performance
    compare_results(results)

    # multi-run success rate evaluation
    evaluate_success_rate(num_runs=5, size=size)


if __name__ == "__main__":
    main()
