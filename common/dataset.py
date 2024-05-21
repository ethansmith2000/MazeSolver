import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from PIL import Image
import torch

def generate_maze(size):
    # 0 = empty, 1 = walls
    maze = np.ones((size * 2 + 1, size * 2 + 1), dtype=np.int8)

    def remove_wall(pos1, pos2):
        maze[pos1[0] + (pos2[0] - pos1[0]) // 2, pos1[1] + (pos2[1] - pos1[1]) // 2] = 0


    stack = [((1, 1), None)]
    while stack:
        (cx, cy), prev = stack.pop()
        if maze[cy, cx] == 1:
            maze[cy, cx] = 0
            if prev:
                remove_wall(prev, (cx, cy))

            neighbors = [(cx - 2, cy), (cx + 2, cy), (cx, cy - 2), (cx, cy + 2)]
            np.random.shuffle(neighbors)
            for nx, ny in neighbors:
                if 1 <= nx < size * 2 and 1 <= ny < size * 2:
                    stack.append(((nx, ny), (cx, cy)))

    # have to create entry and exit points, random for variation
    walls_for_entry_exit = [(i, 0) for i in range(1, size * 2, 2)] + \
                           [(i, size * 2) for i in range(1, size * 2, 2)] + \
                           [(0, i) for i in range(1, size * 2, 2)] + \
                           [(size * 2, i) for i in range(1, size * 2, 2)]
    entry_exit = np.random.choice(len(walls_for_entry_exit), 2, replace=False)
    entry, exit_pt = walls_for_entry_exit[entry_exit[0]], walls_for_entry_exit[entry_exit[1]]

    maze[entry] = 0
    maze[exit_pt] = 0

    return maze, entry, exit_pt



def solve_maze_dfs(maze):
    # find entry and exit points, it can be in either direcction
    h, w = maze.shape
    temp_maze = maze.copy()
    temp_maze[1:-1, 1:-1] = 1

    locs = np.where(temp_maze == 0)
    entry_idx = np.random.randint(0, len(locs[0]))
    exit_idx = (entry_idx + 1) % len(locs[0]) 

    start = (locs[1][entry_idx], locs[0][entry_idx])
    goal = (locs[1][exit_idx], locs[0][exit_idx])
    path = []
    visited = set()

    def dfs(position):
        if position == goal:
            path.append(position)
            return True
        x, y = position

        visited.add(position)

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):  # Check boundaries
                continue
            if maze[ny, nx] == 0 and (nx, ny) not in visited:
                if dfs((nx, ny)):
                    path.append(position)
                    return True
        return False

    dfs(start)
    return path[::-1]


def interpolate_colormap(value, colors):
    # attempt to recreate the matplotlib colormap interpolation
    index = value * (len(colors) - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(colors) - 1)
    interpolation = index - lower_index
    
    lower_color = np.array(colors[lower_index])
    upper_color = np.array(colors[upper_index])
    color = lower_color + (upper_color - lower_color) * interpolation
    return color.astype(np.uint8)

def apply_complex_heatmap_effect(data, colors= [(102, 0, 181), (0, 255, 0), (255, 255, 0)]):
    h, w = data.shape

    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    heatmap_data = np.zeros((*data.shape, 3), dtype=np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            heatmap_data[i, j] = interpolate_colormap(normalized_data[i, j], colors)
    
    heatmap_image = Image.fromarray(heatmap_data).resize((w*10, h*10), Image.NEAREST)
    return heatmap_image

def visualize_maze(maze, path=None):

    path_maze = np.copy(maze)
    if path is not None:
        if isinstance(path, list):
            try:
                for x, y in path:
                    path_maze[y, x] = 2
            except:
                pass
        else:
            path_maze = path_maze + path * 2

    # cmap = plt.cm.jet
    # cmap.set_under('black')
    # cmap.set_over('gold')

    # plt.figure(figsize=(10, 10))
    # plt.imshow(path_maze, cmap=cmap, vmin=0.5, vmax=2.5)
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    # save in a temp file
    # plt.imsave('maze.png', path_maze)# cmap=cmap, vmin=0.5, vmax=2.5)
    # img = Image.open('maze.png')

    img = apply_complex_heatmap_effect(path_maze)

    return img


def heuristic(a, b):
    """Calculate the Manhattan distance between two points a and b"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def solve_maze_a_star(maze):
    start = (1, 1)
    goal = (maze.shape[0] - 2, maze.shape[1] - 2)

    # Priority queue for nodes to explore stores tuples of (cost, position)
    frontier = PriorityQueue()
    frontier.put((0, start))

    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()[1]

        if current == goal:
            break

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next = (current[0] + dx, current[1] + dy)
            if 0 <= next[0] < maze.shape[1] and 0 <= next[1] < maze.shape[0] and maze[next[1], next[0]] == 0:
                new_cost = cost_so_far[current] + 1  # each step costs 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(next, goal)
                    frontier.put((priority, next))
                    came_from[next] = current

    # Reconstruct path
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)  # optional
    path.reverse()  # optional

    return path


def get_movements_from_path(path):

    mapping = {
        (0, 1): [0,'Down'],
        (0, -1): [1,'Up'],
        (1, 0): [2,'Right'],
        (-1, 0): [3,'Left']
    }

    movements = []
    movement_ids = []
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        movements.append((dx, dy))

    movement_ids = [mapping[m][0] for m in movements]
    movements = [mapping[m][1] for m in movements]

    return movement_ids, movements


def get_path_from_movements(movements, maze):
    #find start pos
    start_y = torch.where(maze == 2)[0].item()
    start_x = torch.where(maze == 2)[1].item()
    start = (start_x, start_y)
    # end_y = torch.where(maze == 3)[0].item()
    # end_x = torch.where(maze == 3)[1].item()
    # end = (end_x, end_y)

    mapping = {
        0: (0, 1),
        1: (0, -1),
        2: (1, 0),
        3: (-1, 0)
    }

    path = [start]
    for m in movements:
        try:
            dx, dy = mapping[m.item()]
        except:
            dx, dy = 0, 0
        path.append((path[-1][0] + dx, path[-1][1] + dy))
    
    return path


def get_maze_path_grid(maze, path):
    path_maze = np.zeros_like(maze)
    for x, y in path:
        path_maze[y, x] = 1  # Mark the path with a distinct value
    return path_maze



def default_collate_fn(examples):
    batch = {}
    for k in examples[0].keys():
        if isinstance(examples[0][k], torch.Tensor):
            batch[k] = torch.stack([example[k] for example in examples])
        else:
            batch[k] = [example[k] for example in examples]

    return batch


class MazeDataset(Dataset):

    def __init__(self, maze_size):
        self.maze_size = maze_size

    def __len__(self):
        return 100_000

    def __getitem__(self, idx):
        maze, entry, exit_pt = generate_maze(self.maze_size)
        path = solve_maze_dfs(maze)
        path_grid = get_maze_path_grid(maze, path)
        maze_labeled = maze.copy()
        maze_labeled[entry] = 2
        maze_labeled[exit_pt] = 3

        maze = torch.from_numpy(maze).float().unsqueeze(0)
        maze_labeled = torch.from_numpy(maze_labeled).float().unsqueeze(0)
        # path = torch.from_numpy(path).float()
        path_grid = torch.from_numpy(path_grid).float().unsqueeze(0)

        maze = maze * 2 - 1
        path_grid = path_grid * 2 - 1

        example = {
            "maze": maze,
            "path_grid": path_grid,
        
            "path": path,
            "maze_labeled": maze_labeled,
            # "movements": get_movements_from_path(path)
        }

        return example









