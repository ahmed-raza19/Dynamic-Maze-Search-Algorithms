# Dynamic Maze Search Algorithms

## Overview
This project is an Artificial Intelligence (AI) pathfinding demonstration. It implements and compares seven different search algorithms to find the shortest path in a maze from a starting point to a goal. 

What makes this project unique is the **dynamic environment**: obstacles can randomly appear or move while the algorithms are searching for a path, forcing the algorithms to recalculate and adapt in real-time.

## Algorithms Implemented
The following search algorithms are included and benchmarked against each other:
1. **Breadth-First Search (BFS)**
2. **Depth-First Search (DFS)**
3. **Uniform Cost Search (UCS)**
4. **Depth-Limited Search (DLS)**
5. **Iterative Deepening Search (IDS)**
6. **Greedy Best-First Search**
7. **A* (A-Star) Search**

## Features
- **Dynamic Obstacles:** Obstacles move and appear dynamically, requiring algorithms to reroute.
- **Performance Benchmarking:** Compares algorithms based on time taken, memory used, nodes expanded, and path length.
- **Visualizations:** Automatically generates visual charts and colored grid images of the maze and the discovered paths.
- **Success Rate Evaluation:** Runs multiple tests to calculate the overall success rate of each algorithm.

## How to Run

To run this project, you need Python installed on your computer. Open your terminal or command prompt and run the following commands sequentially:

```bash
# 1. Clone this repository to your local machine
git clone https://github.com/your-username/your-repo-name.git

# 2. Navigate into the project folder
cd your-repo-name

# 3. Install the required external libraries
pip install -r requirements.txt

# 4. Run the main python script
python maze_search.py
