import heapq
import networkx as nx
import matplotlib.pyplot as plt
import math

# Node class to store information about each node
class Node:
    def __init__(self, name, heuristic):
        self.name = name
        self.heuristic = heuristic
    
    def __lt__(self, other):
        return self.heuristic < other.heuristic

# Greedy Best-First Search for Hierarchical Routing
def greedy_best_first_search_hierarchical(graph, start, goal, heuristic, region_map):
    # Priority queue to hold nodes to explore, sorted by heuristic value
    priority_queue = []
    heapq.heappush(priority_queue, Node(start, heuristic[start]))

    visited = set()  # To keep track of visited nodes

    # Path dictionary to track the explored paths
    path = {start: None}

    while priority_queue:
        current_node = heapq.heappop(priority_queue).name

        # If the goal is reached, reconstruct the path
        if current_node == goal:
            return reconstruct_path(path, start, goal)

        visited.add(current_node)

        # Explore neighbors in the same region first, then move to other regions
        current_region = region_map[current_node]
        for neighbor in graph[current_node]:
            if neighbor not in visited and region_map[neighbor] == current_region:
                heapq.heappush(priority_queue, Node(neighbor, heuristic[neighbor]))
                if neighbor not in path:
                    path[neighbor] = current_node

        # Explore neighbors in other regions after same-region neighbors
        for neighbor in graph[current_node]:
            if neighbor not in visited and region_map[neighbor] != current_region:
                heapq.heappush(priority_queue, Node(neighbor, heuristic[neighbor]))
                if neighbor not in path:
                    path[neighbor] = current_node

    return None  # If no path is found

# Helper function to reconstruct the path from start to goal
def reconstruct_path(path, start, goal):
    current = goal
    result_path = []
    while current is not None:
        result_path.append(current)
        current = path[current]
    result_path.reverse()
    return result_path

# Function to visualize the graph and the path
def visualize_graph(graph, path, pos, region_map):
    G = nx.Graph()

    # Add edges to the graph
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    # Plot the graph
    plt.figure(figsize=(10, 8))
    node_colors = []
    for node in G.nodes:
        if obstacle_nodes and node in obstacle_nodes:
            node_colors.append("black")  # Obstacles in black
        elif path and node in path:
            node_colors.append("lightgreen")  # Path nodes in green
        else:
            node_colors.append("skyblue")  # Default color
    # Draw the nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=4000, node_color=node_colors, font_size=15,node_shape='s', font_weight='bold', edge_color='gray')

    # Highlight the path
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='black', width=2.5)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='lightgreen',node_shape='s')
    
         

    # Display region information on the graph
    for node, region in region_map.items():
        plt.text(pos[node][0], pos[node][1] - 0.2, f" {region}", fontsize=8, color='grey')

    plt.title("Greedy Best-First Search for Hierarchical Routing", size=20)
    plt.show()
    plt.axis("equal")



# Complex graph with hierarchical regions


graph = {
      'A': ['B', 'K'],
    'B': ['A', 'C', 'L'],
    'C': ['B', 'D', 'M'],
    'D': ['C', 'E', 'N'],
    'E': ['D', 'F', 'O'],
    'F': ['E', 'G', 'P'],
    'G': ['F', 'H', 'Q'],
    'H': ['G', 'I', 'R'],
    'I': ['H', 'J', 'S'],
    'J': ['I', 'T'],
    'K': ['L', 'A', 'U'],
    'L': ['K', 'M', 'B', 'V'],
    'M': ['L', 'N', 'C', 'W'],
    'N': ['M', 'O', 'D', 'X'],
    'O': ['N', 'P', 'E', 'Y'],
    'P': ['O', 'Q', 'F', 'Z'],
    'Q': ['P', 'R', 'G', 'AA'],
    'R': ['Q', 'S', 'H', 'AB'],
    'S': ['R', 'T', 'I', 'AC'],
    'T': ['S', 'J', 'AD'],
    'U': ['V', 'K', 'AE'],
    'V': ['U', 'W', 'L', 'AF'],
    'W': ['V', 'X', 'M', 'AG'],
    'X': ['W', 'Y', 'N', 'AH'],
    'Y': ['X', 'Z', 'O', 'AI'],
    'Z': ['Y', 'AA', 'P', 'AJ'],
    'AA': ['Z', 'AB', 'Q', 'AK'],
    'AB': ['AA', 'AC', 'R', 'AL'],
    'AC': ['AB', 'AD', 'S', 'AM'],
    'AD': ['AC', 'T', 'AN'],
    'AE': ['AF', 'U', 'AO'],
    'AF': ['AE', 'AG', 'V', 'AP'],
    'AG': ['AF', 'AH', 'W', 'AQ'],
    'AH': ['AG', 'AI', 'X', 'AR'],
    'AI': ['AH', 'AJ', 'Y', 'AS'],
    'AJ': ['AI', 'AK', 'Z', 'AT'],
    'AK': ['AJ', 'AL', 'AA', 'AU'],
    'AL': ['AK', 'AM', 'AB', 'AV'],
    'AM': ['AL', 'AN', 'AC', 'AW'],
    'AN': ['AM', 'AD', 'AX'],
    'AO': ['AP', 'AE', 'AY'],
    'AP': ['AO', 'AQ', 'AF', 'AZ'],
    'AQ': ['AP', 'AR', 'AG', 'BA'],
    'AR': ['AQ', 'AS', 'AH', 'BB'],
    'AS': ['AR', 'AT', 'AI', 'BC'],
    'AT': ['AS', 'AU', 'AJ', 'BD'],
    'AU': ['AT', 'AV', 'AK', 'BE'],
    'AV': ['AU', 'AW', 'AL', 'BF'],
    'AW': ['AV', 'AX', 'AM', 'BG'],
    'AX': ['AW', 'AN', 'BH'],
    'AY': ['AZ', 'AO', 'BI'],
    'AZ': ['AY', 'BA', 'AP', 'BJ'],
    'BA': ['AZ', 'BB', 'AQ', 'BK'],
    'BB': ['BA', 'BC', 'AR', 'BL'],
    'BC': ['BB', 'BD', 'AS', 'BM'],
    'BD': ['BC', 'BE', 'AT', 'BN'],
    'BE': ['BD', 'BF', 'AU', 'BO'],
    'BF': ['BE', 'BG', 'AV', 'BP'],
    'BG': ['BF', 'BH', 'AW', 'BQ'],
    'BH': ['BG', 'AX', 'BR'],
    'BI': ['BJ', 'AY', 'BS'],
    'BJ': ['BI', 'BK', 'AZ', 'BT'],
    'BK': ['BJ', 'BL', 'BA', 'BU'],
    'BL': ['BK', 'BM', 'BB', 'BV'],
    'BM': ['BL', 'BN', 'BC', 'BW'],
    'BN': ['BM', 'BO', 'BD', 'BX'],
    'BO': ['BN', 'BP', 'BE', 'BY'],
    'BP': ['BO', 'BQ', 'BF', 'BZ'],
    'BQ': ['BP', 'BR', 'BG', 'CA'],
    'BR': ['BQ', 'BH', 'CB'],
    'BS': ['BT', 'BI', 'CC'],
    'BT': ['BS', 'BU', 'BJ', 'CD'],
    'BU': ['BT', 'BV', 'BK', 'CE'],
    'BV': ['BU', 'BW', 'BL', 'CF'],
    'BW': ['BV', 'BX', 'BM', 'CG'],
    'BX': ['BW', 'BY', 'BN', 'CH'],
    'BY': ['BX', 'BZ', 'BO', 'CI'],
    'BZ': ['BY', 'CA', 'BP', 'CJ'],
    'CA': ['BZ', 'CB', 'BQ', 'CK'],
    'CB': ['CA', 'BR', 'CL'],
    'CC': ['CD', 'BS', 'CM'],
    'CD': ['CC', 'CE', 'BT', 'CN'],
    'CE': ['CD', 'CF', 'BU', 'CO'],
    'CF': ['CE', 'CG', 'BV', 'CP'],
    'CG': ['CF', 'CH', 'BW', 'CQ'],
    'CH': ['CG', 'CI', 'BX', 'CR'],
    'CI': ['CH', 'CJ', 'BY', 'CS'],
    'CJ': ['CI', 'CK', 'BZ', 'CT'],
    'CK': ['CJ', 'CL', 'CA', 'CU'],
    'CL': ['CK', 'CB', 'CV'],
    'CM': ['CN', 'CC'],
    'CN': ['CM', 'CO', 'CD'],
    'CO': ['CN', 'CP', 'CE'],
    'CP': ['CO', 'CQ', 'CF'],
    'CQ': ['CP', 'CR', 'CG'],
    'CR': ['CQ', 'CS', 'CH'],
    'CS': ['CR', 'CT', 'CI'],
    'CT': ['CS', 'CU', 'CJ'],
    'CU': ['CT', 'CV', 'CK'],
    'CV': ['CU', 'CL']
}



# Define regions for the hierarchical routing (nodes belonging to different regions)
region_map = {
    'A': 1, 'B': 1, 'C': 2, 'D': 2, 'E': 3, 'F': 3, 'G': 4, 'H': 4, 'I': 5, 'J': 5,
    'K': 1, 'L': 1, 'M': 2, 'N': 2, 'O': 3, 'P': 3, 'Q': 4, 'R': 4, 'S': 5, 'T': 5,
    'U': 6, 'V': 6, 'W': 7, 'X': 7, 'Y': 8, 'Z': 8, 'AA': 9, 'AB': 9, 'AC': 10, 'AD': 10,
    'AE': 6, 'AF': 6, 'AG': 7, 'AH': 7, 'AI': 8, 'AJ': 8, 'AK': 9, 'AL': 9, 'AM': 10, 'AN': 10,
    'AO': 11, 'AP': 11, 'AQ': 12, 'AR': 12, 'AS': 13, 'AT': 13, 'AU': 14, 'AV': 14, 'AW': 15, 'AX': 15,
    'AY': 11, 'AZ': 11, 'BA': 12, 'BB': 12, 'BC': 13, 'BD': 13, 'BE': 14, 'BF': 14, 'BG': 15, 'BH': 15,
    'BI': 16, 'BJ': 16, 'BK': 17, 'BL': 17, 'BM': 18, 'BN': 18, 'BO': 19, 'BP': 19, 'BQ': 20, 'BR': 20,
    'BS': 16, 'BT': 16, 'BU': 17, 'BV': 17, 'BW': 18, 'BX': 18, 'BY': 19, 'BZ': 19, 'CA': 20, 'CB': 20,
    'CC': 21, 'CD': 21, 'CE': 22, 'CF': 22, 'CG': 23, 'CH': 23, 'CI': 24, 'CJ': 24, 'CK': 25, 'CL': 25,
    'CM': 21, 'CN': 21, 'CO': 22, 'CP': 22, 'CQ': 23, 'CR': 23, 'CS': 24, 'CT': 24, 'CU': 25, 'CV': 25
}

# Define positions for better visualization layout (can be modified)
pos = {
'A': (0, 0),
    'B': (1, 0),
    'C': (2, 0),
    'D': (3, 0),
    'E': (4, 0),
    'F': (5, 0),
    'G': (6, 0),
    'H': (7, 0),
    'I': (8, 0),
    'J': (9, 0),
    'K': (0, 1),
    'L': (1, 1),
    'M': (2, 1),
    'N': (3, 1),
    'O': (4, 1),
    'P': (5, 1),
    'Q': (6, 1),
    'R': (7, 1),
    'S': (8, 1),
    'T': (9, 1),
    'U': (0, 2),
    'V': (1, 2),
    'W': (2, 2),
    'X': (3, 2),
    'Y': (4, 2),
    'Z': (5, 2),
    'AA': (6, 2),
    'AB': (7, 2),
    'AC': (8, 2),
    'AD': (9, 2),
    'AE': (0, 3),
    'AF': (1, 3),
    'AG': (2, 3),
    'AH': (3, 3),
    'AI': (4, 3),
    'AJ': (5, 3),
    'AK': (6, 3),
    'AL': (7, 3),
    'AM': (8, 3),
    'AN': (9, 3),
    'AO': (0, 4),
    'AP': (1, 4),
    'AQ': (2, 4),
    'AR': (3, 4),
    'AS': (4, 4),
    'AT': (5, 4),
    'AU': (6, 4),
    'AV': (7, 4),
    'AW': (8, 4),
    'AX': (9, 4),
    'AY': (0, 5),
    'AZ': (1, 5),
    'BA': (2, 5),
    'BB': (3, 5),
    'BC': (4, 5),
    'BD': (5, 5),
    'BE': (6, 5),
    'BF': (7, 5),
    'BG': (8, 5),
    'BH': (9, 5),
    'BI': (0, 6),
    'BJ': (1, 6),
    'BK': (2, 6),
    'BL': (3, 6),
    'BM': (4, 6),
    'BN': (5, 6),
    'BO': (6, 6),
    'BP': (7, 6),
    'BQ': (8, 6),
    'BR': (9, 6),
    'BS': (0, 7),
    'BT': (1, 7),
    'BU': (2, 7),
    'BV': (3, 7),
    'BW': (4, 7),
    'BX': (5, 7),
    'BY': (6, 7),
    'BZ': (7, 7),
    'CA': (8, 7),
    'CB': (9, 7),
    'CC': (0, 8),
    'CD': (1, 8),
    'CE': (2, 8),
    'CF': (3, 8),
    'CG': (4, 8),
    'CH': (5, 8),
    'CI': (6, 8),
    'CJ': (7, 8),
    'CK': (8, 8),
    'CL': (9, 8),
    'CM': (0, 9),
    'CN': (1, 9),
    'CO': (2, 9),
    'CP': (3, 9),
    'CQ': (4, 9),
    'CR': (5, 9),
    'CS': (6, 9),
    'CT': (7, 9),
    'CU': (8, 9),
    'CV': (9, 9)

}
# Heuristic values (assumed for this example)

obstacle_nodes = {'A', 'B', 'C','D','E','K','L','M','N','O','U',
'V','W','X','Y','AE','AF','AG','AH','AI','AO','AP','AQ','AR',
'AS','AY','AZ','BA','BB','BC','BI','BJ','BK','BL','BM','BQ','AC','AD',
'BR','AM','AN','BE','AW','AX','BF','BG','BH',}

def dynamic_obstacle_recognition(obstacles):
    # Example: Simulate new obstacles (replace with real LiDAR processing logic)
    dynamic_updates = {'CI', 'CZ'}
    obstacle_nodes.update(dynamic_updates)
    return 
move_motors(forward_speed, forward_speed)
move_motors(-forward_speed, -forward_speed)
move_motors(forward_speed, -forward_speed)
move_motors(-forward_speed, forward_speed)
stop_motors()    #DEFINE MOVEMENT
def move_to_node(current_node, next_node, node_positions):
    x1, y1 = node_positions[current_node]
    x2, y2 = node_positions[next_node]

    # Determine the direction of movement
    if x2 > x1:
        print("Moving right")
        move_motors(forward_speed, -forward_speed)
    elif x2 < x1:
        print("Moving left")
        move_motors(-forward_speed, forward_speed)
    elif y2 > y1:
        print("Moving up")
        move_motors(forward_speed, forward_speed)
    elif y2 < y1:
        print("Moving down")
        move_motors(-forward_speed, -forward_speed)
    stop_motors()
def execute_path(path, node_positions):
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        move_to_node(current_node, next_node, node_positions)

start_node = 'I'
goal_node = 'CL'

def get_coordinates(goal_node, pos):
    return pos.get(goal_node, None)
target_point=get_coordinates(goal_node,pos)

def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
heuristic = {
     point: (float('inf') if point in obstacle_nodes else round(euclidean_distance(coord, target_point), 2))
    for point, coord in pos.items()
}
   

obstacles=set(obstacle_nodes)
dynamic_obstacles=dynamic_obstacle_recognition(obstacles)
# Perform Greedy Best-First Search for hierarchical routing

result_path = greedy_best_first_search_hierarchical(graph, start_node, goal_node, heuristic, region_map)
# IMPLEMENT LIDAR DYNAMIC OBSTACLE RECOGNITION
print("Path from {} to {}: {}".format(start_node, goal_node, result_path))

# Visualize the graph and the found path
visualize_graph(graph, result_path, pos, region_map)
