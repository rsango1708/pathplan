import heapq
import networkx as nx
import matplotlib.pyplot as plt
import math
import string
import itertools
import time
from time import sleep
import ld06
from unittest import mock
import RPi.GPIO as gpio
gpio.setmode(gpio.BCM)




PORT_NAME = '/dev/ttyS0'  # Adjust this based on your Raspberry Pi's connection
lidar = ld06('/dev/ttyS0')

# Set a threshold for detecting obstacles
OBSTACLE_THRESHOLD = 255  # Threshold distance in mm (e.g., 500 mm = 0.5 meters)




def process_lidar_data():
    # Start scanning with the LiDAR
    lidar.start_motor()
    print("Starting scan...")

    try:
        for scan in lidar.iter_scans():
            # scan is a list of tuples (angle, distance) where distance is in mm
            obstacle_detected = False
            dynamic_obstacle_nodes = set()

            for (angle, distance) in scan:
                if distance < OBSTACLE_THRESHOLD:
                    print(f"Obstacle detected at angle {angle}° and distance {distance}mm")
                    dynamic_obstacle_nodes.add((angle, distance))  # Save the detected obstacle info
                    obstacle_detected = True

            # Stop scanning after processing the current scan
            if obstacle_detected:
                print("Obstacle detected in scan!")

            # Optionally, visualize or process obstacles further
            visualize_obstacles(dynamic_obstacle_nodes)
            time.sleep(0.1)  # Delay between scans (for processing)

    except KeyboardInterrupt:
        print("Stopping scan.")
    finally:
        lidar.stop_motor()
        lidar.disconnect()

def visualize_obstacles(dynamic_obstacle_nodes):
    """
    Visualize the obstacles detected by the LiDAR on a polar plot.
    Each obstacle is displayed based on its angle and distance.
    """
    if dynamic_obstacle_nodes:
        angles = [angle for angle, distance in dynamic_obstacle_nodes]
        distances = [distance for angle, distance in dynamic_obstacle_nodes]

        # Convert to polar coordinates for visualization
        plt.figure()
        plt.subplot(111, projection='polar')
        plt.scatter([math.radians(a) for a in angles], distances, c='r', label='Obstacles')
        plt.title("LiDAR Obstacle Detection")
        plt.legend()
        plt.show()

# Call the LiDAR processing function
process_lidar_data()
visualize_obstacles()





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
    path = {start: None}

    while priority_queue:
        current_node = heapq.heappop(priority_queue).name
        
        # If the goal is reached, reconstruct the path
        if current_node == goal:
            return reconstruct_path(path, start, goal)
        if current_node in visited:
            continue

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




# Define regions for the hierarchical routing (nodes belonging to different regions)


def generate_region_map(pos, rows, cols, region_rows, region_cols):
    region_map = {}
    region_count = 1

    for key, (x, y) in pos.items():
        # Determine the region based on rows and columns
        region_x = x // region_cols
        region_y = y // region_rows
        region_number = region_y * (cols // region_cols) + region_x + 1
        region_map[key] = region_number

    return region_map

# Define region dimensions (e.g., 5x5 cells per region)
region_rows, region_cols = 5, 5

# Generate the region map

# Define positions for better visualization layout (can be modified) 510 cm x 580 cm
def generate_coordinates(rows, cols):
    pos = {}
    letters = list(string.ascii_uppercase)  # A-Z

    # Extend keys beyond Z (e.g., AA, AB, ...)
    keys = []
    for length in range(1, 3):  # Support up to 2-character column names
        keys.extend([''.join(k) for k in itertools.product(letters, repeat=length)])

    # Generate the coordinates
    for y in range(rows):
        for x in range(cols):
            pos[keys[y*cols+x]] = (x, y)

    return pos
def generate_graph(pos, rows, cols):
    graph = {}

    for key, (x, y) in pos.items():
        neighbors2 = []

        # Check for valid neighbors (left, right, top, bottom)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                neighbor_key = next(k for k, v in pos.items() if v == (nx, ny))
                neighbors2.append(neighbor_key)

        graph[key] = neighbors2

    return graph
    
pos = generate_coordinates(24, 28)

graph=generate_graph(pos,24,28)

# Generate a 20x29 coordinate map
rows, cols = 25, 29

region_map = generate_region_map(pos, 25, 29, region_rows, region_cols)
# Heuristic values (assumed for this example)

obstacle_nodes = set()
x1,y1=6,26
x2,y2=18,29
for x,y in pos.items():
    
    if y1-1<=y[0]<=y2-1 and x1-1<=y[1]<=x2-1:
        
        obstacle_nodes.add(x)
x1,y1=13,6
x2,y2=17,20
for x,y in pos.items():
    
    if y1-1<=y[0]<=y2-1 and x1-1<=y[1]<=x2-1:
        
        obstacle_nodes.add(x)
x1,y1=20,6
x2,y2=23,20
for x,y in pos.items():
    
    if y1-1<=y[0]<=y2-1 and x1-1<=y[1]<=x2-1:
        
        obstacle_nodes.add(x)
x1,y1=7,6
x2,y2=12,10
for x,y in pos.items():
    
    if y1-1<=y[0]<=y2-1 and x1-1<=y[1]<=x2-1:
        
        obstacle_nodes.add(x)


def get_node_from_angle_distance(angle, distance, pos,current_node):
    """
    Convert LiDAR data (angle, distance) to a node name based on the coordinate grid.
    """
    angle_rad = math.radians(angle)
    x = distance * math.cos(angle_rad)
    y = distance * math.sin(angle_rad)
    x = x / 25.5
    y = y / 20
    x1, y1 = pos[current_node]
    closest_node = None
    min_distance = float('inf')

    

def dynamic_obstacle_recognition(obstacles, path, node_positions):
    # Simulate new obstacles (replace with real LiDAR processing logic)
    dynamic_updates = {}
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
    # Check each node in the path for obstacles
    for scan in lidar.iter_scans:
        for (angle,distance) in scan:
            if distance<OBSTACLE_THRESHOLD:
                x1, y1 = node_positions[current_node]
                x2, y2 = node_positions[next_node]
                if angle>315 or angle<45:
                    
                    if x2>x1:
                        print(f"Obstacle detected at {next_node}")
                        dynamic_updates.add(next_node)
                elif angle>45 and angle<135:
                    if y2>y1:
                        print(f"Obstacle detected at {next_node}")
                        dynamic_updates.add(next_node)
                elif angle>135 and angle<225:
                    if x1>x2:
                        print(f"Obstacle detected at {next_node}")
                        dynamic_updates.add(next_node)
                elif angle>225 and angle<315:
                    if y1>y2:
                        print(f"Obstacle detected at {next_node}")
                        dynamic_updates.add(next_node)

    # Add the dynamically detected obstacles to the main obstacle set
    obstacles.update(dynamic_updates)
    
    # Update the path with dynamic obstacles
    # Any node that has become an obstacle will be removed from the path
    updated_path = [node for node in path if node not in dynamic_updates]
    
    return updated_path, obstacles

motor_select_pins = {
    
    'M1': {'In1': 22, 'In2': 23},
    'M2': {'In1': 5, 'In2': 6},
    'M3': {'In1': 17,  'In2': 27},
    'M4': {'In1': 26, 'In2': 16}

}
for motor, pins in motor_select_pins.items():
    gpio.setup(pins['In1'], gpio.OUT)
    gpio.setup(pins['In2'], gpio.OUT)
pwm_pin=13
gpio.setup(pwm_pin, gpio.OUT)

pwm = gpio.PWM(pwm_pin, 100)  # 100 Hz frequency
pwm.start(0)
def reset_pins():
        for pin in motor_select_pins.values():
            pin.off()


def motor1(direction, speed):
    """Control Motor 1"""
    gpio.output(motor_select_pins['M1_In1'], direction == 'forward')
    gpio.output(motor_select_pins['M1_In2'], direction == 'backward')
    pwm.ChangeDutyCycle(speed)

def motor2(direction, speed):
    """Control Motor 2"""
    gpio.output(motor_select_pins['M2_In1'], direction == 'forward')
    gpio.output(motor_select_pins['M2_In2'], direction == 'backward')
    pwm.ChangeDutyCycle(speed)

def motor3(direction, speed):
    """Control Motor 3"""
    gpio.output(motor_select_pins['M3_In1'], direction == 'forward')
    gpio.output(motor_select_pins['M3_In2'], direction == 'backward')
    pwm.ChangeDutyCycle(speed)

def motor4(direction, speed):
    """Control Motor 4"""
    gpio.output(motor_select_pins['M4_In1'], direction == 'forward')
    gpio.output(motor_select_pins['M4_In2'], direction == 'backward')
    pwm.ChangeDutyCycle(speed)
def movement( direction2, duration=2):
    reset_pins()
    for pin in motor_select_pins.values():
        gpio.output(pin, gpio.LOW)

    if direction2 == "forward":
        motor1(direction='forward',speed=50)
        motor2(direction='forward',speed=50)
        motor3(direction='forward',speed=50)
        motor4(direction='forward',speed=50)
        

    elif direction2 == 'backward':
        motor1(direction='backward',speed=50)
        motor2(direction='backward',speed=50)
        motor3(direction='backward',speed=50)
        motor4(direction='backward',speed=50)
        
    elif direction2 == "right":
        motor1(direction='backward',speed=50)
        motor2(direction='forward',speed=50)
        motor3(direction='forward',speed=50)
        motor4(direction='backward',speed=50)
        
    elif direction2 == "left":
        motor1(direction='forward',speed=50)
        motor2(direction='backward',speed=50)
        motor3(direction='backward',speed=50)
        motor4(direction='forward',speed=50)

    print(f"Moving {direction2}")

    sleep(duration)

    

    # Reset all pins to low to stop motors completely
    for pin in motor_select_pins.values():
        gpio.output(pin, gpio.LOW)
    reset_pins()
    print("Motors stopped.")    
    
    
    #TRIBA NACI CURRENT NODE POSITION


    
def move_to_node(current_node, next_node, node_positions):
    print(f"Moving from {current_node} to {next_node}")
    x1, y1 = node_positions[current_node]
    x2, y2 = node_positions[next_node]

    if x2 > x1:
        movement("motor1", "motor2", "motor3", "motor4", direction="right", speed=1.0, duration=2)
    elif x2 < x1:
        movement("motor1", "motor2", "motor3", "motor4", direction="left", speed=1.0, duration=2)
    elif y2 > y1:
        movement("motor1", "motor2", "motor3", "motor4", direction="forward", speed=1.0, duration=2)
    elif y2 < y1:
        movement("motor1", "motor2", "motor3", "motor4", direction="backward", speed=1.0, duration=2)

def execute_path(path, node_positions):
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        move_to_node(current_node, next_node, node_positions)
        

start_node = 'BZ'
goal_node = 'XY'

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

# Perform Greedy Best-First Search for hierarchical routing


result_path = greedy_best_first_search_hierarchical(graph, start_node, goal_node, heuristic, region_map)
#dynamic_obstacles=dynamic_obstacle_recognition(obstacles,result_path,pos)
execute_path(result_path,pos)
# IMPLEMENT LIDAR DYNAMIC OBSTACLE RECOGNITION
print("Path from {} to {}: {}".format(start_node, goal_node, result_path))

# Visualize the graph and the found path
visualize_graph(graph, result_path, pos, region_map)

