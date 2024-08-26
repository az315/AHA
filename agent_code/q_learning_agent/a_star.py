import heapq
import numpy as np

def a_star(start, goal, grid):
    # Start- und Zielkoordinaten
    x_start, y_start = start
    x_goal, y_goal = goal

    # Kosten für den Startpunkt
    g_cost = {start: 0}
    # Heuristischer Wert für den Startpunkt (euklidische Distanz zum Ziel)
    f_cost = {start: np.linalg.norm(np.array(goal) - np.array(start))}

    # Priority Queue für A*
    open_set = []
    heapq.heappush(open_set, (f_cost[start], start))

    # A* Algorithmus
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            return g_cost[current]  # Gesamtkosten zum Ziel (kürzester Pfad)
        
        x_current, y_current = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (x_current + dx, y_current + dy)
            x, y = neighbor
            
            # Wenn Nachbar im Gitter und begehbar (kein Hindernis)
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] == 0:
                tentative_g_cost = g_cost[current] + 1  # Jeder Schritt kostet 1
                if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g_cost
                    f_cost[neighbor] = tentative_g_cost + np.linalg.norm(np.array(goal) - np.array(neighbor))
                    heapq.heappush(open_set, (f_cost[neighbor], neighbor))
    
    return float('inf')  # Wenn kein Pfad gefunden wurde

def find_nearest_coin_with_a_star(own_position, coins, grid):
    nearest_coin = None
    min_distance = float('inf')

    for coin in coins:
        distance = a_star(own_position, coin, grid)
        if distance < min_distance:
            min_distance = distance
            nearest_coin = coin

    if nearest_coin is None:
        return (0, 0)  # Keine Münzen vorhanden
    else:
        coin_rel_pos = (nearest_coin[0] - own_position[0], nearest_coin[1] - own_position[1])
        return coin_rel_pos
