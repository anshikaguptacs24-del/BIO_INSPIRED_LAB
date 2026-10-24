
import math
import random

# Small TSP example: six cities
cities = [
    (0, 0), (2, 6), (5, 4),
    (7, 1), (6, 7), (1, 3)
]

N = len(cities)
ANTS = 20
ITERATIONS = 100

ALPHA = 1.0
BETA = 2.0
EVAPORATION = 0.1
Q = 100.0

# Levy ACO parameters inspired by the paper
P_THRESHOLD = 0.8
A = 9.5

def distance(i, j):
    x1, y1 = cities[i]
    x2, y2 = cities[j]
    return math.hypot(x1 - x2, y1 - y2)

dist = [[distance(i, j) for j in range(N)]
        for i in range(N)]

pheromone = [[1.0] * N for _ in range(N)]

def choose_next_city(current, unvisited):
    # Sort candidates by desirability, highest first
    candidates = sorted(
        unvisited,
        key=lambda j: (
            pheromone[current][j] ** ALPHA
            * (1.0 / dist[current][j]) ** BETA
        ),
        reverse=True
    )

    weights = [
        pheromone[current][j] ** ALPHA
        * (1.0 / dist[current][j]) ** BETA
        for j in candidates
    ]

    total = sum(weights)
    probabilities = [w / total for w in weights]

    # Random value used to select a candidate
    p_now = random.random()
    p_levy = random.random()

    # Levy-inspired adjustment:
    # encourage exploration of lower-ranked candidates
    if p_levy >= P_THRESHOLD:
        factor = (1.0 / A) * (
            (1.0 - p_levy) / (1.0 - P_THRESHOLD)
        )
        p_new = 1.0 - factor * (1.0 - p_now)
    else:
        p_new = p_now

    # Roulette-wheel selection using the adjusted value
    cumulative = 0.0
    for city, prob in zip(candidates, probabilities):
        cumulative += prob
        if p_new <= cumulative:
            return city

    return candidates[-1]


def tour_length(tour):
    length = sum(
        dist[tour[i]][tour[i + 1]]
        for i in range(N - 1)
    )
    return length + dist[tour[-1]][tour[0]]


best_tour = None
best_distance = float("inf")

for iteration in range(ITERATIONS):
    all_tours = []

    for ant in range(ANTS):
        start = random.randrange(N)
        tour = [start]
        unvisited = set(range(N))
        unvisited.remove(start)

        while unvisited:
            nxt = choose_next_city(tour[-1], unvisited)
            tour.append(nxt)
            unvisited.remove(nxt)

        length = tour_length(tour)
        all_tours.append((tour, length))

        if length < best_distance:
            best_distance = length
            best_tour = tour[:]

    # Evaporate pheromone
    for i in range(N):
        for j in range(N):
            pheromone[i][j] *= (1.0 - EVAPORATION)

    # Deposit pheromone: shorter tours deposit more
    for tour, length in all_tours:
        deposit = Q / length
        for i in range(N):
            a = tour[i]
            b = tour[(i + 1) % N]
            pheromone[a][b] += deposit
            pheromone[b][a] += deposit

    if (iteration + 1) % 10 == 0:
        print(
            f"Iteration {iteration + 1}: "
            f"Best distance = {best_distance:.2f}"
        )

print("\nBest tour:", best_tour + [best_tour[0]])
print(f"Best tour distance: {best_distance:.2f}")
