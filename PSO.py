
import random

# Sample fitness function to simulate model accuracy
# The target values represent a good combination
# of five hyper-parameters.
def fitness(position):
    target = [32, 0.01, 0.5, 5, 0.2]

    error = sum(
        (position[i] - target[i]) ** 2
        for i in range(len(position))
    )

    # Higher fitness means a better solution
    return 100 / (1 + error)


# PSO parameters
NUM_PARTICLES = 20
ITERATIONS = 50
DIMENSIONS = 5

LOWER = [8, 0.001, 0.1, 3, 0.1]
UPPER = [64, 0.021, 0.9, 7, 0.3]

W = 0.7
C1 = 2.0
C2 = 2.0

# Initialize particle positions and velocities
particles = [
    [random.uniform(LOWER[d], UPPER[d])
     for d in range(DIMENSIONS)]
    for _ in range(NUM_PARTICLES)
]

velocities = [
    [random.uniform(-1, 1)
     for _ in range(DIMENSIONS)]
    for _ in range(NUM_PARTICLES)
]

personal_best = [p[:] for p in particles]
personal_scores = [fitness(p) for p in particles]

best_index = personal_scores.index(max(personal_scores))
global_best = personal_best[best_index][:]
global_score = personal_scores[best_index]

# PSO iterations
for iteration in range(ITERATIONS):

    for i in range(NUM_PARTICLES):
        score = fitness(particles[i])

        # Update personal best
        if score > personal_scores[i]:
            personal_scores[i] = score
            personal_best[i] = particles[i][:]

        # Update global best
        if score > global_score:
            global_score = score
            global_best = particles[i][:]

        # Update velocity and position
        for d in range(DIMENSIONS):
            r1 = random.random()
            r2 = random.random()

            velocities[i][d] = (
                W * velocities[i][d]
                + C1 * r1 *
                (personal_best[i][d] - particles[i][d])
                + C2 * r2 *
                (global_best[d] - particles[i][d])
            )

            particles[i][d] += velocities[i][d]

            # Keep values within valid bounds
            particles[i][d] = max(
                LOWER[d],
                min(UPPER[d], particles[i][d])
            )

    if (iteration + 1) % 10 == 0:
        print(
            f"Iteration {iteration + 1}: "
            f"Best fitness = {global_score:.4f}"
        )

print("\nBest hyper-parameter combination:")
print("Neurons:", round(global_best[0]))
print("Learning rate:", round(global_best[1], 4))
print("Momentum:", round(global_best[2], 3))
print("Local epochs:", round(global_best[3]))
print("Client fraction:", round(global_best[4], 3))
print("Best fitness:", round(global_score, 4))
