import numpy as np
import random
from scipy.stats import gamma

# Simulation parameters
NUMBER_OF_LAYERS = 20  # originally 607)
TIMESTEP = 0.005  # hours (18 seconds)
END_TIME = 24  # 1 day for testing
TAU_E = 6.0  # Avg eclipse phase duration (hours)
TAU_I = 12.0  # Avg infection phase duration (hours)
NE = 30.0  # Eclipse compartments
NI = 100.0  # Infection compartments
PROBI = 0.2  # Cell-to-cell infection probability per hour
FUSION_PROB = 0.05  # Probability of fusion per hour for adjacent infected cells
INITIAL_INFECTED = 1  # Number of initially infected cells

# Cell states
HEALTHY = 'h'
ECLIPSE = 'e'
INFECTED = 'i'
DEAD = 'd'
FUSED = 'f'  # New state for fused cells

class ViralABM:
    def __init__(self, layers):
        self.layers = layers
        self.grid_size = 2 * layers - 1
        self.grid = np.full((self.grid_size, self.grid_size), 'o', dtype=str)  # 'o' for empty
        self.eclipse_times = np.zeros((self.grid_size, self.grid_size))
        self.infection_times = np.zeros((self.grid_size, self.grid_size))
        self.healthy_times = np.zeros((self.grid_size, self.grid_size))
        self.universal_times = np.zeros((self.grid_size, self.grid_size))
        self.initialize_grid()
        self.initialize_infected()

    def initialize_grid(self):
        # Simplified hexagonal grid setup (center cells are valid)
        radius = self.layers - 1
        center = self.grid_size // 2
        num_cells = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Approximate hexagonal grid by checking distance from center
                di = i - center
                dj = j - center
                dist = np.sqrt(di**2 + dj**2)
                if dist <= radius:
                    self.grid[i, j] = HEALTHY
                    self.eclipse_times[i, j] = gamma.rvs(NE, scale=TAU_E/NE)
                    self.infection_times[i, j] = gamma.rvs(NI, scale=TAU_I/NI)
                    num_cells += 1
        print(f"Initialized {num_cells} cells")

    def initialize_infected(self):
        # Randomly infect INITIAL_INFECTED cells
        healthy_cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i, j] == HEALTHY]
        infected = random.sample(healthy_cells, min(INITIAL_INFECTED, len(healthy_cells)))
        for i, j in infected:
            self.grid[i, j] = ECLIPSE
            self.eclipse_times[i, j] = gamma.rvs(NE, scale=TAU_E/NE)

    def get_neighbors(self, i, j):
        # Simplified hexagonal neighbors (6 directions)
        neighbors = [
            (i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j+1), (i+1, j-1)
        ]
        valid_neighbors = [
            (ni, nj) for ni, nj in neighbors
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size and self.grid[ni, nj] != 'o'
        ]
        return valid_neighbors

    def step(self):
        # Create a copy of the grid for updates
        new_grid = self.grid.copy()
        new_ecl = self.eclipse_times.copy()
        new_inf = self.infection_times.copy()

        # Track cells to process for fusion
        infected_cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i, j] == INFECTED]

        # Update healthy cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == HEALTHY:
                    self.healthy_times[i, j] += TIMESTEP
                    # Cell-to-cell infection
                    if random.random() < PROBI * TIMESTEP:
                        neighbors = self.get_neighbors(i, j)
                        infected_neighbors = [n for n in neighbors if self.grid[n[0], n[1]] == INFECTED]
                        if infected_neighbors:
                            new_grid[i, j] = ECLIPSE
                            new_ecl[i, j] = gamma.rvs(NE, scale=TAU_E/NE)

        # Update eclipse cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == ECLIPSE:
                    if self.universal_times[i, j] > (self.eclipse_times[i, j] + self.healthy_times[i, j]):
                        new_grid[i, j] = INFECTED
                        new_inf[i, j] = gamma.rvs(NI, scale=TAU_I/NI)

        # Update infected cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == INFECTED:
                    if self.universal_times[i, j] > (self.infection_times[i, j] + self.eclipse_times[i, j] + self.healthy_times[i, j]):
                        new_grid[i, j] = DEAD

        # Stochastic fusion for infected cells
        fused_pairs = set()
        random.shuffle(infected_cells)  # Randomize to avoid bias
        for i, j in infected_cells:
            if (i, j) in fused_pairs:
                continue
            neighbors = self.get_neighbors(i, j)
            infected_neighbors = [(ni, nj) for ni, nj in neighbors if self.grid[ni, nj] == INFECTED and (ni, nj) not in fused_pairs]
            if infected_neighbors and random.random() < FUSION_PROB * TIMESTEP:
                # Choose one neighbor to fuse with
                ni, nj = random.choice(infected_neighbors)
                # Mark both cells as fused
                new_grid[i, j] = FUSED
                new_grid[ni, nj] = FUSED
                fused_pairs.add((i, j))
                fused_pairs.add((ni, nj))
                # Optionally adjust infection time for fused cells
                new_inf[i, j] = gamma.rvs(NI, scale=TAU_I/NI * 1.5)  # Fused cells live longer
                new_inf[ni, nj] = new_inf[i, j]

        # Update fused cells (e.g., transition to dead)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == FUSED:
                    if self.universal_times[i, j] > (self.infection_times[i, j] + self.eclipse_times[i, j] + self.healthy_times[i, j]):
                        new_grid[i, j] = DEAD

        # Apply updates
        self.grid = new_grid
        self.eclipse_times = new_ecl
        self.infection_times = new_inf
        self.universal_times += TIMESTEP

    def run(self):
        steps = int(END_TIME / TIMESTEP)
        for step in range(steps):
            self.step()
            if step % (1 // TIMESTEP) == 0:  # Print every simulated hour
                self.print_summary(step * TIMESTEP)

    def print_summary(self, time):
        counts = {
            HEALTHY: 0, ECLIPSE: 0, INFECTED: 0, DEAD: 0, FUSED: 0
        }
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] in counts:
                    counts[self.grid[i, j]] += 1
        print(f"Time {time:.1f} hours: Healthy={counts[HEALTHY]}, Eclipse={counts[ECLIPSE]}, Infected={counts[INFECTED]}, Fused={counts[FUSED]}, Dead={counts[DEAD]}")

if __name__ == "__main__":
    model = ViralABM(NUMBER_OF_LAYERS)
    model.run()