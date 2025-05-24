# traveling salesman problem

import random
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


random.seed(42)


class TSPGeneticAlgorithm:
    def __init__(self, num_locations=10, population_size=50, mutation_rate=0.1, elite_size=2):
        self.num_locations = num_locations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.locations = self._generate_locations()
        self.population = self._initialize_population()
        
    def _generate_locations(self):
        return [(random.random(), random.random()) for _ in range(self.num_locations)]
    
    def _initialize_population(self):
        return [random.sample(range(self.num_locations), self.num_locations) 
                for _ in range(self.population_size)]
    
    def calculate_distance(self, route):
        distance = 0
        for i in range(len(route)):
            from_city = self.locations[route[i]]
            to_city = self.locations[route[(i + 1) % len(route)]]
            distance += math.sqrt((from_city[0] - to_city[0])**2 + (from_city[1] - to_city[1])**2)
        return distance
    
    def calculate_fitness(self, route):
        distance = self.calculate_distance(route)
        return 1 / (distance + 1e-10)  # Avoid division by zero
    
    def select_parents(self):
        # Roulette wheel selection
        fitnesses = [self.calculate_fitness(route) for route in self.population]
        total_fitness = sum(fitnesses)
        probabilities = [f/total_fitness for f in fitnesses]
        
        # Select two parents
        parents = []
        for _ in range(2):
            r = random.random()
            cumsum = 0
            for i, p in enumerate(probabilities):
                cumsum += p
                if cumsum >= r:
                    parents.append(self.population[i])
                    break
        return parents
    
    def crossover(self, parent1, parent2):
        # Use PMX crossover
        size = len(parent1)
        child1, child2 = [-1] * size, [-1] * size
        
        # Randomly select crossover points
        point1, point2 = sorted(random.sample(range(size), 2))
        
        # Copy crossover segments
        child1[point1:point2] = parent1[point1:point2]
        child2[point1:point2] = parent2[point1:point2]
        
        # Create mapping relationships
        mapping1 = {parent1[i]: parent2[i] for i in range(point1, point2)}
        mapping2 = {parent2[i]: parent1[i] for i in range(point1, point2)}
        
        # Fill remaining positions
        for i in range(size):
            if i < point1 or i >= point2:
                # For child1
                if parent2[i] not in child1:
                    child1[i] = parent2[i]
                else:
                    value = parent2[i]
                    while value in child1:
                        value = mapping1.get(value, value)
                    child1[i] = value
                
                # For child2
                if parent1[i] not in child2:
                    child2[i] = parent1[i]
                else:
                    value = parent1[i]
                    while value in child2:
                        value = mapping2.get(value, value)
                    child2[i] = value
        
        return child1, child2
    
    def mutate(self, route):
        if random.random() < self.mutation_rate:
            # Randomly select two positions to swap
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route
    
    def get_elite(self):
        # Get the best individuals
        fitnesses = [self.calculate_fitness(route) for route in self.population]
        elite_indices = np.argsort(fitnesses)[-self.elite_size:]
        return [self.population[i] for i in elite_indices]
    
    def evolve(self, generations):
        best_distances = []
        best_route = None
        best_distance = float('inf')
        
        for generation in tqdm(range(generations)):
            # Save elite individuals
            elite = self.get_elite()
            
            # Generate new population
            new_population = []
            while len(new_population) < self.population_size - self.elite_size:
                parents = self.select_parents()
                child1, child2 = self.crossover(parents[0], parents[1])
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            
            # Add elite individuals
            new_population.extend(elite)
            self.population = new_population
            
            # Update best solution
            current_best = min(self.population, key=self.calculate_distance)
            current_best_distance = self.calculate_distance(current_best)
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_route = current_best
            
            best_distances.append(best_distance)
            
            # Print progress
            # if generation % 100 == 0:
            #     print(f"Generation {generation}, Best Distance: {best_distance:.2f}")
        
        return best_route, best_distances
    
    def plot_results(self, best_distances):
        plt.figure(figsize=(10, 6))
        plt.plot(best_distances)
        plt.title('Convergence of Genetic Algorithm')
        plt.xlabel('Generation')
        plt.ylabel('Best Distance')
        plt.grid(True)
        plt.show()
    
    def plot_route(self, route):
        plt.figure(figsize=(8, 8))
        x = [self.locations[i][0] for i in route]
        y = [self.locations[i][1] for i in route]
        x.append(x[0])  # Close the path
        y.append(y[0])
        
        plt.plot(x, y, 'b-', linewidth=1)
        plt.scatter(x[:-1], y[:-1], c='red', s=100)
        plt.title('Best Route Found')
        plt.grid(True)
        plt.show()


# Run the algorithm
if __name__ == "__main__":
    # Create algorithm instance
    ga = TSPGeneticAlgorithm(num_locations=100, population_size=50, mutation_rate=0.1, elite_size=2)
    
    # Run the algorithm
    best_route, best_distances = ga.evolve(generations=1000)
    
    # Display results
    print(f"\nBest route found: {best_route}")
    print(f"Best distance: {ga.calculate_distance(best_route):.2f}")
    
    # Plot results
    ga.plot_results(best_distances)
    ga.plot_route(best_route)