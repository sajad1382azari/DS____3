import numpy as np 
import random 
from sklearn.preprocessing import OneHotEncoder 
from tensorflow.keras.datasets import mnist 

dummy_images = np.random.rand(5, 28*28).astype(np.float32) 
nput_filename = '/input.txt' 
with open(input_filename, 'w') as file: 
    for image in dummy_images: 
        file.write(' '.join(map(str, image)) + '\n') 
input_filename 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train = x_train.reshape(-1, 28*28).astype(np.float32) / 255.0 
x_test = x_test.reshape(-1, 28*28).astype(np.float32) / 255.0 
encoder = OneHotEncoder(sparse=False) 
y_train = encoder.fit_transform(y_train.reshape(-1, 1)) 
y_test = encoder.transform(y_test.reshape(-1, 1)) 


operations = ['+', '-', '*', 'mean', 'max', 'min'] 
def create_random_tree(depth=3): 
    if depth == 0 or random.random() > 0.5: 
        return ('const', np.random.rand()) 
    else: 
        op = random.choice(operations) 
        if op in ['+', '-', '*']: 
            return (op, create_random_tree(depth-1), create_random_tree(depth-1)) 
        else: 
            return (op, create_random_tree(depth-1)) 

def tree_to_function(tree, X): 
    if tree[0] == 'const': 
        return tree[1] 
    elif tree[0] == '+': 
        return tree_to_function(tree[1], X) + tree_to_function(tree[2], X) 
    elif tree[0] == '-': 
        return tree_to_function(tree[1], X) - tree_to_function(tree[2], X) 
    elif tree[0] == '*': 
        return tree_to_function(tree[1], X) * tree_to_function(tree[2], X) 
    elif tree[0] == 'mean': 
        return np.mean(tree_to_function(tree[1], X)) 
    elif tree[0] == 'max': 
        return np.max(tree_to_function(tree[1], X)) 
    elif tree[0] == 'min': 
        return np.min(tree_to_function(tree[1], X))

def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def evaluate_tree(tree, X):
    logits = np.zeros((X.shape[0], 10))
    for i in range(10):
        logits[:, i] = tree_to_function(tree, X)
    return softmax(logits)

class GeneticAlgorithm:
    def __init__(self, population_size, n_generations, mutation_rate):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        return [create_random_tree() for _ in range(self.population_size)]

    def fitness_function(self, individual, X, y):
        logits = evaluate_tree(individual, X)
        loss = -np.mean(np.sum(y * np.log(logits), axis=1))
        return loss

    def selection(self):
        fitness_scores = [self.fitness_function(ind, x_train, y_train) for ind in self.population]
        sorted_population = [ind for _, ind in sorted(zip(fitness_scores, self.population))]
        return sorted_population[:self.population_size // 2]

    def crossover(self, parent1, parent2):
        if random.random() > 0.5:
            return parent1
        else:
            return parent2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            return create_random_tree()
        else:
            return individual

    def evolve(self, X, y): 
        for generation in range(self.n_generations): 
            selected_individuals = self.selection() 
            next_population = [] 
            while len(next_population) < self.population_size: 
                parent1, parent2 = random.sample(selected_individuals, 2) 
                offspring = self.crossover(parent1, parent2) 
                offspring = self.mutate(offspring) 
                next_population.append(offspring) 
            self.population = next_population 
            fitness_scores = [self.fitness_function(ind, x_train, y_train) for ind in self.population] 
            print(f'Generation {generation + 1}: Best Fitness = {min(fitness_scores)}') 

    def predict(self, X): 
        best_individual = min(self.population, key=lambda ind: self.fitness_function(ind, x_train, y_train)) 
        return evaluate_tree(best_individual, X) 

population_size = 50 
n_generations = 50 

mutation_rate = 0.1 

ga = GeneticAlgorithm(population_size, n_generations, mutation_rate) 
ga.evolve(x_train, y_train) 
y_pred = ga.predict(x_test) 
accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) 
print(f'Test Accuracy: {accuracy * 100:.2f}%') 

def read_input_file(filename): 
    with open(filename, 'r') as file: 
        images = [list(map(float, line.strip().split())) for line in file] 
    return np.array(images) 

def write_output_file(filename, results): 
    with open(filename, 'w') as file: 
        for result in results: 
            file.write(f'{result}\n') 

input_file = '/content/input.txt' 
output_file = 'output.txt' 
input_images = read_input_file(input_file) 
predictions = ga.predict(input_images) 
write_output_file(output_file, np.argmax(predictions, axis=1))


 
