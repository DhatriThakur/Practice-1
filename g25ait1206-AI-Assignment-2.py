# model_ga.py – Genetic Algorithm for NAS with Roulette-Wheel Selection and weighted fitness
try:
    import torch  # type: ignore
    # Use attribute access to avoid static import resolution issues in some analyzers
    nn = torch.nn
    # Dynamically import torch.optim.AdamW to avoid static-analysis errors when torch is absent.
    import importlib
    import importlib.util

    optim = None
    AdamW = None
    if importlib.util.find_spec("torch.optim") is not None:
        optim = importlib.import_module("torch.optim")
        AdamW = getattr(optim, "AdamW", None)

    # If the real AdamW isn't available at runtime, provide a lightweight wrapper delegating to optim.Adam
    if AdamW is None:
        class AdamW:
            def __init__(self, params, lr=1e-3, weight_decay=0.01, **kwargs):
                if optim is None or not hasattr(optim, "Adam"):
                    # If torch.optim isn't available at runtime, raise a clear error
                    raise ImportError("torch.optim not available; install PyTorch to use AdamW")
                # Use Adam with weight decay as a pragmatic fallback
                self._opt = optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)

            def __getattr__(self, name):
                return getattr(self._opt, name)
except Exception as e:
    raise ImportError(
        "PyTorch (torch) is required but not installed or failed to import. "
        "Install it with 'pip install torch' or follow the instructions at https://pytorch.org/."
    ) from e

import random, os, json
from copy import deepcopy
# Adjust the path below if model_cnn.py is in a different directory
# For example: from .model_cnn import CNN (for relative imports)
# Or: import sys; sys.path.append('path/to/module'); from model_cnn import CNN
import sys
import importlib
import importlib.util

# Ensure current file directory is on sys.path for imports by name
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Robustly load model_cnn: prefer normal import, otherwise try loading from model_cnn.py path
_module_name = "model_cnn"
_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cnn.py")
CNN = None
try:
    # Try regular import first (works when module is installed or on sys.path)
    spec = importlib.util.find_spec(_module_name)
    if spec is not None:
        _mod = importlib.import_module(_module_name)
    elif os.path.exists(_module_path):
        # Load directly from file path if present in same directory
        spec = importlib.util.spec_from_file_location(_module_name, _module_path)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
    else:
        _mod = None

    if _mod is None or not hasattr(_mod, "CNN"):
        raise ImportError(f"module '{_module_name}' not found or missing 'CNN'")

    CNN = _mod.CNN
except Exception as e:
    raise ImportError(
        "model_cnn module could not be found or does not define 'CNN'. "
        "Ensure model_cnn.py is in the same directory as this file and defines a 'CNN' class."
    ) from e


# Define the search space for CNN architecture
class CNNSearchSpace:
    def __init__(self):
        self.conv_layers = [1, 2, 3, 4]
        self.filters = [16, 32, 64, 128]
        self.kernel_sizes = [3, 5, 7]
        self.pool_types = ['max', 'avg']
        self.activations = ['relu', 'leaky_relu']
        self.fc_units = [64, 128, 256, 512]

# Encode an architecture (chromosome) with its gene representation
class Architecture:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = self.random_genes()
        else:
            self.genes = genes
        self.fitness = 0
        self.accuracy = 0
        self.best_epoch = 0

    def random_genes(self):
        space = CNNSearchSpace()
        num_conv = random.choice(space.conv_layers)
        genes = {
            'num_conv': num_conv,
            'conv_configs': [],
            'pool_type': random.choice(space.pool_types),
            'activation': random.choice(space.activations),
            'fc_units': random.choice(space.fc_units)
        }
        for _ in range(num_conv):
            genes['conv_configs'].append({
                'filters': random.choice(space.filters),
                'kernel_size': random.choice(space.kernel_sizes)
            })
        return genes

    def __repr__(self):
        return f"Arch(conv={self.genes['num_conv']}, acc={self.accuracy:.4f})"

# Genetic Algorithm for Neural Architecture Search
class GeneticAlgorithm:
    def __init__(self, population_size=20, generations=10, mutation_rate=0.2, crossover_rate=0.7):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_architecture = None
        self.search_space = CNNSearchSpace()

    def initialize_population(self):
        # Initialize population with random architectures
        self.population = [Architecture() for _ in range(self.population_size)]

    def evaluate_fitness(self, architecture, train_loader, val_loader, device, epochs=100):
        """Train and evaluate a single architecture, returning its fitness score."""
        try:
            model = CNN(architecture.genes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = AdamW(model.parameters(), lr=0.001)
            # Quick training loop (with early stopping patience)
            best_acc = 0.0
            best_epoch = 1
            patience = 10
            step = 1
            for epoch in range(1, epochs + 1):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                # Validation evaluation
                model.eval()
                correct = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels).sum().item()
                accuracy = correct / len(val_loader.dataset)
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_epoch = epoch
                    step = 0
                else:
                    step += 1
                if step >= patience:
                    break
            # Compute number of parameters in Conv blocks vs FC layers
            conv_params = sum(p.numel() for p in model.features.parameters())
            fc_params   = sum(p.numel() for p in model.classifier.parameters())
            # Normalize parameter counts (per million) and apply weighted penalty
            conv_penalty = conv_params / 1e6
            fc_penalty   = fc_params / 1e6
            architecture.accuracy = best_acc
            architecture.best_epoch = best_epoch
            architecture.fitness = best_acc - (0.02 * conv_penalty + 0.01 * fc_penalty)
            # Clean up and return fitness
            del model, inputs, outputs, labels
            torch.cuda.empty_cache()
            return architecture.fitness
        except Exception as e:
            print(f"Error evaluating architecture: {e}", flush=True)
            architecture.fitness = 0
            architecture.accuracy = 0
            return 0

    def selection(self):
        """Roulette-Wheel selection: choose individuals with probability proportional to fitness."""
        selected = []
        total_fitness = sum(ind.fitness for ind in self.population)
        for _ in range(self.population_size):
            pick = random.random() * total_fitness
            current = 0.0
            for ind in self.population:
                current += ind.fitness
                if current >= pick:
                    selected.append(ind)
                    break
        return selected

    def crossover(self, parent1, parent2):
        """Single-point crossover on two parent architectures' genes."""
        # If no crossover occurs, return copies of parents
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        # Otherwise, perform crossover on gene encoding
        child1_genes = deepcopy(parent1.genes)
        child2_genes = deepcopy(parent2.genes)
        # Randomly swap certain gene components
        if random.random() < 0.5:
            child1_genes['num_conv'], child2_genes['num_conv'] = child2_genes['num_conv'], child1_genes['num_conv']
        if random.random() < 0.5:
            child1_genes['pool_type'], child2_genes['pool_type'] = child2_genes['pool_type'], child1_genes['pool_type']
            child1_genes['activation'], child2_genes['activation'] = child2_genes['activation'], child1_genes['activation']
        # Adjust conv_configs list lengths to match new num_conv for each child
        min_len = min(child1_genes['num_conv'], len(child1_genes['conv_configs']))
        child1_genes['conv_configs'] = child1_genes['conv_configs'][:min_len]
        while len(child1_genes['conv_configs']) < child1_genes['num_conv']:
            child1_genes['conv_configs'].append({
                'filters': random.choice(self.search_space.filters),
                'kernel_size': random.choice(self.search_space.kernel_sizes)
            })
        min_len = min(child2_genes['num_conv'], len(child2_genes['conv_configs']))
        child2_genes['conv_configs'] = child2_genes['conv_configs'][:min_len]
        while len(child2_genes['conv_configs']) < child2_genes['num_conv']:
            child2_genes['conv_configs'].append({
                'filters': random.choice(self.search_space.filters),
                'kernel_size': random.choice(self.search_space.kernel_sizes)
            })
        return Architecture(child1_genes), Architecture(child2_genes)

    def mutation(self, architecture):
        """Randomly mutate an architecture's genes."""
        # Mutate number of conv layers
        if random.random() < self.mutation_rate:
            architecture.genes['num_conv'] = random.choice(self.search_space.conv_layers)
        # Mutate conv layer filter counts or kernel sizes
        for conv in architecture.genes['conv_configs']:
            if random.random() < self.mutation_rate:
                conv['filters'] = random.choice(self.search_space.filters)
            if random.random() < self.mutation_rate:
                conv['kernel_size'] = random.choice(self.search_space.kernel_sizes)
        # Mutate pooling type, activation function, and FC layer size
        if random.random() < self.mutation_rate:
            architecture.genes['pool_type'] = random.choice(self.search_space.pool_types)
        if random.random() < self.mutation_rate:
            architecture.genes['activation'] = random.choice(self.search_space.activations)
        if random.random() < self.mutation_rate:
            architecture.genes['fc_units'] = random.choice(self.search_space.fc_units)
        return architecture

    def evolve(self, train_loader, val_loader, device, run=1):
        # Initialize and evaluate initial population
        self.initialize_population()
        print(f"Starting with {self.population_size} Population:\n{self.population}\n", flush=True)
        self.best_architecture = None
        for generation in range(self.generations):
            print(f"\n{'='*60}", flush=True)
            print(f"Generation {generation + 1}/{self.generations}", flush=True)
            print(f"{'='*60}", flush=True)
            # Evaluate all individuals in current population
            for i, arch in enumerate(self.population):
                print(f"Evaluating architecture {i+1}/{self.population_size}...", end=' ', flush=True)
                fitness = self.evaluate_fitness(arch, train_loader, val_loader, device, epochs=20)
                print(f"Fitness: {fitness:.4f}, Accuracy: {arch.accuracy:.4f}", flush=True)
            # Sort population by fitness (descending)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            # Update global best architecture
            if self.best_architecture is None or self.population[0].fitness > self.best_architecture.fitness:
                self.best_architecture = deepcopy(self.population[0])
            print(f"\nSorting population by fitness (high -> low)...", flush=True)
            print(f"Best in generation: {self.population[0]}\n", flush=True)
            print(f"Best overall: {self.best_architecture}", flush=True)
            # Selection (Roulette-Wheel)
            print(f"\nPerforming roulette-wheel selection on population of size {self.population_size}...", flush=True)
            mating_pool = self.selection()
            # Crossover and mutation to form new population
            print(f"Performing Crossover & Mutation...", flush=True)
            new_population = []
            # Elitism: carry over top 2 individuals unchanged
            new_population.extend([deepcopy(self.population[0]), deepcopy(self.population[1])])
            print(f"Elitism: Keeping top 2 architectures for next generation.", flush=True)
            # Fill the rest of new_population
            while len(new_population) < self.population_size:
                parent1 = random.choice(mating_pool)
                parent2 = random.choice(mating_pool)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_population.extend([child1, child2])
            # Prepare next generation (trim if one extra child)
            self.population = new_population[:self.population_size]
            print(f"Next Generation: {self.population}", flush=True)
        # Final evaluation of the best architecture
        final_model = CNN(self.best_architecture.genes).to(device)
        total_params = sum(p.numel() for p in final_model.parameters())
        print(f"\nTotal parameters: {total_params:,}", flush=True)
        print(f"\nModel architecture:\n{final_model}", flush=True)
        return self.best_architecture
