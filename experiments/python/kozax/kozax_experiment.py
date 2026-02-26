import jax
import jax.numpy as jnp
import jax.random as jr
from kozax.fitness_functions.base_fitness_function import BaseFitnessFunction
from kozax.genetic_programming import GeneticProgramming

key = jr.PRNGKey(0)  # Initialize key
data_key, gp_key = jr.split(key)  # Split key for data and genetic programming
x = jr.uniform(data_key, shape=(30,), minval=-5, maxval=5)  # Inputs
y = -0.1 * x**3 + 0.3 * x**2 + 1.5 * x  # Targets


class FitnessFunction(BaseFitnessFunction):
    """
    The fitness function inherits the class BaseFitnessFunction and should implement the __call__ function, with the candidate, data and tree_evaluator as inputs. The tree_evaluator is used to compute the value of the candidate for each input. jax.vmap is used to vectorize the evaluation of the candidate over the inputs. The candidate's predictions are used to compute the fitness value with the mean squared error.
    """

    def __call__(self, candidate, data, tree_evaluator):
        X, Y = data
        predictions = jax.vmap(tree_evaluator, in_axes=[None, 0])(candidate, X)
        return jnp.mean(jnp.square(predictions - Y))


fitness_function = FitnessFunction()

# Define hyperparameters
population_size = 500
num_generations = 100

# Initialize genetic programming strategy
strategy = GeneticProgramming(num_generations, population_size, fitness_function)

# Fit the strategy on the data. With verbose, we can print the intermediate solutions.
strategy.fit(gp_key, (x, y), verbose=True)
