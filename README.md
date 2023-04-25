# Diet Planning with Reinforcement Learning
Final project for Reinforcement Learning.

## Description
Meal planning/dietary monitoring is a fundamental problem in models for personalized health and nutrition. There is a tension between optimizing personalized nutrition of diet vs. optimizing personalized satisfaction/happiness. Other potentially orthogonal factors include environmental impact and financial burden, so how best to optimize all of these factors is a challenging problem?

## Main Files
- `meal_planning_environment.py`: Main file with functionality. Contains the environment classes for the meal planning problem. Specifies subclasses for each of the particular reward types/environments we try (MaxNutritionEnv, HeuristicEnv, GPTEnv). Contains most of our plotting code and other helper functions as well. 

## Usage

Our meal planning environment is ready for use with gym as-is, and can be used as follows.

```{python}
from stable_baselines3 import PPO, DQN, A2C
from meal_planning_environment import HeuristicEnv, MaxNutritionEnv, HeuristicEnv, GPTOnlyEnv
from meal_planning_environment import run_with_learning_algorithm, load_data

# load data on meals, nutrition, and carbon impact using our data loader
possible_meals, meal_categories, nutrition_data, carbon_data = load_data()

# set a number of desired meals for the meal plan and instantiate the environment
# environment options include `MaxNutritionEnv`, `HeuristicEnv`, and `GPTOnlyEnv`
num_meals = 21
env = HeuristicEnv(
    possible_meals=possible_meals, 
    meal_categories=meal_categories, 
    nutrition_data=nutrition_data, 
    carbon_data=carbon_data, 
    num_meals=num_meals
)

# note that for the heuristic environment, the reward function can be modified via the `set_reward_weights()` method!
env.set_reward_weights(coef_nutrition_lower=1, coef_nutrition_upper=-0.5, coef_sequence_entropy=2, coef_repetitions=-2, coef_overall_entropy=2, coef_carbon=1)


# run with your favorite agent using our handy function `run_with_learning_algorithm()` helper function. See notebooks for additional functionality.
log_dir = 'path/to/log/directory'
run_with_learning_algorithm(algorithm=PPO, env=env, num_timesteps=1000, log_dir=log_dir, num_meals=21, seed=0)
```

## Notebooks
- `carbon_impacts.ipynb`: Notebook including tests for HeuristicEnv with just carbon impact and nutrition targets as the reward function.
- `generate_carbon_impacts.ipynb`: Notebook for calculating the carbon impact for each food item by querying ChatGPT 3 times with all of the ingredients in the food item. The carbon impact is the average of the 3 responses.
- `heuristic_composition.ipynb`: Notebook including tests for HeuristicEnv with carbon impact, nutrition targets, and composition targets (i.e. variety) as the reward function.
- `heuristic_nutrition_only.ipynb`: Notebook including tests for HeuristicEnv with only nutrition targets as the reward function.
- `max_nutrition.ipynb`: Notebook including tests for MaxNutritionEnv with maximizing nutrition as the reward function.
- `random_agent.ipynb` / `testing_random_agent.ipynb`: Notebooks with miscellaneous tests and including tests for RandomAgentEnv for HeuristicEnv.
- `test_gpt_only_env_algorithms.ipynb`: Notebook including tests for GPTEnv with RL "H" F reward function that queries ChatGPT for reward for a meal plan at any time step and then uses that as the reward + carbon impact.

#### Disclaimer
Notebooks include lots of plots and output. See slides for overview of results.

## Databases
- `gpt_responses.db`: sqlite3 database containing the responses from ChatGPT for each food item. The table is called `gpt_responses` and has the columns `meals` and `responses`. `meals` has the unique ids for each meal composing a meal plan and response has the chatGPT response for that mealplan.
- `carbon_impacts.db`: sqlite3 database containing the carbon impacts for each meal item. The table is called `carbon_impacts` and has the columns `meal` and `carbon_impact`.
