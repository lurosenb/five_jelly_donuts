from itertools import groupby
from collections import Counter

import gym
import numpy as np
from scipy import stats


def entropy_of_sequence(input_list):
    # get counts
    count_data = list(Counter(input_list).values())
    
    # get entropy from counts
    entropy = stats.entropy(count_data)  
    
    return entropy


class MealPlanningEnv(gym.Env):
    metadata = {'render.modes': ['human', 'json']}

    def __init__(self, possible_meals, meal_categories, nutrition_data, num_meals):
        super(MealPlanningEnv, self).__init__()

        self.possible_meals = possible_meals
        self.meal_categories = meal_categories
        # add 'empty' meal category if not there
        if 'empty' not in self.meal_categories:
            self.meal_categories = np.append(self.meal_categories, ['empty'])
            
        self.num_possible_meals = len(self.possible_meals)
        self.unique_meal_categories = np.unique(self.meal_categories)
        self.num_meal_categories = len(self.unique_meal_categories)
        self.nutrition_data = nutrition_data
        self.num_meals = num_meals
        self.current_step = None
        self.action_space = gym.spaces.Discrete(self.num_possible_meals)
        
        self.nutrition_history_shape = (self.num_meals, len(self.nutrition_data.columns))
        self.observation_space = gym.spaces.Dict({
            'meal_history': gym.spaces.Box(
                low=0,
                high=self.num_possible_meals,
                shape=(self.num_meals,),
                dtype=np.int64
            ),
            'nutrition_history': gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=self.nutrition_history_shape,
                dtype=np.float32
            ),
            'category_history': gym.spaces.Box(
                low=0,
                high=self.num_meal_categories,
                shape=(self.num_meals,),
                dtype=np.int64
            ),
            'goal_nutrition': gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(len(nutrition_data.columns),),
                dtype=np.float32
            )
        })
    
    def _get_goal_nutrition(self):
        # this is the dietkit guideline for a sequence of 19 meals:
        # energy in [1260, 1540]
        # protein >= 20
        # fat in [15, 30]
        # carbs in [55, 65]
        # total dietary in [11, 20]
        # calcium in [500, 2500]
        # iron in [5, 40]
        # sodium < 1600
        # vitamin a in [230, 750]
        # vitamin b1 thiamine > 0.4
        # vitamine b2 riboflavine > 0.5
        # vitamin c in [35, 510]
        # linoleic acid in [4.6, 9.1]
        # alpha-linolenic acid in [0.6, 1.17]
        
        # set goal at min of each range
        goal_nutrition_for_19_meals = np.array([
            1260, 
            20,
            15,
            55,
            16,
            230,
            35,
            0.4,
            0.5,
            500,
            5,
            1600,
            4.6,
            0.6
        ])
        
        return goal_nutrition_for_19_meals * self.num_meals / 19
    
    def _calculate_current_nutrition(self):
        return self.nutrition_history[0:self.current_step, :].sum(axis=0)

    def reset(self):
        self.current_step = 0
        # set all meals to the 'empty' meal
        self.meal_history = np.zeros(self.num_meals) + self.num_possible_meals - 1
        self.nutrition_history = np.zeros(self.nutrition_history_shape)
        # set all categories to the 'empty' category
        self.category_history = np.zeros(self.num_meals)
        self.goal_nutrition = self._get_goal_nutrition()
        return self._next_observation()

    def step(self, action):
        chosen_meal = self.possible_meals[action]
        chosen_meal_category = self.meal_categories[action]
        nutrition = self.nutrition_data.loc[chosen_meal].values
        
        self.meal_history[self.current_step] = action
        self.nutrition_history[self.current_step, :] = nutrition
        self.category_history[self.current_step] = self.unique_meal_categories.tolist().index(chosen_meal_category)

        done = self.current_step == self.num_meals - 1
        info = {}

        self.current_step += 1

        reward = self._calculate_reward()

        return self._next_observation(), reward, done, info

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(f'Step: {self.current_step}')
            print(f'Chosen Meal: {self.possible_meals[self.meal_history[self.current_step - 1].astype(int)]}')
            print(f'Chosen Meal Category: {self.meal_categories[self.meal_history[self.current_step - 1].astype(int)]}')
            print(f'Meal History: {self.possible_meals[self.meal_history.astype(int)]}')
            print(f'Category History: {self.unique_meal_categories[self.category_history.astype(int)]}')
            print(f'Reward: {self._calculate_reward()}')
        else:
            pass

    def _next_observation(self):
        obs = {
            'meal_history': self.meal_history,
            'nutrition_history': self.nutrition_history,
            'category_history': self.category_history,
            'goal_nutrition': self.goal_nutrition
        }
        return obs

    def _calculate_reward(self):
        # take current percentage of nutrition met, treating each category equally, on [0, 1] scale
        current_nutrition = self._calculate_current_nutrition()
        nutrition_fractions = np.divide(current_nutrition, self.goal_nutrition)
        nutrition_fractions = np.array([min(fraction, 1) for fraction in nutrition_fractions])
        mean_nutrition_fraction = nutrition_fractions.mean()
        
        # calculate 3 measures of current compositional diversity, all on [0, 1] scale
        # 1. if there are category repetitions in a sequence of length unique_sequence_length, penalize
        unique_sequence_length = 3
        max_entropy_per_sequence = entropy_of_sequence(range(unique_sequence_length))
        entropy_fractions = []
        for start_index in range(self.num_meals - unique_sequence_length + 1):
            end_index = start_index + unique_sequence_length
            sequence_to_check = self.category_history[start_index:end_index]
            entropy_fraction = entropy_of_sequence(sequence_to_check) / max_entropy_per_sequence
            entropy_fractions.append(entropy_fraction)
        mean_sequence_entropy_fraction = np.mean(entropy_fractions)
        
        # 2. If there are more than num_allowed_in_a_row, penalize even more
        # count repetitions of each element in order
        sequential_counts = [(category, len(list(appearances))) for category, appearances in groupby(self.category_history)]
        max_num_allowed_in_a_row = 1
        total_num_repetitions = np.sum([appearances > max_num_allowed_in_a_row for _, appearances in sequential_counts])
        
        # 3. Check entropy of meals overall as fraction of max possible
        overall_entropy = entropy_of_sequence(self.meal_history)
        max_overall_entropy = entropy_of_sequence(list(range(self.num_meals)))
        overall_entropy_fraction = 1 - overall_entropy / max_overall_entropy
        
        # overall reward linear combo of nutrition and composition
        coef_nutrition = 2 / self.num_meals
        coef_sequence_entropy = 3 / self.num_meals
        coef_repetitions = -1 / self.num_meals
        coef_overall_entropy = 1 / self.num_meals
        reward = float(
            coef_nutrition * mean_nutrition_fraction + 
            coef_sequence_entropy * mean_sequence_entropy_fraction + 
            coef_repetitions * total_num_repetitions + 
            coef_overall_entropy * overall_entropy_fraction
        )
        # reward = -(100 * overall_entropy_fraction)**2
        
        # TODO: each day hit different categories, then penalize overall across meals not categories
        
        # reward = np.sum([self.possible_meals[meal_index] == self.possible_meals[0] for meal_index in self.meal_history.astype(int)])
        return reward