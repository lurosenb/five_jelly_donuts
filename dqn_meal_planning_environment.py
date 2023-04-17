import gym
import numpy as np


class MealPlanningEnv(gym.Env):
    metadata = {'render.modes': ['human', 'json']}

    def __init__(self, possible_meals, nutrition_data, num_meals):
        super(MealPlanningEnv, self).__init__()

        self.possible_meals = possible_meals
        self.num_possible_meals = len(self.possible_meals)
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
            'goal_nutrition': gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(len(nutrition_data.columns),),
                dtype=np.float32
            )
        })
    
    def _get_goal_nutrition(self):
        return self.nutrition_data.mean().values * self.num_meals
    
    def _calculate_current_nutrition(self):
        return self.nutrition_history[0:self.current_step, :].sum(axis=0)

    def reset(self):
        self.current_step = 0
        # set all meals to the 'empty' meal
        self.meal_history = np.zeros(self.num_meals) + self.num_possible_meals - 1
        self.nutrition_history = np.zeros(self.nutrition_history_shape)
        self.goal_nutrition = self._get_goal_nutrition()
        return self._next_observation()

    def step(self, action):
        chosen_meal = self.possible_meals[action]
        nutrition = self.nutrition_data.loc[chosen_meal].values
        self.meal_history[self.current_step] = action
        self.nutrition_history[self.current_step, :] = nutrition

        reward = self._calculate_reward()
        done = self.current_step == self.num_meals - 1
        info = {}

        self.current_step += 1

        return self._next_observation(), reward, done, info

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(f'Step: {self.current_step}')
            print(f'Chosen Meal: {self.possible_meals[self.meal_history[self.current_step - 1].astype(int)]}')
            print(f'Meal History: {self.possible_meals[self.meal_history.astype(int)]}')
            print(f'Nutrition History: {self.nutrition_history.round(2)}')
            print(f'Goal Nutrition: {self.goal_nutrition.round(2)}')
            print(f'Current Nutrition: {self._calculate_current_nutrition().round(2)}')
            print(f'Reward: {self._calculate_reward().round(2)}')
            display(None)
        else:
            pass

    def _next_observation(self):
        obs = {
            'meal_history': self.meal_history,
            'nutrition_history': self.nutrition_history,
            'goal_nutrition': self.goal_nutrition
        }
        return obs['nutrition_history']

    def _calculate_reward(self):
        current_nutrition = self._calculate_current_nutrition()
        diff_from_goal = current_nutrition - self.goal_nutrition
        reward = -np.abs(diff_from_goal).sum()
        # reward = current_nutrition.sum()
        return reward