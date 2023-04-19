import gym
import numpy as np

import openai
import pandas as pd
import re

import sqlite3

class MealPlanningEnv(gym.Env):
    ## We want to prompt with all meals in the meal plan, and then get a rating for each meal

    # then, after warmup, we want to batch for input into sets of 3 meals 

    # add or subtract reward "at the end of each day" (after 3 meals)
    metadata = {'render.modes': ['human', 'json']}

    def __init__(self, possible_meals, meal_objects, nutrition_data, num_meals, gpt_warmup_steps=100):
        super(MealPlanningEnv, self).__init__()

        self.possible_meals = possible_meals
        self.meal_objects = meal_objects
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

        # Accrue meal text for prompting
        self.meal_text_history = ""

        with open('gpt_key.txt', 'r') as file:
            openai.api_key = file.read().rstrip()

        # store previous responses in sqlite database to avoid re-prompting
        # try to load from database, otherwise create new
        self.sql_conn = sqlite3.connect('response_store.db')
        self.sql_conn.execute('CREATE TABLE IF NOT EXISTS response_store (meal text, response text)')
        self.sql_conn.commit()

    def save_response_df(self):
        # save response dataframe to sqlite database
        # self.response_df.to_sql('response_df', con=sqlite3.connect('response_df.db'), if_exists='replace')
        pass
    
    def add_response(self, meal, response, verbose=False):
        # add response to dataframe using concat
        # self.response_df = pd.concat([self.response_df, pd.DataFrame({'meal': [meal], 'response': [response]})])

        # add response to sqlite database
        try:
            self.sql_conn.execute(f'INSERT INTO response_store VALUES ("{meal}", "{response}")')
            self.sql_conn.commit()
        except:
            if verbose:
                print('Error adding response to database')

    def retrieve_response(self, meal, verbose=False):
        # retrieve response from response_store database
        cursor = self.sql_conn.execute(f'SELECT response FROM response_store WHERE meal="{meal}"')
        response = cursor.fetchone()
        if response is not None:
            if verbose:
                print('Retrieved response from database')
            return response[0]
        else:
            return None

    def get_prompt(self, prompt_type):
        prompt_types = {
            'Lucas_per_meal': 'I am a 6ft tall 25 year old male, 168 pounds. I am very active.\
                Please rate the following proposed meal on a scale of 1-10, where\
                    1 is an unhealthy or unnatural meal and 10 is a perfect, balanced healthy meal.\n',
            'Lucas_full': f'I am a 6ft tall 25 year old male, 168 pounds. I am very active.\
                Please rate the following proposed meal plan over a period of {self.num_meals} meals on a scale of 1-10, where\
                    1 is an unhealthy plan and 10 is a perfect, balanced healthy plan.\n',
        }  
        return prompt_types[prompt_type]

    def nutrition_to_string(self, nutrition):
        nutrition_string = ''
        for key in nutrition:
            nutrition_string += key + ': ' + str(nutrition[key]) + ', '
        return nutrition_string

    def meal_to_string(self, meal):
        return meal.name.replace('S ', '') + ' (' + self.nutrition_to_string(meal.nutrition) + ')'

    def parse_meal_rating(self, response):
        # try to find a number in the response
        try:
            # 'a X out of 10'
            num = re.search(r'\d+',re.search(r'\d+ out of \d+', response).group()).group()
        except:
            try:
                # 'a X/10'
                num = re.search(r'\d+',re.search(r'\d+/\d+', response).group()).group()
            except:
                # 'a X'
                num = re.search(r'\d+', response).group()
            
        return int(num)
    
    def meal_reward_from_chatgpt(self, chosen_meal, chosen_meal_string, verbose=False, person_type='Lucas_per_meal', save_response=True):
        """
        Given a proposed meal, get a reward from the chatgpt model.
        """
        # print(chosen_meal_string)
        prompt = self.get_prompt(person_type)
        for meal_type, meal_value in chosen_meal_string.items():
            prompt += f'{meal_type}: {meal_value}\n'
        prompt += 'Please explain your answer, but end with a single numeric rating.'

        # check if we've checked this meal before,
        # if so, retrieve the response (no need to requery)
        response = self.retrieve_response(chosen_meal)
        if response is not None:
            if verbose:
                print(response)
            return self.parse_meal_rating(response)
        else:
            if verbose:
                print('No response found, querying chatgpt...')
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a helpful assistant and an expert in nutrition and meal planning."},
                        {"role": "user", "content": prompt},
                    ]
            )
            response = resp['choices'][0]['message']['content']
            if save_response:
                self.add_response(chosen_meal, response)
                # self.save_response_df()
            if verbose:
                print(response)
        return self.parse_meal_rating(response)

    def _get_goal_nutrition(self):
        return self.nutrition_data.mean().values * self.num_meals
    
    def _calculate_current_nutrition(self):
        return self.nutrition_history[0:self.current_step, :].sum(axis=0)

    def reset(self):
        self.current_step = 0
        # set all meals to the 'empty' meal
        self.meal_history = np.zeros(self.num_meals) + self.num_possible_meals - 1
        self.nutrition_history = np.zeros(self.nutrition_history_shape)
        self.meal_text_history = ""
        self.goal_nutrition = self._get_goal_nutrition()
        return self._next_observation()

    def step(self, action):
        chosen_meal = self.possible_meals[action]
        nutrition = self.nutrition_data.loc[chosen_meal].values
        self.meal_history[self.current_step] = action
        self.nutrition_history[self.current_step, :] = nutrition

        done = self.current_step == self.num_meals - 1
        reward = self._calculate_reward(chosen_meal, done=done)
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
            print(f'Reward: {self._calculate_reward(self.possible_meals[self.meal_history[self.current_step - 1].astype(int)])}')
        else:
            pass

    def _next_observation(self):
        obs = {
            'meal_history': self.meal_history,
            'nutrition_history': self.nutrition_history,
            'goal_nutrition': self.goal_nutrition
        }
        return obs

    def _calculate_reward(self, chosen_meal, done=False):
        ## TODO: Replace with GPT Warmup
        ## ALSO, prompt for carbon impact of meals as well
        ## ALSO, add long term reward
        try:
            if not done:
                self.full_text_history = self.meal_text_history + self.meal_to_string(self.meal_objects[chosen_meal]) + '\n\n'
                meal_reward = self.meal_reward_from_chatgpt(chosen_meal, {'meal': self.meal_to_string(self.meal_objects[chosen_meal])}, verbose=False, person_type='Lucas_per_meal')
                return float(meal_reward)
            else:
                full_plan_reward = self.meal_reward_from_chatgpt(chosen_meal, {'meal': self.meal_to_string(self.meal_objects[chosen_meal])}, verbose=False, person_type='Lucas_full')
                return float(full_plan_reward)
        except:
            return 0.0