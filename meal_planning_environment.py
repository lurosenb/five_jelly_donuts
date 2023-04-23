import gym
import openai

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import re
from math import log, e
from itertools import groupby
from collections import Counter
import os 
from collections import Counter

import sqlite3


from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter



def entropy_of_sequence(input_list):
    # get counts
    count_data = list(Counter(input_list).values())
    
    # get entropy from counts
    entropy = stats.entropy(count_data)  
    
    return entropy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True

def meal_sample(model, env):
    obs = env.reset()
    done, state = False, None
    episode_reward = 0
    while not done:
        action, state = model.predict(obs, state=state, deterministic=False)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    env.render()
    print(f'Epsiode reward: {episode_reward}')
    
def min_reward_meal_sample(model, env, min_reward):
    episode_reward = 0
    while episode_reward < min_reward:
        obs = env.reset()
        done, state = False, None
        episode_reward = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=False)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
    env.render()
    print(f'Epsiode reward: {episode_reward}')

def run_with_learning_algorithm(algorithm, 
                                env,
                                num_timesteps, 
                                log_dir, 
                                num_meals=21,
                                seed=0,
                                print_before_after=True,
                                plot=True):
    env_name = type(env).__name__
    algorithm_name = algorithm.__name__

    env = Monitor(env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model = algorithm('MultiInputPolicy', env, verbose=1, seed=seed)
    
    if print_before_after:
        print('Initial Results')
        meal_sample(model, env)
        print('\n')

    model.learn(total_timesteps=num_timesteps, callback=callback)
    
    n_eval_episodes = 100
    reward_means, _ = evaluate_policy(model=model, env=env, n_eval_episodes=n_eval_episodes, render=False, return_episode_rewards=True, deterministic=False)

    if print_before_after:
        print('Final Results')
        print('Example meal plan that exceeds 75th percentile reward:')
        min_reward_meal_sample(model, env, min_reward=np.quantile(reward_means, 0.75))
        print('\n')

    if plot:
        # Helper from the library
        results_plotter.plot_results(
            [log_dir], 1e5, results_plotter.X_TIMESTEPS, f"{env_name} - {algorithm_name}"
        )
        
        plt.savefig(log_dir + '/rewards_per_episode.png', bbox_inches='tight')

        plot_learning_curve(log_dir, title=f'Learning Curve Smoothed: {env_name} - {algorithm_name}')
        
        plt.figure(dpi=75)
        plt.hist(reward_means)
        plt.title(f'Histogram of episode reward across {n_eval_episodes} episodes')
        plt.savefig(log_dir + '/reward_histogram.png', bbox_inches='tight')
        plt.show()
        
        num_episodes = 1000
        num_meals_to_show = 30
        episode_rewards = []
        meal_counter = Counter()
        nutrition_counter = {nutrition_category: [] for nutrition_category in env.nutrition_data.columns}
        for _ in range(num_episodes):
            obs = env.reset()
            done, state = False, None
            episode_reward = 0
            while not done:
                action, state = model.predict(obs, state=state, deterministic=False)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
            chosen_meals = env.possible_meals[env.meal_history.astype(int)]
            meal_counter.update(chosen_meals)
            
            nutrition_totals = env.nutrition_history.sum(axis=0)
            for i, nutrition_category in enumerate(env.nutrition_data.columns):
                nutrition_counter[nutrition_category].append(nutrition_totals[i])
                

        _, axes = plt.subplots(nrows=2, ncols=7, figsize=(20, 5))
        axes = axes.ravel()
        for i, nutrition_category in enumerate(env.nutrition_data.columns):
            axes[i].hist(nutrition_counter[nutrition_category])
            axes[i].axvline(x=env.lower_goal_nutrition[i], color='red')
            axes[i].axvline(x=env.upper_goal_nutrition[i], color='red')
            axes[i].set_title(f'{nutrition_category}')
        plt.suptitle(f'Nutrition values across {num_episodes} episodes')
        plt.tight_layout()
        plt.savefig(log_dir + '/nutrition_values.png', bbox_inches='tight')
        plt.show()

        meals_chosen, counts = list(zip(*sorted(meal_counter.items(), key=lambda x: x[1], reverse=False)))
        selected_meals = meals_chosen[-num_meals_to_show:]
        selected_counts = counts[-num_meals_to_show:]
        y_pos = np.arange(len(selected_meals))
        plt.figure(dpi=150)
        plt.barh(y=y_pos, width=selected_counts)
        plt.yticks(y_pos, selected_meals, size=5)
        plt.title(f'Top {num_meals_to_show} meals chosen across {num_episodes} episodes')
        plt.savefig(log_dir + '/top_meals.png', bbox_inches='tight')
        plt.show()

        plt.figure(dpi=75)
        plt.hist(counts)
        plt.title(f'Histogram of meal appearances across {num_episodes} epsiodes')
        plt.savefig(log_dir + '/meal_appearances.png', bbox_inches='tight')
        plt.show() 
    
    return model, env

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_learning_curve(log_folder, title="Learning Curve"):
    """
    plot the learning curve

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.savefig(log_folder + '/learning_curve.png', bbox_inches='tight')
    plt.show()


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
        
        # initial reward weights that can also be customized using self.set_reward_weights()
        self.set_reward_weights()
        
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
                dtype=np.float64
            ),
            'category_history': gym.spaces.Box(
                low=0,
                high=self.num_meal_categories,
                shape=(self.num_meals,),
                dtype=np.int64
            ),
            'lower_goal_nutrition': gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(len(nutrition_data.columns),),
                dtype=np.float64
            ),
            'upper_goal_nutrition': gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(len(nutrition_data.columns),),
                dtype=np.float64
            )
        })
        
    def set_reward_weights(
            self, 
            coef_nutrition_lower=1,
            coef_nutrition_upper=-1,
            coef_sequence_entropy=1,
            coef_repetitions=-1,
            coef_overall_entropy=1
        ):
        self.reward_weights = dict(
            coef_nutrition_lower=coef_nutrition_lower,
            coef_nutrition_upper=coef_nutrition_upper,
            coef_sequence_entropy=coef_sequence_entropy,
            coef_repetitions=coef_repetitions,
            coef_overall_entropy=coef_overall_entropy
        )
    
    def _get_lowerbound_goal_nutrition(self):
        # # get minimum and maximum nutrition range from the sample of 500 "really good" dietkit diets
        # # we'll set minimum to be 90% of observed minimum and maximum to be 110% of observed maximum
        # # dividing by plan_length to get values for one meal
        # sample_ingredients = load_ingredient(sample_language = 'eng')
        # sample_menus = load_menu(ingredients = sample_ingredients, sample_language = 'eng')
        # sample_diets = load_diet(menus = sample_menus, num_loads = 500, sample_language = 'eng', sample_name = 'ML')
        # nutrition_range_df = pd.DataFrame([sample_diets.nutrition[i] for i in range(500)]).describe() / sample_diets.plan_length
        # display(nutrition_range_df)
        # nutrition_lower_bound = nutrition_range_df.loc['min', :] * 0.9
        # print(f'>> Lower bound:\n>> {nutrition_lower_bound.values.round(3).tolist()}')
        # nutrition_upper_bound = nutrition_range_df.loc['max', :] * 1.1
        # print(f'>> Upper bound:\n>> {nutrition_upper_bound.values.round(3).tolist()}')
        # # >> Lower bound:
        # # >> [44.007, 1.828, 0.527, 6.922, 0.262, 9.591, 0.295, 48.556, 6.635, 0.028, 0.034, 0.63, 160.102, 17.41]
        # # >> Upper bound:
        # # >> [100.36, 4.721, 3.396, 15.128, 1.481, 64.853, 0.952, 166.827, 67.558, 0.168, 0.161, 10.887, 765.701, 215.002]

        single_meal_lowerbound = np.array([44.007, 1.828, 0.527, 6.922, 0.262, 9.591, 0.295, 48.556, 6.635, 0.028, 0.034, 0.63, 160.102, 17.41])
        return single_meal_lowerbound * self.num_meals
    
    def _get_upperbound_goal_nutrition(self):
        single_meal_upperbound = np.array([100.36, 4.721, 3.396, 15.128, 1.481, 64.853, 0.952, 166.827, 67.558, 0.168, 0.161, 10.887, 765.701, 215.002])
        return single_meal_upperbound * self.num_meals
    
    def _calculate_current_nutrition(self):
        return self.nutrition_history[0:self.current_step, :].sum(axis=0)

    def reset(self):
        self.current_step = 0
        # set all meals to the 'empty' meal
        self.meal_history = (np.zeros(self.num_meals) + self.num_possible_meals - 1).astype(int)
        self.nutrition_history = np.zeros(self.nutrition_history_shape)
        # set all categories to the 'empty' category
        self.category_history = np.zeros(self.num_meals).astype(int)
        self.lower_goal_nutrition = self._get_lowerbound_goal_nutrition()
        self.upper_goal_nutrition = self._get_upperbound_goal_nutrition()
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
            'lower_goal_nutrition': self.lower_goal_nutrition,
            'upper_goal_nutrition': self.upper_goal_nutrition
        }
        return obs

    def _calculate_reward(self):
        # take current percentage of nutrition met, treating each category equally, on [0, 1] scale
        
        # reaches 1 when at minimum nutrition
        # otherwise mean fraction of current nutrition requirement
        current_nutrition = self._calculate_current_nutrition()
        lower_nutrition_fractions = np.divide(current_nutrition, self.lower_goal_nutrition)
        lower_nutrition_fractions = np.array([min(fraction, 1) for fraction in lower_nutrition_fractions])
        lower_nutrition_score = lower_nutrition_fractions.mean()
        
        # reaches 1 when all categories above maximum nutrition
        # otherwise fraction of categories that are above maximum
        upper_nutrition_fractions = np.divide(current_nutrition, self.upper_goal_nutrition)
        upper_nutrition_fractions = np.array([fraction > 1 for fraction in upper_nutrition_fractions])
        upper_nutrition_penalty = upper_nutrition_fractions.mean()
        
        
        # calculate 3 measures of current compositional diversity, all on [0, 1] scale
        # 1. if there are meals repetitions in a sequence of length unique_sequence_length, penalize
        unique_sequence_length = 3
        max_entropy_per_sequence = entropy_of_sequence(range(unique_sequence_length))
        entropy_fractions = []
        for start_index in range(self.num_meals - unique_sequence_length + 1):
            end_index = start_index + unique_sequence_length
            sequence_to_check = self.meal_history[start_index:end_index]
            entropy_fraction = entropy_of_sequence(sequence_to_check) / max_entropy_per_sequence
            entropy_fractions.append(entropy_fraction)
        mean_sequence_entropy_fraction = np.mean(entropy_fractions)
        
        # 2. If there are more than num_allowed_in_a_row, penalize even more
        # count repetitions of each element in order
        # reaches 1 when everything is repeated (so to speak)
        # otherwise fraction of total possible # repetitions
        sequential_counts = [(meal, len(list(appearances))) for meal, appearances in groupby(self.meal_history)]
        max_num_allowed_in_a_row = 1
        total_num_repetitions = np.sum([appearances > max_num_allowed_in_a_row for _, appearances in sequential_counts])
        reptition_penalty = total_num_repetitions / self.num_meals
        
        # 3. Check entropy of meals overall as fraction of max possible, on [0, 1] scale
        overall_entropy = entropy_of_sequence(self.meal_history)
        max_overall_entropy = entropy_of_sequence(list(range(self.num_meals)))
        overall_entropy_fraction = 1 - overall_entropy / max_overall_entropy
        
        # overall reward linear combo of nutrition and composition
        
        reward = float(
            self.reward_weights['coef_nutrition_lower'] * lower_nutrition_score + 
            self.reward_weights['coef_nutrition_upper'] * upper_nutrition_penalty +
            self.reward_weights['coef_sequence_entropy'] * mean_sequence_entropy_fraction + 
            self.reward_weights['coef_repetitions'] * total_num_repetitions + 
            self.reward_weights['coef_overall_entropy'] * overall_entropy_fraction
        ) / self.num_meals
        
        return reward
    
class MaxNutritionEnv(MealPlanningEnv):
    def __init__(self, possible_meals, meal_categories, nutrition_data, num_meals):
        super(MaxNutritionEnv, self).__init__(possible_meals, meal_categories, nutrition_data, num_meals)

    def _calculate_reward(self):
        current_nutrition = self._calculate_current_nutrition()
        reward = current_nutrition.sum()
        return reward

class HeuristicEnv(MealPlanningEnv):
    def __init__(self, possible_meals, meal_categories, nutrition_data, num_meals):
        super(HeuristicEnv, self).__init__(possible_meals, meal_categories, nutrition_data, num_meals)
    
class GPTOnlyEnv(MealPlanningEnv):
    def __init__(self, possible_meals, meal_categories, nutrition_data, num_meals):
        super(GPTOnlyEnv, self).__init__(possible_meals, meal_categories, nutrition_data, num_meals)
        # Accrue meal text for prompting
        self.meal_text_history = ""

        with open('gpt_key.txt', 'r') as file:
            openai.api_key = file.read().rstrip()

        # store previous responses in sqlite database to avoid re-prompting
        # try to load from database, otherwise create new
        self.sql_conn = sqlite3.connect('response_store.db')
        self.sql_conn.execute('CREATE TABLE IF NOT EXISTS response_store (meal text, response text)')
        self.sql_conn.commit()

    def reset(self):
        self.current_step = 0
        # set all meals to the 'empty' meal
        self.meal_history = np.zeros(self.num_meals) + self.num_possible_meals - 1
        self.nutrition_history = np.zeros(self.nutrition_history_shape)
        self.meal_text_history = ""
        self.goal_nutrition = self._get_goal_nutrition()
        return self._next_observation()

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
                    1 is an unhealthy plan and 10 is a perfect, balanced healthy plan, taking into account the given nutritional information.\n',
            'Carbon_impact': 'Please rate the following proposed meal on a scale of -10 to 10, where\
                    -10 is a meal with a very high carbon impact and 10 is a meal with a very low carbon impact,\
                    and 0 is a meal with a neutral carbon impact.\n',
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
        
class RLHFEnv(GPTOnlyEnv):
    def __init__(self, possible_meals, meal_categories, nutrition_data, num_meals):
        super(RLHFEnv, self).__init__(possible_meals, meal_categories, nutrition_data, num_meals)

    
    def _calculate_reward(self, chosen_meal, done=False):
        # TODO: add human input here.
        raise NotImplementedError