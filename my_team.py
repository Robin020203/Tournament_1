# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='HybridReflexAgentSwitch', second='HybridReflexAgentDefence', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


class HybridReflexAgentSwitch(ReflexCaptureAgent):
    def __init__(self, index):
        super().__init__(index)
        self.threshold = 5
        self.food = 0
        self.already_counted_food = set()
        self.is_defensive = False

    def food_on_enemy_side(self, food_position, game_state):
        """
        Check of food op de vijandelijke helft is.
        """
        mid_x = game_state.data.layout.width // 2
        is_enemy_side = (food_position[0] >= mid_x) if self.red else (food_position[0] < mid_x)
        return is_enemy_side


    def pacman_on_own_side(self, game_state):
        """
        Check of de agent zich op zijn eigen helft bevindt.
        """
        mid_x = game_state.data.layout.width // 2
        my_pos = game_state.get_agent_position(self.index)

        if self.red:
            return my_pos[0] < mid_x
        else:
            return my_pos[0] >= mid_x


    def update_food(self, game_state):
        """
        Update self.food
        """
        current_food_list = self.get_food(game_state).as_list()  # Huidige voedselpositie
        prev_state = self.get_previous_observation()  # Vorige state
        current_pos = game_state.get_agent_position(self.index)  # Huidige positie


        if current_pos == self.start or self.pacman_on_own_side(game_state):
            self.food = 0

        #if not my_state.is_pacman and self.is_defensive:
        #    self.food = 0


        if prev_state:
            prev_food_list = self.get_food(prev_state).as_list()  # Vorige foodlist
            pacman_prev_pos = prev_state.get_agent_position(self.index)  # Vorige positie

            #if 'already_counted_food' not in self.__dict__:
            #    self.already_counted_food = set()

            eaten_food = []  # geteld voedsel
            for food in prev_food_list:
                if food not in current_food_list and food not in self.already_counted_food:  # Als voedsel nu weg is en nog niet geteld is
                    eaten_food.append(food)

            for food in eaten_food:
                if self.get_maze_distance(pacman_prev_pos, food) <= 1 and self.food_on_enemy_side(food, game_state):
                    if food not in self.already_counted_food:  # Alleen optellen als het nog niet geteld is
                        self.food += 1
                        self.already_counted_food.add(food)


            # +1 als lengte food veranderd
            #food_collected = len(prev_food_list) - len(current_food_list)
            #if food_collected > 0:
            #    self.food += 1

            #if self.food < 0:
            #    self.food = 0

        print(f"Agent {self.index} heeft {self.food} voedsel verzameld")  # Debugging



    def get_features(self, game_state, action):
        self.update_food(game_state)  # Update voedselstatus voordat we beslissingen maken
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()


        # DEFENSIVE features
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        def_capsules = self.get_capsules_you_are_defending(game_state)
        if def_capsules:
            min_distance_to_capsule = min(self.get_maze_distance(my_pos, capsule) for capsule in def_capsules)
            features['distance_to_capsule'] = min_distance_to_capsule
        else:
            features['distance_to_capsule'] = 0


        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # OFFENSIVE features

        features['successor_score'] = -len(food_list)

        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        off_capsules = self.get_capsules(game_state)
        if off_capsules:
            min_distance_to_capsule = min(self.get_maze_distance(my_pos, capsule) for capsule in off_capsules)
            features['distance_to_enemy_capsule'] = min_distance_to_capsule
        else:
            features['distance_to_enemy_capsule'] = 0

        # GHOST DISTANCE
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        #features['num_defenders'] = len(defenders)
        if len(defenders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
            mindist = min(dists)
            if mindist < 3:
                features['ghost_really_close'] = mindist
            else:
                features['ghost_close'] = mindist

        # ENEMY SCARED TIMER
        opponents = self.get_opponents(game_state)
        max_scared_time = 0
        closest_scared_ghost_dist = float('inf')
        for opponent in opponents:
            ghost_state = game_state.get_agent_state(opponent)
            if not ghost_state.is_pacman: #spook
                scared_timer = ghost_state.scared_timer
                max_scared_time = max(max_scared_time, scared_timer)
                if scared_timer > 5:
                    ghost_pos = ghost_state.get_position()
                    if ghost_pos:
                        dist = self.get_maze_distance(my_pos, ghost_pos)
                        closest_scared_ghost_dist = min(closest_scared_ghost_dist, dist)
                        features['distance_to_scared_ghost'] = closest_scared_ghost_dist
                else:
                    features['distance_to_scared_ghost'] = 0
        features['scared_ghost_time'] = max_scared_time

        # OUR SCARED TIMER
        my_scared_timer = my_state.scared_timer
        features['is_scared'] = 1 if my_scared_timer > 0 else 0

        if my_scared_timer > 0:
            closest_pacman_dist = float('inf')
            for invader in invaders:
                dist = self.get_maze_distance(my_pos, invader.get_position())
                closest_pacman_dist = min(closest_pacman_dist, dist)
            features['distance_to_attacking_pacman'] = closest_pacman_dist
        else:
            features['distance_to_attacking_pacman'] = 0


        # DISTANCE TO HOME
        mid_x = game_state.data.layout.width // 2
        if self.red:
            target_range = range(mid_x - 2, mid_x)  # Left side 2 columns
        else:
            target_range = range(mid_x, mid_x + 2)  # Right side 2 columns

        best_distance = float('inf')

        # All possible (x, y) positions on our side
        for x in target_range:
            for y in range(game_state.data.layout.height):
                if not game_state.has_wall(x, y):  # Controleer of er geen muur is
                    distance = self.get_maze_distance(my_pos, (x, y))
                    if distance < best_distance:
                        best_distance = distance

        if self.food > 1:
            features['distance_to_home'] = best_distance
        else:
            features['distance_to_home'] = 0
        # features['distance_to_home'] = best_distance if self.food > 1 else 0

        # ENEMY FOOD
        enemies = []
        for i in self.get_opponents(game_state):
            enemy_state = game_state.get_agent_state(i)
            enemies.append(enemy_state)

        distance_enemy_most_food = 0
        most_enemy_food = 0

        for enemy in enemies:
            if enemy.is_pacman and enemy.get_position() is not None:
                if enemy.num_carrying > most_enemy_food:
                    most_enemy_food = enemy.num_carrying
                    distance_enemy_most_food = self.get_maze_distance(my_pos, enemy.get_position())

        if most_enemy_food >= 6:
            features['distance_enemy_most_food'] = distance_enemy_most_food * 20  # super high prio
        if most_enemy_food >= 5:
            features['distance_enemy_most_food'] = distance_enemy_most_food * 10  # high prio
        elif most_enemy_food >= 3:
            features['distance_enemy_most_food'] = distance_enemy_most_food * 5  # mid prio
        elif most_enemy_food >= 1:
            features['distance_enemy_most_food'] = distance_enemy_most_food * 1  # low prio
        else:
            features['distance_enemy_most_food'] = 0  # no prio


        # STRATEGY
        enemies_on_their_side = sum(1 for enemy in enemies if not enemy.is_pacman)
        enemies_on_our_side = sum(1 for enemy in enemies if enemy.is_pacman and enemy.get_position() is not None)

        enemies_far_from_mid = True
        for enemy in enemies:
            if enemy.get_position() is not None:
                enemy_x, enemy_y = enemy.get_position()
                distance_to_mid = abs(mid_x - enemy_x)
                if distance_to_mid <= 5:
                    enemies_far_from_mid = False
                    break

        if most_enemy_food >= 5:
            self.is_defensive = True
            #print(f"️‼️ Agent {self.index} ONMIDDELIJK defensive!‼️")  # DEBUGGING

        if self.food > 2:
            self.is_defensive = True
            #print(f"⚠️ Agent {self.index} gaat defensive! self.food = {self.food}") # DEBUGGING

        elif enemies_on_our_side >= 1 and self.get_score(game_state) >= self.threshold:
            self.is_defensive = True

        elif enemies_on_their_side == len(enemies) and self.food == 0 and enemies_far_from_mid:
            self.is_defensive = False
            #print(f"⚔️️ Agent {self.index} OPNIEUW offensive")  # DEBUGGING

        elif self.get_score(game_state) >= self.threshold:
            self.is_defensive = True
            #print(f"⚠️⚠️ Agent {self.index} defensive! self.treshold = {self.threshold}")  # DEBUGGING

        else:
            self.is_defensive = False
            #print(f"⚔️️ Agent {self.index} offensive")  # DEBUGGING


        return features

    def get_weights(self, game_state, action):
        if self.is_defensive:
            #defensive weights
            weights = {'num_invaders': -1000,
                        'distance_to_capsule': -5,
                        'on_defense': 100,
                        'invader_distance': -10,
                        'stop': -10,
                        'reverse': -2,
                        'scared_ghost_time': 5,
                        'distance_to_scared_ghost': -100,
                        'is_scared': -200,
                        'distance_to_attacking_pacman': 50,
                        'distance_to_home': -50}

            my_state = game_state.get_agent_state(self.index)
            if my_state.scared_timer == 0:
                weights['distance_enemy_most_food'] = -300
            else:
                weights['distance_enemy_most_food'] = 0


            return weights


        else:
            #offensive weights
            return {'successor_score': 100,
                    'distance_to_food': -1,
                    'ghost_really_close': 1000,
                    'ghost_close': 50,
                    'stop': -100,
                    'scared_ghost_time': 5,
                    'distance_to_scared_ghost': -100,
                    'is_scared': -200,
                    'distance_to_attacking_pacman': 50,
                    'distance_to_home': -50 * self.food,
                    'distance_to_enemy_capsule': -50}



class HybridReflexAgentDefence(ReflexCaptureAgent):
    def __init__(self, index):
        super().__init__(index)
        self.threshold = 5
        self.food = 0
        self.already_counted_food = set()
        self.is_defensive = False

    def food_on_enemy_side(self, food_position, game_state):
        """
        Check of food op de vijandelijke helft is.
        """
        mid_x = game_state.data.layout.width // 2
        is_enemy_side = (food_position[0] >= mid_x) if self.red else (food_position[0] < mid_x)
        return is_enemy_side


    def pacman_on_own_side(self, game_state):
        """
        Check of de agent zich op zijn eigen helft bevindt.
        """
        mid_x = game_state.data.layout.width // 2
        my_pos = game_state.get_agent_position(self.index)

        if self.red:
            return my_pos[0] < mid_x
        else:
            return my_pos[0] >= mid_x


    def update_food(self, game_state):
        """
        Update self.food
        """
        current_food_list = self.get_food(game_state).as_list()  # Huidige voedselpositie
        prev_state = self.get_previous_observation()  # Vorige state
        current_pos = game_state.get_agent_position(self.index)  # Huidige positie


        if current_pos == self.start or self.pacman_on_own_side(game_state):
            self.food = 0

        #if not my_state.is_pacman and self.is_defensive:
        #    self.food = 0


        if prev_state:
            prev_food_list = self.get_food(prev_state).as_list()  # Vorige foodlist
            pacman_prev_pos = prev_state.get_agent_position(self.index)  # Vorige positie

            #if 'already_counted_food' not in self.__dict__:
            #    self.already_counted_food = set()

            eaten_food = []  # geteld voedsel
            for food in prev_food_list:
                if food not in current_food_list and food not in self.already_counted_food:  # Als voedsel nu weg is en nog niet geteld is
                    eaten_food.append(food)

            for food in eaten_food:
                if self.get_maze_distance(pacman_prev_pos, food) <= 1 and self.food_on_enemy_side(food, game_state):
                    if food not in self.already_counted_food:  # Alleen optellen als het nog niet geteld is
                        self.food += 1
                        self.already_counted_food.add(food)


            # +1 als lengte food veranderd
            #food_collected = len(prev_food_list) - len(current_food_list)
            #if food_collected > 0:
            #    self.food += 1

            #if self.food < 0:
            #    self.food = 0

        print(f"Agent {self.index} heeft {self.food} voedsel verzameld")  # Debugging



    def get_features(self, game_state, action):
        self.update_food(game_state)  # Update voedselstatus voordat we beslissingen maken
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()


        # DEFENSIVE features
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        capsules = self.get_capsules_you_are_defending(game_state)
        if capsules:
            min_distance_to_capsule = min(self.get_maze_distance(my_pos, capsule) for capsule in capsules)
            features['distance_to_capsule'] = min_distance_to_capsule
        else:
            features['distance_to_capsule'] = 0


        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # OFFENSIVE features

        features['successor_score'] = -len(food_list)

        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        off_capsules = self.get_capsules(game_state)
        if off_capsules:
            min_distance_to_capsule = min(self.get_maze_distance(my_pos, capsule) for capsule in off_capsules)
            features['distance_to_enemy_capsule'] = min_distance_to_capsule
        else:
            features['distance_to_enemy_capsule'] = 0

        # GHOST DISTANCE
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        #features['num_defenders'] = len(defenders)
        if len(defenders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
            mindist = min(dists)
            if mindist < 3:
                features['ghost_really_close'] = mindist
            else:
                features['ghost_close'] = mindist

        # ENEMY SCARED TIMER
        opponents = self.get_opponents(game_state)
        max_scared_time = 0
        closest_scared_ghost_dist = float('inf')
        for opponent in opponents:
            ghost_state = game_state.get_agent_state(opponent)
            if not ghost_state.is_pacman: #spook
                scared_timer = ghost_state.scared_timer
                max_scared_time = max(max_scared_time, scared_timer)
                if scared_timer > 5:
                    ghost_pos = ghost_state.get_position()
                    if ghost_pos:
                        dist = self.get_maze_distance(my_pos, ghost_pos)
                        closest_scared_ghost_dist = min(closest_scared_ghost_dist, dist)
                        features['distance_to_scared_ghost'] = closest_scared_ghost_dist
                else:
                    features['distance_to_scared_ghost'] = 0
        features['scared_ghost_time'] = max_scared_time

        # OUR SCARED TIMER
        my_scared_timer = my_state.scared_timer
        features['is_scared'] = 1 if my_scared_timer > 0 else 0

        if my_scared_timer > 0:
            closest_pacman_dist = float('inf')
            for invader in invaders:
                dist = self.get_maze_distance(my_pos, invader.get_position())
                closest_pacman_dist = min(closest_pacman_dist, dist)
            features['distance_to_attacking_pacman'] = closest_pacman_dist
        else:
            features['distance_to_attacking_pacman'] = 0


        # DISTANCE TO HOME
        mid_x = game_state.data.layout.width // 2
        if self.red:
            target_range = range(mid_x - 2, mid_x)  # Left side 2 columns
        else:
            target_range = range(mid_x, mid_x + 2)  # Right side 2 columns

        best_distance = float('inf')

        # All possible (x, y) positions on our side
        for x in target_range:
            for y in range(game_state.data.layout.height):
                if not game_state.has_wall(x, y):  # Controleer of er geen muur is
                    distance = self.get_maze_distance(my_pos, (x, y))
                    if distance < best_distance:
                        best_distance = distance

        if self.food > 1:
            features['distance_to_home'] = best_distance
        else:
            features['distance_to_home'] = 0
        # features['distance_to_home'] = best_distance if self.food > 1 else 0

        # ENEMY FOOD
        enemies = []
        for i in self.get_opponents(game_state):
            enemy_state = game_state.get_agent_state(i)
            enemies.append(enemy_state)

        distance_enemy_most_food = 0
        most_enemy_food = 0

        for enemy in enemies:
            if enemy.is_pacman and enemy.get_position() is not None:
                if enemy.num_carrying > most_enemy_food:
                    most_enemy_food = enemy.num_carrying
                    distance_enemy_most_food = self.get_maze_distance(my_pos, enemy.get_position())

        if most_enemy_food >= 6:
            features['distance_enemy_most_food'] = distance_enemy_most_food * 20  # super high prio
        if most_enemy_food >= 5:
            features['distance_enemy_most_food'] = distance_enemy_most_food * 10  # high prio
        elif most_enemy_food >= 3:
            features['distance_enemy_most_food'] = distance_enemy_most_food * 5  # mid prio
            self.is_defensive = True
        elif most_enemy_food >= 1:
            features['distance_enemy_most_food'] = distance_enemy_most_food * 3  # low prio
        else:
            features['distance_enemy_most_food'] = 0  # no prio


        # STRATEGY
        enemies_on_their_side = sum(1 for enemy in enemies if not enemy.is_pacman)

        if most_enemy_food >= 5:
            self.is_defensive = True
            #print(f"️‼️ Agent {self.index} ONMIDDELIJK defensive!‼️")  # DEBUGGING

        if self.food > 2:
            self.is_defensive = True
            #print(f"⚠️ Agent {self.index} gaat defensive! self.food = {self.food}") # DEBUGGING

        elif self.get_score(game_state) >= self.threshold:
            self.is_defensive = True
            #print(f"⚠️⚠️ Agent {self.index} defensive! self.treshold = {self.threshold}")  # DEBUGGING

        else:
            self.is_defensive = False
            #print(f"⚔️️ Agent {self.index} offensive")  # DEBUGGING


        return features

    def get_weights(self, game_state, action):
        if self.is_defensive:
            #defensive weights
            weights = {'num_invaders': -1000,
                        'distance_to_capsule': -50,
                        'on_defense': 100,
                        'invader_distance': -10,
                        'stop': -10,
                        'reverse': -2,
                        'scared_ghost_time': 5,
                        'distance_to_scared_ghost': -100,
                        'is_scared': -200,
                        'distance_to_attacking_pacman': 50,
                        'distance_to_home': -50}

            my_state = game_state.get_agent_state(self.index)
            if my_state.scared_timer == 0:
                weights['distance_enemy_most_food'] = -300
            else:
                weights['distance_enemy_most_food'] = 0

            return weights

        else:
            #offensive weights
            return {'successor_score': 100,
                    'distance_to_food': -1,
                    'ghost_really_close': 1000,
                    'ghost_close': 50,
                    'stop': -100,
                    'scared_ghost_time': 5,
                    'distance_to_scared_ghost': -100,
                    'is_scared': -200,
                    'distance_to_attacking_pacman': 50,
                    'distance_to_home': -50 * self.food,
                    'distance_to_enemy_capsule': -50}