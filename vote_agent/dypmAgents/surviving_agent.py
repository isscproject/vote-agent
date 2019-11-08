# (C) Copyright IBM Corp. 2018
from pommerman import constants
from pommerman import characters
from pommerman import utility
import numpy as np
from collections import defaultdict

from dypmAgents.base_agent import MyBaseAgent


class SurvivingAgent(MyBaseAgent):

    def __init__(self, search_range=13):
        self._search_range = search_range
        self._inv_tmp = 10000
        self.random = np.random.RandomState(1)
        super().__init__()        

    def _get_most_survivable_actions(self, n_survivable):

        if len(n_survivable) == 0:

            return list()

        elif len(n_survivable) == 1:

            return list(n_survivable)

        else:

            survivable_actions = [a for a in n_survivable]
            score = [sum([n for n in n_survivable[a]]) for a in survivable_actions]
            max_score = max(score)
            best_indices = np.where(np.array(score) == max_score)
            most_survivable_actions = [survivable_actions[i] for i in best_indices[0]]    
            return most_survivable_actions

    def _get_longest_survivable_actions(self, n_survivable):

        if len(n_survivable) == 0:

            return list()

        elif len(n_survivable) == 1:

            return list(n_survivable)

        else:

            survivable_actions = [a for a in n_survivable]
            score = [sum([n>0 for n in n_survivable[a]]) for a in survivable_actions]         
            max_score = max(score)
            best_indices = np.where(np.array(score) == max_score)
            longest_survivable_actions = [survivable_actions[i] for i in best_indices[0]]
            return longest_survivable_actions

    def act(self, obs, action_space, info):

        #
        # Definitions
        #

        board = info['recently_seen']
        #board = obs['board']
        my_position = obs["position"]  # tuple([x,y]): my position
        my_ammo = obs['ammo']  # int: the number of bombs I have
        my_blast_strength = obs['blast_strength']
        my_kick = obs["can_kick"]  # whether I can kick
        my_enemies = [constants.Item(e) for e in obs['enemies'] if e != constants.Item.AgentDummy]
        if obs["teammate"] != constants.Item.AgentDummy:
            my_teammate = obs["teammate"]
        else:
            my_teammate = None

        all_feasible_actions = [a for a in info["my_next_position"] if info["my_next_position"][a]]

        # positions that might be blocked
        if info["teammate_position"] is None:
            agent_positions = info["enemy_positions"]
        else:
            agent_positions = info["enemy_positions"] + [info["teammate_position"]]

        #
        # Fraction of blocked node in the survival trees of enemies
        #
        
        _list_boards = info["list_boards_no_move"]
        if obs["bomb_blast_strength"][my_position]:
            for b in _list_boards:
                if utility.position_is_agent(b, my_position):
                    b[my_position] = constants.Item.Bomb.value
        else:
            for b in _list_boards:
                if utility.position_is_agent(b, my_position):
                    b[my_position] = constants.Item.Passage.value

        total_frac_blocked, n_survivable_nodes, blocked_time_positions \
            = self._get_frac_blocked(_list_boards, my_enemies, board, obs["bomb_life"],
                                     ignore_dying_agent=False)
        
        if info["teammate_position"] is not None:
            total_frac_blocked_teammate, n_survivable_nodes_teammate, blocked_time_positions_teammate \
                = self._get_frac_blocked(_list_boards, [my_teammate], board, obs["bomb_life"],
                                         ignore_dying_agent=True)

        block = defaultdict(float)
        for action in [constants.Action.Stop,
                       constants.Action.Up, constants.Action.Down,
                       constants.Action.Left, constants.Action.Right]:

            next_position = info["my_next_position"][action]

            if next_position is None:
                continue

            if next_position in info["all_kickable"]:
                # kick will be considered later
                continue

            block[action] = total_frac_blocked[next_position]
            if info["teammate_position"] is not None and block[action] > 0:
                block[action] *= (1 - total_frac_blocked_teammate[next_position])

            if block[action] > 0:
                block[action] *= self._inv_tmp
                block[action] -=  np.log(-np.log(self.random.uniform()))

        if all([my_ammo > 0,
                obs["bomb_blast_strength"][my_position] == 0]):

            list_boards_with_bomb, _ \
                = self._board_sequence(board,
                                       info["curr_bombs"],
                                       info["curr_flames"],
                                       self._search_range,
                                       my_position,
                                       my_blast_strength=my_blast_strength,
                                       my_action=constants.Action.Bomb,
                                       step_to_collapse=info["step_to_collapse"],
                                       collapse_ring=info["collapse_ring"])

            block[constants.Action.Bomb] \
                = self._get_frac_blocked_two_lists(list_boards_with_bomb,
                                                   n_survivable_nodes,
                                                   board,
                                                   my_enemies,
                                                   ignore_dying_agent=False)
            block[constants.Action.Bomb] \
                += total_frac_blocked[my_position] * (1 - block[constants.Action.Bomb])

            if info["teammate_position"] is not None:
                block_teammate_with_bomb = self._get_frac_blocked_two_lists(list_boards_with_bomb,
                                                                            n_survivable_nodes_teammate,
                                                                            board,
                                                                            [my_teammate],
                                                                            ignore_dying_agent=True)
                # this is an approximation
                block_teammate_with_bomb \
                    += total_frac_blocked_teammate[my_position] * (1 - block_teammate_with_bomb)

                block[constants.Action.Bomb] *= (1 - block_teammate_with_bomb)

            if block[constants.Action.Bomb] > 0:
                block[constants.Action.Bomb] *= self._inv_tmp
                block[constants.Action.Bomb] -=  np.log(-np.log(self.random.uniform()))

        block_teammate_with_kick = defaultdict(float)
        for next_position in info["all_kickable"]:

            my_action = self._get_direction(my_position, next_position)

            backedup = False
            if board[next_position] != constants.Item.Bomb.value:
                backup_cell = board[next_position]
                board[next_position] = constants.Item.Bomb.value  # an agent will be overwritten
                backedup = True

            list_boards_with_kick, _ \
                = self._board_sequence(board,
                                       info["curr_bombs"],
                                       info["curr_flames"],
                                       self._search_range,
                                       my_position,
                                       my_action=my_action,
                                       can_kick=True,
                                       step_to_collapse=info["step_to_collapse"],
                                       collapse_ring=info["collapse_ring"])

            if backedup:
                board[next_position] = backup_cell
            
            block[my_action] \
                = self._get_frac_blocked_two_lists(list_boards_with_kick,
                                                   n_survivable_nodes,
                                                   board,
                                                   my_enemies)
            block[my_action] \
                += total_frac_blocked[next_position] * (1 - block[my_action])
            
            if block[my_action] > 0 and info["teammate_position"] is not None:
                block_teammate_with_kick[next_position] \
                    = self._get_frac_blocked_two_lists(list_boards_with_kick,
                                                       n_survivable_nodes_teammate,
                                                       board,
                                                       [my_teammate],
                                                       ignore_dying_agent=True)

                # this is an approximation
                block_teammate_with_kick[next_position] \
                    += total_frac_blocked_teammate[next_position] * (1 - block_teammate_with_kick[next_position])
                
                block[my_action] *= (1 - block_teammate_with_kick[next_position])

            if block[my_action] > 0:
                block[my_action] *= self._inv_tmp
                block[my_action] -=  np.log(-np.log(self.random.uniform()))
            
        n_survivable_move, is_survivable_move, list_boards_move \
            = self._get_survivable(obs, info, my_position, info["my_next_position"], info["enemy_positions"],
                                   info["all_kickable"], allow_kick_to_fog=True,
                                   enemy_mobility=1, enemy_bomb=0,
                                   ignore_dying_agent=False,
                                   step_to_collapse=info["step_to_collapse"],
                                   collapse_ring=info["collapse_ring"])

        for a in all_feasible_actions:
            if a not in n_survivable_move:
                n_survivable_move[a] = np.zeros(self._search_range)
                
        enemy_can_place_bomb = any([obs["bomb_blast_strength"][position] == 0 for position in info["enemy_positions"]])

        if enemy_can_place_bomb:

            n_survivable_bomb, is_survivable_bomb, list_boards_bomb \
                = self._get_survivable(obs, info, my_position, info["my_next_position"], info["enemy_positions"],
                                       info["all_kickable"], allow_kick_to_fog=True,
                                       enemy_mobility=0, enemy_bomb=1,
                                       ignore_dying_agent=False,
                                       step_to_collapse=info["step_to_collapse"],
                                       collapse_ring=info["collapse_ring"])

            for a in all_feasible_actions:
                if a not in n_survivable_bomb:
                    n_survivable_bomb[a] = np.zeros(self._search_range)

            might_survivable_actions = set([a for a in n_survivable_bomb if n_survivable_bomb[a][-1] > 0]
                                           + [a for a in n_survivable_move if n_survivable_move[a][-1] > 0])

            might_survivable_actions -= info["might_block_actions"]
            for a in info["might_block_actions"]:
                n_survivable_bomb[a] = np.zeros(self._search_range)
                n_survivable_move[a] = np.zeros(self._search_range)
            
            for a in might_survivable_actions:
                if a not in n_survivable_bomb:
                    n_survivable_bomb[a] = np.zeros(self._search_range)
                if a not in n_survivable_move:
                    n_survivable_move[a] = np.zeros(self._search_range)
                
            survivable_actions = list()
            for action in might_survivable_actions:
                if n_survivable_move[action][-1] > 0 and n_survivable_bomb[action][-1] > 0:
                    if not info["might_blocked"][action] or n_survivable_move[constants.Action.Stop][-1] > 0:
                        survivable_actions.append(action)

            n_survivable_expected = dict()
            for a in survivable_actions:
                if info["might_blocked"][a]:
                    n_survivable_expected[a] \
                        = np.array(n_survivable_bomb[a]) \
                        + np.array(n_survivable_move[constants.Action.Stop]) \
                        + np.array(n_survivable_move[a])
                    n_survivable_expected[a] = n_survivable_expected[a] / 3
                else:
                    n_survivable_expected[a] = np.array(n_survivable_bomb[a]) + 2 * np.array(n_survivable_move[a])
                    n_survivable_expected[a] = n_survivable_expected[a] / 3
                n_survivable_expected[a] = n_survivable_expected[a]

        else:

            might_survivable_actions = set([a for a in n_survivable_move if n_survivable_move[a][-1] > 0])

            might_survivable_actions -= info["might_block_actions"]
            for a in info["might_block_actions"]:
                n_survivable_move[a] = np.zeros(self._search_range)

            survivable_actions = list()
            for action in might_survivable_actions:
                if n_survivable_move[action][-1] > 0:
                    if not info["might_blocked"][action] or n_survivable_move[constants.Action.Stop][-1] > 0:
                        survivable_actions.append(action)
                        
            for a in might_survivable_actions:
                if a not in n_survivable_move:
                    n_survivable_move[a] = np.zeros(self._search_range)
                
            n_survivable_expected = dict()
            for a in survivable_actions:
                if info["might_blocked"][a]:
                    n_survivable_expected[a] \
                        = np.array(n_survivable_move[constants.Action.Stop]) \
                        + np.array(n_survivable_move[a])
                    n_survivable_expected[a] = n_survivable_expected[a] / 2
                else:
                    n_survivable_expected[a] = np.array(n_survivable_move[a])

        #
        # Choose actions
        #
        
        if len(survivable_actions) == 1:

            action = survivable_actions.pop()
            return action.value

        if len(survivable_actions) > 1:
            
            most_survivable_actions = self._get_most_survivable_actions(n_survivable_expected)

            if len(most_survivable_actions) == 1:

                return most_survivable_actions[0].value

            elif len(most_survivable_actions) > 1:
                
                # tie break by block score
                max_block = 0  # do not choose 0
                best_action = None
                for action in all_feasible_actions:
                    if action not in most_survivable_actions:
                        # for deterministic behavior
                        continue 
                    if info["might_block_teammate"][action]:
                        continue
                    if block[action] > max_block:
                        max_block = block[action]
                        best_action = action
                if best_action is not None:
                    return best_action.value

        #
        # no survivable actions for all cases
        #

        if enemy_can_place_bomb:
            
            n_survivable_expected = dict()
            for a in all_feasible_actions:
                if info["might_blocked"][a]:
                    if is_survivable_move[constants.Action.Stop]:
                        n_survivable_expected[a] \
                            = np.array(n_survivable_bomb[a]) \
                            + np.array(n_survivable_move[constants.Action.Stop]) \
                            + np.array(n_survivable_move[a])
                    else:
                        n_survivable_expected[a] \
                            = np.array(n_survivable_bomb[a]) \
                            + np.array(n_survivable_move[a])
                    n_survivable_expected[a] = n_survivable_expected[a] / 3
                else:
                    n_survivable_expected[a] = np.array(n_survivable_bomb[a]) + 2 * np.array(n_survivable_move[a])
                    n_survivable_expected[a] = n_survivable_expected[a] / 3

        else:

            n_survivable_expected = dict()
            for a in all_feasible_actions:
                if info["my_next_position"][a] is None:
                    continue
                if info["might_blocked"][a]:
                    if is_survivable_move[constants.Action.Stop]:
                        n_survivable_expected[a] \
                            = np.array(n_survivable_move[constants.Action.Stop]) \
                            + np.array(n_survivable_move[a])
                    else:
                        n_survivable_expected[a] = np.array(n_survivable_move[a])
                    n_survivable_expected[a] = n_survivable_expected[a] / 2
                else:
                    n_survivable_expected[a] = np.array(n_survivable_move[a])

        if len(might_survivable_actions) == 1:

            action = might_survivable_actions.pop()
            return action.value

        if len(might_survivable_actions) > 1:
            
            most_survivable_actions = self._get_most_survivable_actions(n_survivable_expected)

            if len(most_survivable_actions) == 1:

                return most_survivable_actions[0].value

            elif len(most_survivable_actions) > 1:
                
                # tie break by block score
                max_block = 0  # do not choose 0
                best_action = None
                for action in all_feasible_actions:
                    if action not in most_survivable_actions:
                        # for deterministic behavior
                        continue
                    if info["might_block_teammate"][action]:
                        continue
                    if block[action] > max_block:
                        max_block = block[action]
                        best_action = action

                if best_action is not None:
                    return best_action.value
            
        # no survivable action found for any cases
        # TODO : Then consider killing enemies or helping teammate
        
        max_block = 0  # do not choose 0
        best_action = None
        for action in all_feasible_actions:
            if action not in block:
                # for deterministic behavior
                continue
            if info["might_block_teammate"][action]:
                continue
            if all([action==constants.Action.Bomb,
                    info["teammate_position"] is not None]):
                if block_teammate_with_bomb > 0:
                    continue
            next_position = info["my_next_position"][action]
            if all([next_position in info["all_kickable"],
                    block_teammate_with_kick[next_position] > 0]):
                continue
            if block[action] > max_block:
                max_block = block[action]
                best_action = action

        if best_action is not None:
            return best_action.value

        # longest survivable action

        longest_survivable_actions = self._get_longest_survivable_actions(n_survivable_expected)

        if len(longest_survivable_actions) == 1:

            return longest_survivable_actions[0].value

        elif len(longest_survivable_actions) > 1:

            # break tie by most survivable actions
            for a in n_survivable_expected:
                if a not in longest_survivable_actions:
                    n_survivable_expected[a] = np.zeros(self._search_range)
            most_survivable_actions = self._get_most_survivable_actions(n_survivable_expected)

            if len(most_survivable_actions) == 1:

                return most_survivable_actions[0].value

            elif len(most_survivable_actions) > 1:

                if info["teammate_position"] is not None:
                    min_block = np.inf
                    best_action = None
                    for a in all_feasible_actions:
                        if a not in most_survivable_actions:
                            # for deterministic behavior
                            continue
                        if a == constants.Action.Bomb:
                            score = block_teammate_with_bomb  # do not choose Bomb unless it is strictly better than others
                        else:
                            next_position = info["my_next_position"][a]
                            if next_position in info["all_kickable"]:
                                score = block_teammate_with_kick[next_position] - self.random.uniform(0, 1e-6)
                            else:
                                score = total_frac_blocked_teammate[next_position] - self.random.uniform(0, 1e-6)
                        if score < min_block:
                            min_block = score
                            best_action = a
                    if best_action is not None:
                        return best_action.value
                else:
                    # remove Bomb (as it is most affected by bugs)                    
                    #most_survivable_actions = list(set(most_survivable_actions) - {constants.Action.Bomb})
                    most_survivable_actions = [a for a in all_feasible_actions
                                               if a in most_survivable_actions and a != constants.Action.Bomb]

                    index = self.random.randint(len(most_survivable_actions))
                    random_action = most_survivable_actions[index]
                    return random_action.value
        
        
        # The following will not be used

        self.random.shuffle(all_feasible_actions)
        if len(all_feasible_actions):
            action = all_feasible_actions[0]
            return action.value

        action = constants.Action.Stop
        return action.value
