# (C) Copyright IBM Corp. 2018
from pommerman import constants
from pommerman import characters
from pommerman import utility
import numpy as np
from collections import defaultdict

from dypmAgents.base_agent import MyBaseAgent
from dypmAgents.generic_agent import GenericAgent
from dypmAgents.tools import search_time_expanded_network


class BattleAgent(GenericAgent):
    
    def act(self, obs, action_space, info):

        #
        # Definitions
        #

        #board = obs['board']
        board = info["recently_seen"]
        my_position = obs["position"]  # tuple([x,y]): my position
        my_ammo = obs['ammo']  # int: the number of bombs I have
        my_blast_strength = obs['blast_strength']
        my_enemies = [constants.Item(e) for e in obs['enemies'] if e != constants.Item.AgentDummy]
        if obs["teammate"] != constants.Item.AgentDummy:
            my_teammate = obs["teammate"]
        else:
            my_teammate = None
        my_kick = obs["can_kick"]  # whether I can kick

        #
        # Understand current situation
        #

        # positions that might be blocked
        if info["teammate_position"] is None:
            agent_positions = info["enemy_positions"]
        else:
            agent_positions = info["enemy_positions"] + [info["teammate_position"]]

        # survivable actions
        
        if len(info["enemy_positions"]) > 0: 
            mobility = self._enemy_mobility 
        else: 
            mobility = 0 

        n_survivable, is_survivable, list_boards \
            = self._get_survivable(obs, info, my_position, info["my_next_position"], agent_positions,
                                   info["all_kickable"], allow_kick_to_fog=False,
                                   enemy_mobility=mobility, enemy_bomb=self._enemy_bomb,
                                   step_to_collapse=info["step_to_collapse"],
                                   collapse_ring=info["collapse_ring"])

        for a in info["might_block_actions"]:
            n_survivable[a] = np.zeros(self._search_range)
            is_survivable[a] = False
        
        survivable_actions = list()
        for a in is_survivable:
            if not is_survivable[a]:
                continue
            if info["might_blocked"][a] and not is_survivable[constants.Action.Stop]:
                continue
            if n_survivable[a][-1] <= 1:
                is_survivable[a] = False
                continue
            survivable_actions.append(a)

        #
        # Choose action
        #
                
        if len(survivable_actions) == 0:

            #
            # return None, if no survivable actions
            #
        
            return None

        elif len(survivable_actions) == 1:

            #
            # Choose the survivable action, if it is the only choice
            #
            
            action = survivable_actions[0]
            return action.value

        if all([info["prev_action"] not in [constants.Action.Stop, constants.Action.Bomb],
                info["prev_position"] == my_position]):
            # if previously blocked, do not reapeat with some probability
            self._inv_tmp *= self._backoff
        else:
            self._inv_tmp = self._inv_tmp_init

        #
        # Bomb at a target
        #

        # fraction of blocked node in the survival trees of enemies
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
            = self._get_frac_blocked(_list_boards, my_enemies, board, obs["bomb_life"])
    
        if info["teammate_position"] is not None:
            total_frac_blocked_teammate, n_survivable_nodes_teammate, blocked_time_positions_teammate \
                = self._get_frac_blocked(_list_boards, [my_teammate], board, obs["bomb_life"])

            if n_survivable_nodes_teammate[my_teammate] > 0:
                LB = self._teammate_survivability_threshold / n_survivable_nodes_teammate[my_teammate]
                positions_teammate_safe = np.where(total_frac_blocked_teammate < LB)
                total_frac_blocked_teammate[positions_teammate_safe] = 0

        p_survivable = defaultdict(float)
        for action in n_survivable:
            p_survivable[action] = sum(n_survivable[action]) / self._my_survivability_threshold
            if p_survivable[action] > 1:
                p_survivable[action] = 1

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
            if all([utility.position_is_flames(board, next_position),
                    info["flame_life"][next_position] > 1,
                    is_survivable[constants.Action.Stop]]):
                # if the next position is flames,
                # I want to stop to wait, which must be feasible
                block[action] = total_frac_blocked[next_position] * p_survivable[constants.Action.Stop]
                if info["teammate_position"] is not None:
                    block[action] *= (1 - total_frac_blocked_teammate[next_position])
                if block[action] > 0:
                    block[action] *= self._inv_tmp
                    block[action] -=  np.log(-np.log(self.random.uniform()))
                continue
            elif not is_survivable[action]:
                continue
            if all([info["might_blocked"][action],
                    not is_survivable[constants.Action.Stop]]):
                continue

            block[action] = total_frac_blocked[next_position] * p_survivable[action]
            if info["teammate_position"] is not None:
                block[action] *= (1 - total_frac_blocked_teammate[next_position])
            if block[action] > 0:
                block[action] *= self._inv_tmp
                block[action] -=  np.log(-np.log(self.random.uniform()))

            if info["might_blocked"][action]:
                block[action] = (total_frac_blocked[my_position] * p_survivable[constants.Action.Stop]
                                 + total_frac_blocked[next_position] * p_survivable[action]) / 2
                if info["teammate_position"] is not None:                    
                    block[action] *= (1 - total_frac_blocked_teammate[next_position])
                if block[action] > 0:
                    block[action] *= self._inv_tmp
                    block[action] -=  np.log(-np.log(self.random.uniform()))

        if is_survivable[constants.Action.Bomb]:
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

            n_survivable_nodes_with_bomb = defaultdict(int)
            for enemy_position in info["enemy_positions"]:
                # get survivable tree of the enemy
                _survivable = search_time_expanded_network(list_boards_with_bomb,
                                                           enemy_position)
                n_survivable_nodes_with_bomb[enemy_position] = sum([len(positions) for positions in _survivable])

            n_with_bomb = sum([n_survivable_nodes_with_bomb[enemy_position]
                               for enemy_position in info["enemy_positions"]])
            n_with_none = sum([n_survivable_nodes[enemy] for enemy in my_enemies])
            if n_with_none == 0:
                total_frac_blocked_with_bomb = 0

                # place more bombs, so the stacked enemy cannot kick
                x, y = my_position
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    next_position = (x + dx, y + dy)
                    following_position = (x + 2 * dx, y + 2 * dy)
                    if not self._on_board(following_position):
                        continue
                    if all([obs["bomb_life"][next_position] > 0,
                            board[following_position] > constants.Item.AgentDummy.value]):
                        total_frac_blocked_with_bomb = 1
            else:
                total_frac_blocked_with_bomb = 1 - n_with_bomb / n_with_none

            action = constants.Action.Bomb
            block[action] = total_frac_blocked_with_bomb
            # block[action] += total_frac_blocked[my_position] * (eisenachAgents - total_frac_blocked_with_bomb)
            block[action] *= p_survivable[action]

            block_teammate_with_bomb = None
            if block[action] > 0:
                if info["teammate_position"] is not None:
                    block_teammate_with_bomb \
                        = self._get_frac_blocked_two_lists(list_boards_with_bomb,
                                                           n_survivable_nodes_teammate,
                                                           board,
                                                           [my_teammate],
                                                           ignore_dying_agent=True)
                
                    block_teammate_with_bomb \
                        += total_frac_blocked_teammate[my_position] * (1 - block_teammate_with_bomb)
                    block[action] *= (1 - block_teammate_with_bomb)

                block[action] *= self._inv_tmp
                block[action] -=  np.log(-np.log(self.random.uniform()))

        for next_position in info["kickable"]:

            action = self._get_direction(my_position, next_position)
            if not is_survivable[action]:
                continue

            list_boards_with_kick, _ \
                = self._board_sequence(board,
                                       info["curr_bombs"],
                                       info["curr_flames"],
                                       self._search_range,
                                       my_position,
                                       my_action=action,
                                       can_kick=True,
                                       step_to_collapse=info["step_to_collapse"],
                                       collapse_ring=info["collapse_ring"])

            block[action] \
                = self._get_frac_blocked_two_lists(list_boards_with_kick,
                                                   n_survivable_nodes,
                                                   board,
                                                   my_enemies,
                                                   ignore_dying_agent=True)
            block[action] += total_frac_blocked[next_position] * (1 - block[action])
            block[action] *= p_survivable[action]

            if block[action] > 0:
                if info["teammate_position"] is not None:
                    block_teammate_with_kick \
                        = self._get_frac_blocked_two_lists(list_boards_with_kick,
                                                           n_survivable_nodes_teammate,
                                                           board, [my_teammate],
                                                           ignore_dying_agent=True)
                    block_teammate_with_kick \
                        += total_frac_blocked_teammate[next_position] * (1 - block_teammate_with_kick)
                    block[action] *= (1 - block_teammate_with_kick)
                    
                block[action] *= self._inv_tmp
                block[action] -=  np.log(-np.log(self.random.uniform()))

        max_block = 0  # do not choose zero blocking action as the best
        best_action = None
        for action in block:
            if block[action] > max_block:
                max_block = block[action]
                best_action = action

        if block[constants.Action.Bomb] > self._bomb_threshold * self._inv_tmp:
            if info["teammate_position"] is not None:
                if block_teammate_with_bomb is None:
                    block_teammate_with_bomb \
                        = self._get_frac_blocked_two_lists(list_boards_with_bomb,
                                                           n_survivable_nodes_teammate,
                                                           board,
                                                           [my_teammate],
                                                           ignore_dying_agent=True)

                teammate_safety = block_teammate_with_bomb * n_survivable_nodes_teammate[my_teammate]
                if any([teammate_safety > self._teammate_survivability_threshold,
                        block_teammate_with_bomb < self._interfere_threshold,
                        block_teammate_with_bomb < total_frac_blocked_teammate[my_position]]):
                    teammate_ok = True
                else:
                    teammate_ok = False
            else:
                teammate_ok = True

            if teammate_ok:
                if best_action == constants.Action.Bomb:                    
                    return constants.Action.Bomb.value

                if best_action == constants.Action.Stop:
                    return constants.Action.Bomb.value

        #
        # Move towards where to bomb
        #

        if best_action not in [None, constants.Action.Bomb]:
            next_position = info["my_next_position"][best_action]

            should_chase = (total_frac_blocked[next_position] > self._chase_threshold)

            if info["teammate_position"] is not None:
                teammate_safety = total_frac_blocked_teammate[next_position] * n_survivable_nodes_teammate[my_teammate]
                if any([teammate_safety > self._teammate_survivability_threshold,
                        total_frac_blocked_teammate[next_position] < self._interfere_threshold,
                        total_frac_blocked_teammate[next_position] < total_frac_blocked_teammate[my_position]]):
                    teammate_ok = True
                else:
                    teammate_ok = False
            else:
                teammate_ok = True

            if should_chase and teammate_ok:
                if all([utility.position_is_flames(board, next_position),
                        info["flame_life"][next_position] > 1,
                        is_survivable[constants.Action.Stop]]):
                    action = constants.Action.Stop
                    return action.value
                else:
                    return best_action.value                

        # Exclude the action representing stop to wait
        max_block = 0  # do not choose zero blocking action as the best
        best_action = None
        for action in survivable_actions:
            if block[action] > max_block:
                max_block = block[action]
                best_action = action
                
        #
        # Do not take risky actions when not interacting with enemies
        #

        most_survivable_action = self._action_most_survivable(n_survivable)

        if total_frac_blocked[my_position] > 0:
            # ignore actions with low survivability
            _survivable_actions = list()
            for action in n_survivable:
                n = sum(n_survivable[action])
                if not is_survivable[action]:
                    continue
                elif n > self._my_survivability_threshold:
                    _survivable_actions.append(action)
                else:
                    is_survivable[action] = False

            if len(_survivable_actions) > 1:
                survivable_actions = _survivable_actions
            elif best_action is not None:
                return best_action.value
            else:
                # Take the most survivable action
                return most_survivable_action.value

        #
        # Do not interfere with teammate
        #

        if all([info["teammate_position"] is not None,
                len(info["enemy_positions"]) > 0 or len(info["curr_bombs"]) > 0]):
            # ignore actions that interfere with teammate
            min_interfere = np.inf
            least_interfere_action = None
            _survivable_actions = list()
            for action in survivable_actions:
                if action == constants.Action.Bomb:
                    """
                    if block_teammate_with_bomb is None:
                        block_teammate_with_bomb \
                            = self._get_frac_blocked_two_lists(list_boards_with_bomb,
                                                               n_survivable_nodes_teammate,
                                                               board,
                                                               [my_teammate],
                                                               ignore_dying_agent=True)                        
                    frac = block_teammate_with_bomb 
                    """
                    continue
                else:
                    next_position = info["my_next_position"][action]
                    frac = total_frac_blocked_teammate[next_position]
                if frac < min_interfere:
                    min_interfere = frac
                    least_interfere_action = action
                if frac < self._interfere_threshold:
                    _survivable_actions.append(action)
                else:
                    is_survivable[action] = False

            if len(_survivable_actions) > 1:
                survivable_actions = _survivable_actions
            elif best_action is not None:
                # Take the least interfering action
                return best_action.value
            else:
                return least_interfere_action.value

        consider_bomb = True
        if not is_survivable[constants.Action.Bomb]:
            consider_bomb = False
            
        #
        # Find reachable items
        #

        # List of boards simulated
        list_boards, _ = self._board_sequence(board,
                                              info["curr_bombs"],
                                              info["curr_flames"],
                                              self._search_range,
                                              my_position,
                                              enemy_mobility=mobility,
                                              enemy_bomb=self._enemy_bomb,
                                              enemy_positions=agent_positions,
                                              agent_blast_strength=info["agent_blast_strength"],
                                              step_to_collapse=info["step_to_collapse"],
                                              collapse_ring=info["collapse_ring"])

        # some bombs may explode with extra bombs, leading to under estimation
        for t in range(len(list_boards)):
            flame_positions = np.where(info["list_boards_no_move"][t] == constants.Item.Flames.value)
            list_boards[t][flame_positions] = constants.Item.Flames.value
        
        # List of the set of survivable time-positions at each time
        # and preceding positions
        survivable, prev, _, _ = self._search_time_expanded_network(list_boards,
                                                                    my_position)        
        if len(survivable[-1]) == 0:
            survivable = [set() for _ in range(len(survivable))]

        # Items and bomb target that can be reached in a survivable manner
        if "escape" in info:
            reachable_items, _, next_to_items \
                = self._find_reachable_items(list_boards,
                                             my_position,
                                             survivable,
                                             might_powerup=info["escape"])  # might_powerup is the escape from collapse
        else:
            _, _, next_to_items \
                = self._find_reachable_items(list_boards,
                                             my_position,
                                             survivable)
        
        #
        # If I have seen an enemy recently and cannot see him now, them move to the last seen position
        #

        action = self._action_to_enemy(my_position, next_to_items[constants.Item.Fog], prev, is_survivable, info)
        if action is not None:
            return action.value            
        
        #
        # If I have seen a teammate recently, them move away from the last seen position
        #

        action = self._action_away_from_teammate(my_position,
                                                 next_to_items[constants.Item.Fog],
                                                 prev,
                                                 is_survivable,
                                                 info)
        if action is not None:
            return action.value            

        #
        # Move to the places that will not be collapsed
        #

        if "escape" in info:
            # might_powerup is the escape from collapse
            action = self._action_to_might_powerup(my_position, reachable_items, prev, is_survivable)
            if action is not None:
                print("Escape from collapse", action)
                return action.value
        
        #
        # Move towards a fog where we have not seen longest
        #
        
        action = self._action_to_fog(my_position, next_to_items[constants.Item.Fog], prev, is_survivable, info)

        if action is not None:
            #if True:
            if self.random.uniform() < 0.8:
                return action.value            

        #
        # Choose most survivable action
        #

        max_block = 0
        best_action = None
        for action in survivable_actions:
            if action == constants.Action.Bomb:
                continue
            score = block[action]
            if action != constants.Action.Bomb:
                score += np.random.uniform(0, 1e-3)
            if score > max_block:
                max_block = score
                best_action = action

        if best_action is None:
            max_p = 0
            best_action = None
            for action in p_survivable:
                score = p_survivable[action]
                if action != constants.Action.Bomb:
                    score += np.random.uniform(0, 1e-3)
                if score > max_p:
                    max_p = score
                    best_action = action

        if best_action is None:
            # this should not be the case
            return None
        else:
            return best_action.value
        
        
