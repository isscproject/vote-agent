# (C) Copyright IBM Corp. 2018
from pommerman import constants
from pommerman import characters
from pommerman import utility
import numpy as np
from collections import defaultdict

from dypmAgents.base_agent import MyBaseAgent
from dypmAgents.tools import search_time_expanded_network


class GenericAgent(MyBaseAgent):
    
    def __init__(self,
                 search_range=13,
                 enemy_mobility=3,
                 enemy_bomb=0,
                 chase_until=25,
                 inv_tmp=200,
                 interfere_threshold=0.6,
                 my_survivability_coeff=0.8,
                 teammate_survivability_coeff=0.5,
                 bomb_threshold=0.1,
                 chase_threshold=0.1,
                 backoff=0.95):

        #
        # Parameters
        #
        
        self._search_range = search_range
        self._enemy_mobility = enemy_mobility
        self._enemy_bomb = enemy_bomb
        self._inv_tmp = inv_tmp  # inverse temperature in choosing best position against enemies
        self._inv_tmp_init = inv_tmp  # inverse temperature in choosing best position against enemies
        self._chase_until = chase_until  # how long to chase enemies after losing sight
        self._interfere_threshold = interfere_threshold   # fraction of teammate survivability allowed to interfere
        self._my_survivability_threshold \
            = my_survivability_coeff * self._search_range * (self._search_range + 1)  # avoid risky actions below this threshold
        self._teammate_survivability_threshold \
            = teammate_survivability_coeff * self._search_range * (self._search_range + 1)  # avoid risky actions below this threshold
        self._bomb_threshold = bomb_threshold  # survivability-tradeoff threshold to place bomb
        self._chase_threshold = chase_threshold  # minimum fraction of blocks to chase
        self._backoff = backoff  # factor to reduce inv_tmp to avoid deadlock
        
        self.random = np.random.RandomState(0)
        super().__init__()


    def _action_most_survivable(self, n_survivable):

        most_survivable_action = None
        max_n_survivable = 0
        for action in n_survivable:
            n = sum(n_survivable[action]) + self.random.uniform()
            if n > max_n_survivable:
                max_n_survivable = n
                most_survivable_action = action

        return most_survivable_action
    
    def _action_to_target(self, my_position, reachable_items, prev, is_survivable):
        
        good_time_positions = reachable_items["target"]

        return self._find_distance_minimizer(my_position,
                                             good_time_positions,
                                             prev,
                                             is_survivable)
    
    def _action_to_powerup(self, my_position, reachable_items, prev, is_survivable):

        good_items = [constants.Item.ExtraBomb, constants.Item.IncrRange, constants.Item.Kick]

        good_time_positions = set()  # positions with good items
        for item in good_items:
            good_time_positions = good_time_positions.union(reachable_items[item])

        return self._find_distance_minimizer(my_position,
                                             good_time_positions,
                                             prev,
                                             is_survivable)
    
    def _action_to_might_powerup(self, my_position, reachable_items, prev, is_survivable):
        
        good_time_positions = reachable_items["might_powerup"]

        return self._find_distance_minimizer(my_position,
                                             good_time_positions,
                                             prev,
                                             is_survivable)
        
    def _action_to_enemy(self, my_position, next_to_fogs, prev, is_survivable, info):

        enemy_position = None
        latest = np.inf
        for position in info["enemy_positions"]:
            when = info["since_last_seen"][position]
            if when > self._chase_until:
                # I have not seen him too long
                continue
            if when < latest:
                latest = when
                enemy_position = position

        if enemy_position is None:
            return None

        xx, yy = enemy_position
        # move towards the fog nearest to the enemy position
        shortest_distance = np.inf
        best_time_position = []
        for t, x, y in next_to_fogs:
            fog_to_enemy = abs(x - xx) + abs(y - yy)
            fog_to_enemy += self.random.uniform()
            if t + fog_to_enemy < shortest_distance:
                shortest_distance = t + fog_to_enemy
                best_time_position = [(t, x, y)]
        
        return self._find_distance_minimizer(my_position,
                                             best_time_position,
                                             prev,
                                             is_survivable)

    def _action_away_from_teammate(self, my_position, next_to_fogs, prev, is_survivable, info):

        if info["teammate_position"] is None:
            return None

        if info["since_last_seen"][info["teammate_position"]] > self._chase_until:
            return None
        
        xx, yy = info["teammate_position"]

        # move towards the fog nearest to the teammate position
        longest_distance = -np.inf
        best_time_position = []
        for t, x, y in next_to_fogs:
            fog_to_enemy = abs(x - xx) + abs(y - yy)
            fog_to_enemy += self.random.uniform()
            if t + fog_to_enemy > longest_distance:
                longest_distance = t + fog_to_enemy
                best_time_position = [(t, x, y)]
        
        return self._find_distance_minimizer(my_position,
                                             best_time_position,
                                             prev,
                                             is_survivable)

    def _action_to_fog(self, my_position, next_to_fogs, prev, is_survivable, info):

        best_time_position = []
        oldest = 0
        for t, x, y in next_to_fogs:
            neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            age = max([info["since_last_seen"][position] for position in neighbors if self._on_board(position)])
            age += self.random.uniform()
            if age > oldest:
                oldest = age
                best_time_position = [(t, x, y)]

        return self._find_distance_minimizer(my_position,
                                             best_time_position,
                                             prev,
                                             is_survivable)
    
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

        #
        # Bomb to break wood
        #

        # where to place bombs to break wood
        if info["steps"] < 100:
            digging, bomb_target = self._get_digging_positions(board, my_position, info)
        else:
            digging = None

        if digging is None:
            bomb_target, n_breakable \
                = self._get_bomb_target(info["list_boards_no_move"][-1],
                                        my_position,
                                        my_blast_strength,
                                        constants.Item.Wood)

        consider_bomb = True
        if not is_survivable[constants.Action.Bomb]:
            consider_bomb = False
        elif self._might_break_powerup(info["list_boards_no_move"][-1],
                                       my_position,
                                       my_blast_strength,
                                       info["might_powerup"]):
            # if might break an item, do not bomb
            consider_bomb = False

        if consider_bomb and bomb_target[my_position]:
            # place bomb if I am at a bomb target
            return constants.Action.Bomb.value

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
                                              
        # wood might be disappeared, because of overestimated bombs
        for t in range(len(list_boards)):
            wood_positions = np.where(info["list_boards_no_move"][t] == constants.Item.Wood.value)
            list_boards[t][wood_positions] = constants.Item.Wood.value                       

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
        reachable_items, reached, next_to_items \
            = self._find_reachable_items(list_boards,
                                         my_position,
                                         survivable,
                                         bomb_target,
                                         info["might_powerup"])

        #
        # Move to dig
        #

        action_to_target = self._action_to_target(my_position, reachable_items, prev, is_survivable)

        if all([digging is not None,
                action_to_target is not None]):
            time_to_reach = reachable_items["target"][0][0]
            if any([my_ammo and board[digging] in [constants.Item.Passage.value,
                                                   constants.Item.ExtraBomb.value,
                                                   constants.Item.IncrRange.value,
                                                   constants.Item.Kick.value],
                    info["flame_life"][digging] <= time_to_reach
                    and utility.position_is_flames(board, digging)]):
                return action_to_target.value        

        #
        # Move towards good items
        #

        action = self._action_to_powerup(my_position, reachable_items, prev, is_survivable)
        if action is not None:
            return action.value

        #
        # Move towards where to bomb to break wood
        #

        if action_to_target is not None:
            return action_to_target.value

        #
        # Move toward might powerups
        #

        action = self._action_to_might_powerup(my_position, reachable_items, prev, is_survivable)
        if action is not None:
            return action.value

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
        
        
