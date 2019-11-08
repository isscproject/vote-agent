# (C) Copyright IBM Corp. 2018
import numpy as np
from collections import defaultdict
from copy import deepcopy
from pommerman import utility
from pommerman import constants

from dypmAgents.base_agent import MyBaseAgent
from dypmAgents.isolated_agent import IsolatedAgent
from dypmAgents.generic_agent import GenericAgent
from dypmAgents.battle_agent import BattleAgent
from dypmAgents.surviving_agent import SurvivingAgent


class MasterAgent(MyBaseAgent):

    def __init__(self,
                 search_range=10,
                 enemy_mobility=3,
                 enemy_bomb=0,
                 chase_until=25,
                 inv_tmp=200,
                 interfere_threshold=0.6,
                 my_survivability_coeff=0.8,
                 teammate_survivability_coeff=0.5,
                 bomb_threshold=0.1,
                 chase_threshold=0.1,
                 backoff=0.95,
                 use_last_seen=5):

        """
        The master agent determines the phase of the game,
        and let the expert agent for that phase choose the action.
        """
        
        super().__init__()
        self._search_range = search_range

        self._steps = 0
        self._use_last_seen = use_last_seen
        
        # Keep essential information in the previous steps
        self._prev_bomb_life = np.zeros(self.board_shape, dtype="uint8")
        self._prev_bomb_blast_strength = np.zeros(self.board_shape, dtype="uint8")
        self._prev_flame_life = np.zeros(self.board_shape, dtype="uint8")
        self._prev_moving_direction = np.full(self.board_shape, None)
        self._prev_board = None
        self._prev_bomb_position_strength = list()
        self._agent_blast_strength = dict()
        self._prev_action = None
        self._prev_position = None
        self._isolated = True
        self._just_collapsed = None

        # Estimated map
        self._num_rigid_found = 0
        self._might_remaining_powerup = True
        self._is_rigid = np.full(self.board_shape, False)  # false may be unknown
        self._last_seen = np.full(self.board_shape, constants.Item.Fog.value, dtype="uint8")
        self._since_last_seen = np.full(self.board_shape, np.inf)
        self._unreachable = np.full(self.board_shape, False)

        # collapse timings
        first = constants.FIRST_COLLAPSE
        last = constants.MAX_STEPS
        self._collapse_steps = list(range(first, last, int((last - first) / 4)))

        self._when_collapse = np.full(self.board_shape, np.inf)
        for L, t in enumerate(self._collapse_steps):
            U = self.board_shape[0] - 1 - L
            self._when_collapse[L, :][L:U+1] = t
            self._when_collapse[U, :][L:U+1] = t
            self._when_collapse[:, L][L:U+1] = t
            self._when_collapse[:, U][L:U+1] = t
            
        # Slaves
        self.isolated_slave = IsolatedAgent(search_range=search_range)
        self.generic_slave = GenericAgent(search_range=search_range,
                                          enemy_mobility=enemy_mobility,
                                          enemy_bomb=enemy_bomb,
                                          inv_tmp=inv_tmp,
                                          chase_until=chase_until,
                                          interfere_threshold=interfere_threshold,
                                          my_survivability_coeff=my_survivability_coeff,
                                          teammate_survivability_coeff=teammate_survivability_coeff,
                                          bomb_threshold=bomb_threshold,
                                          chase_threshold=chase_threshold)
        self.battle_slave = BattleAgent(search_range=search_range,
                                        enemy_mobility=enemy_mobility,
                                        enemy_bomb=enemy_bomb,
                                        inv_tmp=inv_tmp,
                                        chase_until=chase_until,
                                        interfere_threshold=interfere_threshold,
                                        my_survivability_coeff=my_survivability_coeff,
                                        teammate_survivability_coeff=teammate_survivability_coeff,
                                        bomb_threshold=bomb_threshold,
                                        chase_threshold=chase_threshold)
        self.surviving_slave = SurvivingAgent(search_range=search_range)

    def act(self, obs, action_space):

        # The number of steps
        self._steps += 1

        # Collapse the board if just collapsed in the previous step
        info = dict()
        info["steps"] = self._steps
        info["recently_seen"] = deepcopy(obs["board"])
        if self._just_collapsed is not None:
            L = self._just_collapsed
            U = obs["board"].shape[0] - 1 - L

            flames_positions = np.where(obs["board"] == constants.Item.Flames.value)
                
            info["recently_seen"][L, :][L:U+1] = constants.Item.Rigid.value
            info["recently_seen"][U, :][L:U+1] = constants.Item.Rigid.value
            info["recently_seen"][:, L][L:U+1] = constants.Item.Rigid.value
            info["recently_seen"][:, U][L:U+1] = constants.Item.Rigid.value

            info["recently_seen"][flames_positions] = constants.Item.Flames.value
            
            obs["bomb_life"][L, :][L:U+1] = 0
            obs["bomb_life"][U, :][L:U+1] = 0
            obs["bomb_life"][:, L][L:U+1] = 0
            obs["bomb_life"][:, U][L:U+1] = 0
            
            obs["bomb_blast_strength"][L, :][L:U+1] = 0
            obs["bomb_blast_strength"][U, :][L:U+1] = 0
            obs["bomb_blast_strength"][:, L][L:U+1] = 0
            obs["bomb_blast_strength"][:, U][L:U+1] = 0

        #
        # Whether each location is Rigid
        #
        #  False may be an unknown
        #
        if self._num_rigid_found < constants.NUM_RIGID:
            self._is_rigid += (obs["board"] == constants.Item.Rigid.value)
            self._is_rigid += (obs["board"].T == constants.Item.Rigid.value)
            self._num_rigid_found = np.sum(self._is_rigid)
            self._unreachable = ~self._get_reachable(self._is_rigid)
            self._unreachable_locations = np.where(self._unreachable)
            
        #
        # What we have seen last time, and how many steps have past since then
        #
        visible_locations = np.where(obs["board"] != constants.Item.Fog.value)
        self._last_seen[visible_locations] = obs["board"][visible_locations]
        self._last_seen[self._unreachable_locations] = constants.Item.Rigid.value
        self._since_last_seen += 1
        self._since_last_seen[visible_locations] = 0
        self._since_last_seen[np.where(self._is_rigid)] = 0
        if self._steps == 0:
            # We have some knowledge about the initial configuration of the board
            C = constants.BOARD_SIZE - 2
            self._last_seen[(1, 1)] = constants.Item.Agent0.value
            self._last_seen[(C, 1)] = constants.Item.Agent1.value
            self._last_seen[(C, C)] = constants.Item.Agent2.value
            self._last_seen[(1, C)] = constants.Item.Agent3.value
            rows = np.array([1, C, 1, C])
            cols = np.array([1, 1, C, C])
            self._since_last_seen[(rows, cols)] = 0
            rows = np.array([1, 1, 1, 1, 2, 3, C - 1, C - 2, C, C, C, C, 2, 3, C - 1, C - 2])
            cols = np.array([2, 3, C - 1, C - 2, 1, 1, 1, 1, 2, 3, C - 1, C - 2, C, C, C, C])
            self._last_seen[(rows, cols)] = constants.Item.Passage.value
            self._since_last_seen[(rows, cols)] = 0

        #
        # We know exactly how my teamate is digging
        #
        my_position = obs["position"]

        if self._steps == 33:
            passage_under_fog \
                = (self._last_seen.T == constants.Item.Passage.value) * (self._last_seen == constants.Item.Fog.value)
            positions = np.where(passage_under_fog)
            self._last_seen[positions] = constants.Item.Passage.value
            self._since_last_seen[positions] = 0
        
        info["since_last_seen"] = self._since_last_seen
        info["last_seen"] = self._last_seen

        if not self._just_collapsed:
            # then we do not see the true board, so skip
            recently_seen_positions = (info["since_last_seen"] < self._use_last_seen)
            info["recently_seen"][recently_seen_positions] = info["last_seen"][recently_seen_positions]

        # TODO: deepcopy are not needed with Docker
        board = info["recently_seen"]
        bomb_life = obs["bomb_life"]
        bomb_blast_strength = obs["bomb_blast_strength"]
        my_enemies = [constants.Item(e) for e in obs['enemies'] if e != constants.Item.AgentDummy]
        if obs["teammate"] != constants.Item.AgentDummy:
            my_teammate = obs["teammate"]
        else:
            my_teammate = None

        info["prev_action"] = self._prev_action
        info["prev_position"] = self._prev_position

        #
        # Modify the board
        #

        board[self._unreachable_locations] = constants.Item.Rigid.value

        #
        # Summarize information about bombs
        #
        #  curr_bombs : list of current bombs
        #  moving_direction : array of moving direction of bombs
        info["curr_bombs"], info["moving_direction"] \
            = self._get_bombs(obs["board"],  # use observation to keep the bombs under fog
                              bomb_blast_strength, self._prev_bomb_blast_strength,
                              bomb_life, self._prev_bomb_life)

        self._prev_bomb_life = bomb_life.copy()
        self._prev_bomb_blast_strength = bomb_blast_strength.copy()

        #
        # Bombs to be exploded in the next step
        #
        curr_bomb_position_strength = list()
        rows, cols = np.where(bomb_blast_strength > 0)
        for position in zip(rows, cols):
            strength = int(bomb_blast_strength[position])
            curr_bomb_position_strength.append((position, strength))

        #
        # Summarize information about flames
        #
        if self._prev_board is not None:
            info["curr_flames"], self._prev_flame_life \
                = self._get_flames(obs["board"],  # use observation to keep the bombs under fog
                                   self._prev_board[-1],
                                   self._prev_flame_life,
                                   self._prev_bomb_position_strength,
                                   curr_bomb_position_strength,
                                   self._prev_moving_direction)
        else:
            info["curr_flames"] = []
        info["flame_life"] = self._prev_flame_life

        self._prev_moving_direction = deepcopy(info["moving_direction"])

        self._prev_bomb_position_strength = curr_bomb_position_strength
        
        #
        # List of simulated boards, assuming enemies stay unmoved
        #

        step_to_collapse = None
        collapse_ring = None
        if obs["game_env"] == 'pommerman.envs.v1:Pomme':
            # Collapse mode

            # cannot trust the board just collapsed, so skip
            if self._just_collapsed is None:
                already_collapsed = (self._when_collapse < self._steps)
                not_rigid = (obs["board"] != constants.Item.Rigid.value) * (obs["board"] != constants.Item.Fog.value)
                not_collapsed_positions = np.where(already_collapsed * not_rigid)
                self._when_collapse[not_collapsed_positions] = np.inf

            collapse_steps = [step for step in self._collapse_steps if step >= self._steps]
            if len(collapse_steps):
                step_to_collapse = min(collapse_steps) - self._steps
                collapse_ring = len(self._collapse_steps) - len(collapse_steps)
                if step_to_collapse == 0:
                    self._just_collapsed = collapse_ring
                else:
                    self._just_collapsed = None
            else:
                self._just_collapsed = None

        info["step_to_collapse"] = step_to_collapse
        info["collapse_ring"] = collapse_ring

        info["list_boards_no_move"], _ \
            = self._board_sequence(board,
                                   info["curr_bombs"],
                                   info["curr_flames"],
                                   self._search_range,
                                   my_position,
                                   enemy_mobility=0,
                                   step_to_collapse=step_to_collapse,
                                   collapse_ring=collapse_ring)

        #
        # Might appear item from flames
        #

        info["might_powerup"] = np.full(self.board_shape, False)
        if self._prev_board is None:
            # Flame life is 2
            # flame life is hardcoded in pommmerman/characters.py class Flame
            self._prev_board = [deepcopy(board), deepcopy(board), deepcopy(board)]
        else:
            old_board = self._prev_board.pop(0)
            self._prev_board.append(deepcopy(board))
            if self._might_remaining_powerup:
                # was wood and now flames
                was_wood = (old_board == constants.Item.Wood.value)
                now_flames = (board == constants.Item.Flames.value)
                info["might_powerup"] = was_wood * now_flames

                # now wood and will passage
                now_wood = (board == constants.Item.Wood.value)
                become_passage = (info["list_boards_no_move"][-1] ==constants.Item.Passage.value)
                info["might_powerup"] += now_wood * become_passage

                maybe_powerup = info["might_powerup"] \
                                + (self._last_seen == constants.Item.Wood.value) \
                                + (self._last_seen == constants.Item.ExtraBomb.value) \
                                + (self._last_seen == constants.Item.IncrRange.value) \
                                + (self._last_seen == constants.Item.Kick.value)            
                if not maybe_powerup.any():
                    self._might_remaining_powerup = False

        # update the estimate of enemy blast strength
        rows, cols = np.where(bomb_life == constants.DEFAULT_BOMB_LIFE - 1)
        for position in zip(rows, cols):
            if position == my_position:
                continue
            enemy = board[position]
            self._agent_blast_strength[enemy] = bomb_blast_strength[position]
        info["agent_blast_strength"] = self._agent_blast_strength

        # enemy positions
        info["enemy_positions"] = list()
        for enemy in my_enemies:
            rows, cols = np.where(board==enemy.value)
            if len(rows) == 1:
                info["enemy_positions"].append((rows[0], cols[0]))
            elif len(rows) > 1:
                # choose the most recently seen enemy of this ID, because only one
                time_passed = info["since_last_seen"][(rows, cols)]
                idx = np.argmin(time_passed)
                enemy_position = (rows[idx], cols[idx])
                board[(rows, cols)] = constants.Item.Passage.value  # overwrite old teammates by passage
                board[enemy_position] = enemy.value
                info["enemy_positions"].append(enemy_position)

        # teammate position
        info["teammate_position"] = None
        if my_teammate is not None:
            rows, cols = np.where(board==my_teammate.value)
            if len(rows) == 1:
                info["teammate_position"] = (rows[0], cols[0])
            elif len(rows) > 1:
                # choose the most recently seen teammate, because only one
                time_passed = info["since_last_seen"][(rows, cols)]
                idx = np.argmin(time_passed)
                info["teammate_position"] = (rows[idx], cols[idx])
                board[(rows, cols)] = constants.Item.Passage.value  # overwrite old teammates by passage
                board[info["teammate_position"]] = my_teammate.value

        # next positions
        info["my_next_position"] = {constants.Action.Stop: my_position}
        if all([obs["ammo"] > 0,
                obs["bomb_blast_strength"][my_position] == 0]):
            info["my_next_position"][constants.Action.Bomb] = my_position
        else:
            info["my_next_position"][constants.Action.Bomb] = None
        for action in [constants.Action.Up, constants.Action.Down,
                       constants.Action.Left, constants.Action.Right]:
            next_position = self._get_next_position(my_position, action)
            if self._on_board(next_position):
                if board[next_position] in [constants.Item.Rigid.value, constants.Item.Wood.value]:
                    info["my_next_position"][action] = None
                else:
                    info["my_next_position"][action] = next_position
            else:
                info["my_next_position"][action] = None

        # kickable positions
        if obs["can_kick"]:
            is_bomb = np.full(self.board_shape, False)
            is_bomb[np.where(obs["bomb_blast_strength"] > 0)] = True
            info["kickable"], info["might_kickable"] \
                = self._kickable_positions(obs, is_bomb, info["moving_direction"])
            info["all_kickable"] = set.union(info["kickable"], info["might_kickable"])
        else:
            info["kickable"] = set()
            info["might_kickable"] = set()
            info["all_kickable"] = set()
                
        # might block/blocked actions
        # I am the leader if agent0 or agent1
        # I am the follower otherwise
        # If leader, not blocked by teammate
        # If follower, do not block teammate
        info["might_blocked"] = self._get_might_blocked(board,
                                                        my_position,
                                                        info["enemy_positions"],
                                                        info["might_kickable"])
        if all([board[my_position] in [constants.Item.Agent2.value, constants.Item.Agent3.value],
                info["teammate_position"] is not None]):
            info["might_block_teammate"] = self._get_might_blocked(board,
                                                                   my_position,
                                                                   [info["teammate_position"]],
                                                                   info["might_kickable"])
        else:
            info["might_block_teammate"] = defaultdict(bool)
        info["might_block_actions"] = set([a for a in info["might_block_teammate"] if info["might_block_teammate"][a]])
        
        #
        # Choose a slave to act
        #

        if self._isolated:
            is_wood_visible = (constants.Item.Wood.value in board)
            is_closed = self._is_closed(board, my_position)
            if any([not is_wood_visible, not is_closed]):
                self._isolated = False

        action = None
        if self._isolated:
            # Act with an agent who do not consider other dypmAgents
            action = self.isolated_slave.act(obs, action_space, info)
        elif not self._might_remaining_powerup:
            # Act with an agent who do not consider powerups

            if obs["game_env"] == 'pommerman.envs.v1:Pomme':
                info["escape"] = (self._when_collapse == np.inf) * (info["last_seen"] == constants.Item.Passage.value)

            action = self.battle_slave.act(obs, action_space, info)
        else:
            action = self.generic_slave.act(obs, action_space, info)
        
        if action is None:
            # Act with a special agent, who only seeks to survive
            action = self.surviving_slave.act(obs, action_space, info)

        self._prev_action = action
        self._prev_position = my_position
        
        return action
            
    def _is_closed(self, board, position):

        """
        Check whether the position is srounded by Wood/Rigid.

        Parameters
        ----------
        board = np.array(obs['board'])

        position = tuple(obs['position'])
        """

        is_done = np.full(board.shape, False)
        is_done[position] = True
        to_search = [position]

        while to_search:
            x, y = to_search.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_position = (x + dx, y + dy)
                if not self._on_board(new_position):
                    continue
                if is_done[new_position]:
                    continue
                is_done[new_position] = True
                if utility.position_is_agent(board, new_position):
                    return False
                if utility.position_is_wall(board, new_position):
                    continue
                if utility.position_is_fog(board, new_position):
                    continue
                to_search.append(new_position)

        return True
