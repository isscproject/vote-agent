from collections import Counter
import xgboost as xgb
from pommerman.agents import BaseAgent
from dypmAgents.master_agent4 import MasterAgent as dypm_agent4
from a3c_mlp import A3C_MLP_NET
from utils import *
from features import *
from action_prune import *
import random
from collections import defaultdict
from dypmAgents.base_agent import MyBaseAgent
from dypmAgents.isolated_agent import IsolatedAgent
from dypmAgents.generic_agent import GenericAgent
from dypmAgents.battle_agent import BattleAgent
from dypmAgents.surviving_agent import SurvivingAgent

# (C) Copyright IBM Corp. 2018
from copy import deepcopy



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
            self._when_collapse[L, :][L:U + 1] = t
            self._when_collapse[U, :][L:U + 1] = t
            self._when_collapse[:, L][L:U + 1] = t
            self._when_collapse[:, U][L:U + 1] = t

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

            info["recently_seen"][L, :][L:U + 1] = constants.Item.Rigid.value
            info["recently_seen"][U, :][L:U + 1] = constants.Item.Rigid.value
            info["recently_seen"][:, L][L:U + 1] = constants.Item.Rigid.value
            info["recently_seen"][:, U][L:U + 1] = constants.Item.Rigid.value

            info["recently_seen"][flames_positions] = constants.Item.Flames.value

            obs["bomb_life"][L, :][L:U + 1] = 0
            obs["bomb_life"][U, :][L:U + 1] = 0
            obs["bomb_life"][:, L][L:U + 1] = 0
            obs["bomb_life"][:, U][L:U + 1] = 0

            obs["bomb_blast_strength"][L, :][L:U + 1] = 0
            obs["bomb_blast_strength"][U, :][L:U + 1] = 0
            obs["bomb_blast_strength"][:, L][L:U + 1] = 0
            obs["bomb_blast_strength"][:, U][L:U + 1] = 0

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
                become_passage = (info["list_boards_no_move"][-1] == constants.Item.Passage.value)
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
            rows, cols = np.where(board == enemy.value)
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
            rows, cols = np.where(board == my_teammate.value)
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


def convert_bombs(bomb_map):
        '''Flatten outs the bomb array'''
        ret = []
        locations = np.where(bomb_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'blast_strength': int(bomb_map[(r, c)])
            })
        return ret


def onevote(input:list):
        c = Counter(input)
        return c.most_common(1)[0][0]

class ISSCAgent(BaseAgent):
    def __init__(self, modelpath1, modelpath2, *args, **kwargs):
        super(ISSCAgent, self).__init__(*args, **kwargs)
        model1 = load_checkpoint(modelpath1, A3C_MLP_NET())
        self.model_issc = model1
        bst = xgb.Booster(model_file=modelpath2)
        self.model_xgb = bst
        self.reward = None
        self.stage_signal = 0
        self.prev_actions = [None for _ in range(2)]
        self.prev_positions = [None for _ in range(2)]
        self._recently_visited_positions = []
        self._recently_visited_length = 6
        self._prev_direction = None
        self.prev_states = [None for _ in range(2)]


    def clear_actions(self):
        self.prev_states = [None for _ in range(2)]
        self.prev_actions = [None for _ in range(2)]
        self.prev_postions = [None for _ in range(2)]
        return self


    def act(self, obs, action_space):
        act = dypm_agent4.act(MasterAgent(), obs, action_space)
        self.my_position = tuple(obs['position'])
        self.board = np.array(obs['board'])
        self.bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
        self.enemies = [constants.Item(e) for e in obs['enemies']]
        self.ammo = int(obs['ammo'])
        self.blast_strength = int(obs['blast_strength'])
        self.items, self.dist, self.prev = self._djikstra(
            self.board, self.my_position, self.bombs, self.enemies, depth=10)
        enemy_flag = False
        for idx in self.enemies:
            if idx.value in obs["board"]:
                enemy_flag = True
                break
        # print(enemy_flag)
        if enemy_flag:
            if near_target(self.my_position, self.items, self.enemies, self.dist, self.prev, 3) != None:
                safe_actions = get_filtered_actions(obs, self.prev_states) #, self.prev_actions, self.prev_positions
                data = get_feature(self.board, safe_actions, self.my_position, self.dist, self.items, self.prev, self.enemies)
                #------------------------------------------------
                input_tmp = torch.tensor(data).float()
                logit,value = self.model_issc(input_tmp)
                data = xgb.DMatrix(data)
                pred = self.model_xgb.predict(data)
                train_predictions = [round(value) for value in pred]
                # print(train_predictions)
                logit_cp = copy.deepcopy(logit.clone().detach().numpy())
                actionxgb = -1
                for idx in range(6):
                    if idx in safe_actions:
                        if idx == int(train_predictions[0]):
                            actionxgb = idx
                    else:
                        logit_cp[0][idx] = -float("inf")
                action = torch.argmax(torch.Tensor(logit_cp), dim=-1)
                action = action.item()
                act_set = [act, action, actionxgb]
                # print(act_set)
                action = onevote(act_set)
                # print('vote:', action, '     dypm', act)
            else:
                action = act
        else:
            action = act
        #------------------------------------------------------

        # if near_target(self.my_position, self.items, self.enemies, self.dist, self.prev, 3) != None:
        #     act = safe_actions[random.randint(0,len(safe_actions)-1)]
        #     self.stage_signal = 1
        # else:
        #     self.stage_signal = 0
        #     while True:
        #         # Move if we are in an unsafe place.
        #         unsafe_directions = self._directions_in_range_of_bomb(
        #             self.board, self.my_position, self.bombs, self.dist)
        #         if unsafe_directions:
        #             directions = self._find_safe_directions(
        #                 self.board, self.my_position, unsafe_directions, self.bombs, self.enemies)
        #             act = random.choice(directions).value
        #             if act not in safe_actions:
        #                 act = safe_actions[random.randint(0, len(safe_actions) - 1)]
        #             break
        #
        #
        #         # Lay pomme if we are adjacent to an enemy.
        #         if self._is_adjacent_enemy(self.items, self.dist, self.enemies) and self._maybe_bomb(
        #                 self.ammo, self.blast_strength, self.items, self.dist, self.my_position):
        #             act = constants.Action.Bomb.value
        #             break
        #
        #         # Move towards an enemy if there is one in exactly three reachable spaces.
        #         direction = self._near_enemy(self.my_position, self.items, self.dist, self.prev, self.enemies, 3)
        #         if direction is not None and (self._prev_direction != direction or
        #                                       random.random() < .5):
        #             self._prev_direction = direction
        #             act = direction
        #             if act not in safe_actions:
        #                 act = safe_actions[random.randint(0, len(safe_actions) - 1)]
        #             break
        #             # return direction.value
        #
        #         # Move towards a good item if there is one within two reachable spaces.
        #         direction = self._near_good_powerup(self.my_position, self.items, self.dist, self.prev, 2)
        #         if direction is not None:
        #             act = direction
        #             if act not in safe_actions:
        #                 act = safe_actions[random.randint(0, len(safe_actions) - 1)]
        #             break
        #             # return direction.value
        #
        #         # Maybe lay a bomb if we are within a space of a wooden wall.
        #         if self._near_wood(self.my_position, self.items, self.dist, self.prev, 1):
        #             if self._maybe_bomb(self.ammo, self.blast_strength, self.items, self.dist, self.my_position):
        #                 act = constants.Action.Bomb.value
        #                 break
        #             else:
        #                 act = constants.Action.Stop.value
        #                 break
        #
        #         # Move towards a wooden wall if there is one within two reachable spaces and you have a bomb.
        #         direction = self._near_wood(self.my_position, self.items, self.dist, self.prev, 2)
        #         if direction is not None:
        #             directions = self._filter_unsafe_directions(self.board, self.my_position,
        #                                                         [direction], self.bombs)
        #             if directions:
        #                 act = directions[0].value
        #                 if act not in safe_actions:
        #                     act = safe_actions[random.randint(0, len(safe_actions) - 1)]
        #                 break
        #                 # return directions[0].value
        #
        #         # Choose a random but valid direction.
        #         directions = [
        #             constants.Action.Stop, constants.Action.Left,
        #             constants.Action.Right, constants.Action.Up, constants.Action.Down
        #         ]
        #         valid_directions = self._filter_invalid_directions(
        #             self.board, self.my_position, directions, self.enemies)
        #         directions = self._filter_unsafe_directions(self.board, self.my_position,
        #                                                     valid_directions, self.bombs)
        #         directions = self._filter_recently_visited(
        #             directions, self.my_position, self._recently_visited_positions)
        #         if len(directions) > 1:
        #             directions = [k for k in directions if k != constants.Action.Stop]
        #         if not len(directions):
        #             directions = [constants.Action.Stop]
        #
        #         # Add this position to the recently visited uninteresting positions so we don't return immediately.
        #         self._recently_visited_positions.append(self.my_position)
        #         self._recently_visited_positions = self._recently_visited_positions[
        #                                            -self._recently_visited_length:]
        #
        #         act = random.choice(directions).value
        #         if act not in safe_actions:
        #             act = safe_actions[random.randint(0, len(safe_actions) - 1)]
        #         break
        self.prev_actions[:-1] = self.prev_actions[1:]
        self.prev_actions[-1] = action
        self.prev_positions[:-1] = self.prev_positions[1:]
        self.prev_positions[-1] = self.my_position
        self.prev_states[0] = self.prev_states[1]
        self.prev_states[1] = obs
        return action

    @staticmethod
    def _djikstra(board, my_position, bombs, enemies, depth=None, exclude=None):
        assert (depth is not None)

        if exclude is None:
            exclude = [
                constants.Item.Fog, constants.Item.Rigid, constants.Item.Flames
            ]

        def out_of_range(p_1, p_2):
            '''Determines if two points are out of rang of each other'''
            x_1, y_1 = p_1
            x_2, y_2 = p_2
            return abs(y_2 - y_1) + abs(x_2 - x_1) > depth

        items = defaultdict(list)
        dist = {}
        prev = {}
        Q = queue.Queue()

        my_x, my_y = my_position
        for r in range(max(0, my_x - depth), min(len(board), my_x + depth)):
            for c in range(max(0, my_y - depth), min(len(board), my_y + depth)):
                position = (r, c)
                if any([
                    out_of_range(my_position, position),
                    utility.position_in_items(board, position, exclude),
                ]):
                    continue

                prev[position] = None
                item = constants.Item(board[position])
                items[item].append(position)

                if position == my_position:
                    Q.put(position)
                    dist[position] = 0
                else:
                    dist[position] = np.inf

        for bomb in bombs:
            if bomb['position'] == my_position:
                items[constants.Item.Bomb].append(my_position)

        while not Q.empty():
            position = Q.get()

            if utility.position_is_passable(board, position, enemies):
                x, y = position
                val = dist[(x, y)] + 1
                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + x, col + y)
                    if new_position not in dist:
                        continue

                    if val < dist[new_position]:
                        dist[new_position] = val
                        prev[new_position] = position
                        Q.put(new_position)
                    elif (val == dist[new_position] and random.random() < .5):
                        dist[new_position] = val
                        prev[new_position] = position

        return items, dist, prev

    def _directions_in_range_of_bomb(self, board, my_position, bombs, dist):
        ret = defaultdict(int)

        x, y = my_position
        for bomb in bombs:
            position = bomb['position']
            distance = dist.get(position)
            if distance is None:
                continue

            bomb_range = bomb['blast_strength']
            if distance > bomb_range:
                continue

            if my_position == position:
                # We are on a bomb. All directions are in range of bomb.
                for direction in [
                    constants.Action.Right,
                    constants.Action.Left,
                    constants.Action.Up,
                    constants.Action.Down,
                ]:
                    ret[direction] = max(ret[direction], bomb['blast_strength'])
            elif x == position[0]:
                if y < position[1]:
                    # Bomb is right.
                    ret[constants.Action.Right] = max(
                        ret[constants.Action.Right], bomb['blast_strength'])
                else:
                    # Bomb is left.
                    ret[constants.Action.Left] = max(ret[constants.Action.Left],
                                                     bomb['blast_strength'])
            elif y == position[1]:
                if x < position[0]:
                    # Bomb is down.
                    ret[constants.Action.Down] = max(ret[constants.Action.Down],
                                                     bomb['blast_strength'])
                else:
                    # Bomb is down.
                    ret[constants.Action.Up] = max(ret[constants.Action.Up],
                                                   bomb['blast_strength'])
        return ret

    def _find_safe_directions(self, board, my_position, unsafe_directions,
                              bombs, enemies):

        def is_stuck_direction(next_position, bomb_range, next_board, enemies):
            '''Helper function to do determine if the agents next move is possible.'''
            Q = queue.PriorityQueue()
            Q.put((0, next_position))
            seen = set()

            next_x, next_y = next_position
            is_stuck = True
            while not Q.empty():
                dist, position = Q.get()
                seen.add(position)

                position_x, position_y = position
                if next_x != position_x and next_y != position_y:
                    is_stuck = False
                    break

                if dist > bomb_range:
                    is_stuck = False
                    break

                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + position_x, col + position_y)
                    if new_position in seen:
                        continue

                    if not utility.position_on_board(next_board, new_position):
                        continue

                    if not utility.position_is_passable(next_board,
                                                        new_position, enemies):
                        continue

                    dist = abs(row + position_x - next_x) + abs(col + position_y - next_y)
                    Q.put((dist, new_position))
            return is_stuck

        # All directions are unsafe. Return a position that won't leave us locked.
        safe = []

        if len(unsafe_directions) == 4:
            next_board = board.copy()
            next_board[my_position] = constants.Item.Bomb.value

            for direction, bomb_range in unsafe_directions.items():
                next_position = utility.get_next_position(
                    my_position, direction)
                next_x, next_y = next_position
                if not utility.position_on_board(next_board, next_position) or \
                        not utility.position_is_passable(next_board, next_position, enemies):
                    continue

                if not is_stuck_direction(next_position, bomb_range, next_board,
                                          enemies):
                    # We found a direction that works. The .items provided
                    # a small bit of randomness. So let's go with this one.
                    return [direction]
            if not safe:
                safe = [constants.Action.Stop]
            return safe

        x, y = my_position
        disallowed = []  # The directions that will go off the board.

        for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            position = (x + row, y + col)
            direction = utility.get_direction(my_position, position)

            # Don't include any direction that will go off of the board.
            if not utility.position_on_board(board, position):
                disallowed.append(direction)
                continue

            # Don't include any direction that we know is unsafe.
            if direction in unsafe_directions:
                continue

            if utility.position_is_passable(board, position,
                                            enemies) or utility.position_is_fog(
                board, position):
                safe.append(direction)

        if not safe:
            # We don't have any safe directions, so return something that is allowed.
            safe = [k for k in unsafe_directions if k not in disallowed]

        if not safe:
            # We don't have ANY directions. So return the stop choice.
            return [constants.Action.Stop]

        return safe

    @staticmethod
    def _is_adjacent_enemy(items, dist, enemies):
        for enemy in enemies:
            for position in items.get(enemy, []):
                if dist[position] == 1:
                    return True
        return False

    @staticmethod
    def _has_bomb(obs):
        return obs['ammo'] >= 1

    @staticmethod
    def _maybe_bomb(ammo, blast_strength, items, dist, my_position):
        """Returns whether we can safely bomb right now.

        Decides this based on:
        1. Do we have ammo?
        2. If we laid a bomb right now, will we be stuck?
        """
        # Do we have ammo?
        if ammo < 1:
            return False

        # Will we be stuck?
        x, y = my_position
        for position in items.get(constants.Item.Passage):
            if dist[position] == np.inf:
                continue

            # We can reach a passage that's outside of the bomb strength.
            if dist[position] > blast_strength:
                return True

            # We can reach a passage that's outside of the bomb scope.
            position_x, position_y = position
            if position_x != x and position_y != y:
                return True

        return False

    @staticmethod
    def _nearest_position(dist, objs, items, radius):
        nearest = None
        dist_to = max(dist.values())

        for obj in objs:
            for position in items.get(obj, []):
                d = dist[position]
                if d <= radius and d <= dist_to:
                    nearest = position
                    dist_to = d

        return nearest

    @staticmethod
    def _get_direction_towards_position(my_position, position, prev):
        if not position:
            return None

        next_position = position
        while prev[next_position] != my_position:
            next_position = prev[next_position]

        return utility.get_direction(my_position, next_position)

    @classmethod
    def _near_enemy(cls, my_position, items, dist, prev, enemies, radius):
        nearest_enemy_position = cls._nearest_position(dist, enemies, items,
                                                       radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_enemy_position, prev)

    @classmethod
    def _near_good_powerup(cls, my_position, items, dist, prev, radius):
        objs = [
            constants.Item.ExtraBomb, constants.Item.IncrRange,
            constants.Item.Kick
        ]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    @classmethod
    def _near_wood(cls, my_position, items, dist, prev, radius):
        objs = [constants.Item.Wood]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    @staticmethod
    def _filter_invalid_directions(board, my_position, directions, enemies):
        ret = []
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if utility.position_on_board(
                    board, position) and utility.position_is_passable(
                board, position, enemies):
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_unsafe_directions(board, my_position, directions, bombs):
        ret = []
        for direction in directions:
            x, y = utility.get_next_position(my_position, direction)
            is_bad = False
            for bomb in bombs:
                bomb_x, bomb_y = bomb['position']
                blast_strength = bomb['blast_strength']
                if (x == bomb_x and abs(bomb_y - y) <= blast_strength) or \
                        (y == bomb_y and abs(bomb_x - x) <= blast_strength):
                    is_bad = True
                    break
            if not is_bad:
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_recently_visited(directions, my_position,
                                 recently_visited_positions):
        ret = []
        for direction in directions:
            if not utility.get_next_position(
                    my_position, direction) in recently_visited_positions:
                ret.append(direction)

        if not ret:
            ret = directions
        return ret
