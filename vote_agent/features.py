from pommerman import utility
from pommerman import constants
from pommerman import characters
from collections import defaultdict
from pommerman.agents.simple_agent import SimpleAgent
import queue
import random
import numpy as np
from enum import Enum
'''
判断是否与敌人相邻
'''
def is_adjacent_enemy(items, dist, enemies):
    for enemy in enemies:
        for position in items.get(enemy, []):
            if dist[position] == 1:
                return True
    return False

'''
返回一个最近的点
'''
def nearest_position(dist, objs, items, radius):
    nearest = None
    dist_to = max(dist.values())
    for obj in objs:
        for position in items.get(obj, []):
            d = dist[position]
            if d <= radius and d <= dist_to:
                nearest = position
                dist_to = d
    return nearest

class POSTION(Enum):
    '''The Actions an agent can take'''
    none = -1

'''
得到通往目标点的方向
'''
def get_direction_towards_position(my_position, position, prev):
    if not position:
        return None

    next_position = position
    if prev[next_position] == None:
        return POSTION.none

    while prev[next_position] != my_position:
        next_position = prev[next_position]

    return utility.get_direction(my_position, next_position)

def near_enemy(my_position, items, dist, prev, enemies, radius):
    nearest_enemy_position = nearest_position(dist, enemies, items,
                                                   radius)
    return get_direction_towards_position(my_position,
                                               nearest_enemy_position, prev)
'''
得到附近物资方向
'''
def near_good_powerup( my_position, items, dist, prev, radius):
    objs = [
        constants.Item.ExtraBomb, constants.Item.IncrRange,
        constants.Item.Kick
    ]
    nearest_item_position = nearest_position(dist, objs, items, radius)
    return get_direction_towards_position(my_position,
                                               nearest_item_position, prev)

'''
得到附近的木墙的方向
'''
def near_wood(my_position, items, dist, prev, radius):
    objs = [constants.Item.Wood]
    nearest_item_position = nearest_position(dist, objs, items, radius)
    return get_direction_towards_position(my_position,
                                               nearest_item_position, prev)

def near_target( my_position, items, enemy, dist, prev, radius):
    objs = [
        constants.Item.ExtraBomb, constants.Item.IncrRange,
        constants.Item.Kick, constants.Item.Wood, constants.Item.Bomb,
        constants.Item.Flames
    ]
    objs += enemy
    nearest_item_position = nearest_position(dist, objs, items, radius)
    return nearest_item_position


'''
过滤最近选择的方向
'''
def filter_recently_visited(directions, my_position,
                             recently_visited_positions):
    ret = []
    for direction in directions:
        if not utility.get_next_position(
                my_position, direction) in recently_visited_positions:
            ret.append(direction)

    if not ret:
        ret = directions
    return ret

'''
寻路算法
'''
def djikstra(board, my_position, bombs, enemies, depth=None, exclude=None):
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

'''
判断我们目前能否安全的放一个炸弹(包括有没有炸弹)
'''

def maybe_bomb(ammo, blast_strength, items, dist, my_position):
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

def get_next_position(position, direction):
    '''Returns the next position coordinates'''
    x, y = position
    if direction == constants.Action.Right:
        return (x, y + 1)
    elif direction == constants.Action.Left:
        return (x, y - 1)
    elif direction == constants.Action.Down:
        return (x + 1, y)
    elif direction == constants.Action.Up:
        return (x - 1, y)
    elif direction == constants.Action.Stop:
        return (x, y)
    elif direction == constants.Action.Bomb:
        return (x, y)
    raise constants.InvalidAction("We did not receive a valid direction.")

def filter_invalid_directions(board, my_position, directions, enemies,):
    ret = []
    for direction in directions:
        position = utility.get_next_position(my_position, direction)
        if utility.position_on_board(
                board, position) and utility.position_is_passable(
                    board, position, enemies):
            ret.append(direction)
        return ret

def get_feature(board, safe_actions, my_position, dist, items, prev, enemies):
    have_enemy = []
    have_good = []
    have_woodwall = []
    have_bomb = []
    have_fire = []
    goods = [
        constants.Item.ExtraBomb,
        constants.Item.IncrRange,
        constants.Item.Kick
    ]
    woods = [
        constants.Item.Wood
    ]
    bombsValue = [
        constants.Item.Bomb
    ]
    Flames = [
        constants.Item.Flames
    ]
    item_direction = [0, 0, 0, 0, 0, 0]
    valid_direction = [0, 0, 0, 0, 0, 0]
    '''
    1个距离内是否有(敌人, 物资, 木墙, 炸弹, 火焰)
    '''
    EP1 = nearest_position(dist, enemies, items, 1)
    GP1 = nearest_position(dist, goods, items, 1)
    WP1 = nearest_position(dist, woods, items, 1)
    BP1 = nearest_position(dist, bombsValue, items, 1)
    FP1 = nearest_position(dist, Flames, items, 1)
    have_enemy.append(0 if EP1 == None else 1)
    have_good.append(0 if GP1 == None else 1)
    have_woodwall.append(0 if WP1 == None else 1)
    have_bomb.append(0 if BP1 == None else 1)
    have_fire.append(0 if FP1 == None else 1)

    '''
    2个距离内是否有(敌人, 物资, 木墙, 炸弹, 火焰)
    '''
    EP2 = nearest_position(dist, enemies, items, 2)
    GP2 = nearest_position(dist, goods, items, 2)
    WP2 = nearest_position(dist, woods, items, 2)
    BP2 = nearest_position(dist, bombsValue, items, 2)
    FP2 = nearest_position(dist, Flames, items, 2)
    have_enemy.append(0 if EP2 == None else 1)
    have_good.append(0 if GP2 == None else 1)
    have_woodwall.append(0 if WP2 == None else 1)
    have_bomb.append(0 if BP2 == None else 1)
    have_fire.append(0 if FP2 == None else 1)
    '''
    3个距离内是否有(敌人, 物资, 木墙, 炸弹, 火焰)
    '''
    EP3 = nearest_position(dist, enemies, items, 3)
    GP3 = nearest_position(dist, goods, items, 3)
    WP3 = nearest_position(dist, woods, items, 3)
    BP3 = nearest_position(dist, bombsValue, items, 3)
    FP3 = nearest_position(dist, Flames, items, 3)
    have_enemy.append(0 if EP3 == None else 1)
    have_good.append(0 if GP3 == None else 1)
    have_woodwall.append(0 if WP3 == None else 1)
    have_bomb.append(0 if BP3 == None else 1)
    have_fire.append(0 if FP3 == None else 1)
    '''
    (敌人, 物资, 木墙, 炸弹, 火焰)的位置朝向
    '''
    if EP3 != None:
        item_direction[0] = get_direction_towards_position(my_position, EP3, prev).value
    else:
        EP3 = (-1, -1)
    if GP3 != None:
        item_direction[1] = get_direction_towards_position(my_position, GP3, prev).value
    else:
        GP3 = (-1, -1)
    if WP3 != None:
        item_direction[2] = get_direction_towards_position(my_position, WP3, prev).value
    else:
        WP3 = (-1, -1)
    if BP3 != None:
        item_direction[3] = get_direction_towards_position(my_position, BP3, prev).value
    else:
        BP3 = (-1, -1)
    if FP3 != None:
        item_direction[4] = get_direction_towards_position(my_position, FP3, prev).value
    else:
        FP3 = (-1, -1)
    '''
    当前合法的动作列表
    '''
    directions = [
        constants.Action.Stop, constants.Action.Left,
        constants.Action.Right, constants.Action.Up, constants.Action.Down
    ]
    valid_directions = filter_invalid_directions(
        board, my_position, directions, enemies)
    for d in valid_directions:
        valid_direction[directions.index(d)] += 1
    '''
    当前安全的动作列表
    '''
    for s in safe_actions:
        # print(s)
        valid_direction[s] += 1

    state_features = have_enemy + have_woodwall + \
                     have_good + have_bomb + have_fire + \
                     item_direction + valid_direction + \
                     list(EP3) + list(GP3) + list(WP3) + \
                     list(BP3) + list(FP3) + list(my_position)

    return np.array([state_features])