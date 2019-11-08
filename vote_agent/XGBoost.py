from utils import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sharedAdam import *
from New_Reward import *
import torch
import pickle as pkl
import hashlib
import action_prune
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

DataCheck = []
GAMMA = 0.9
DATASET_DIR = 'new_data'#'/home/officer/repark/BBMServer/Dataset/0919'#'/data/DataSet/1010'#'/home/officer/repark/BBMServer/Dataset/1009' # '/dev/shm'#
CKPT_DIR = './model_1022/'
model_path = './model_1022/'
SUMMARYWRITER_DIR = './runs_4conv_1022/'
CKPT_FILE_NAME = CKPT_DIR+'epoch_18000.pth'#''./model_4conv_save/0.pth'


def get_str_md5(str):
    md5 = hashlib.md5()
    myMd5 = md5.update(str.encode("utf-8"))
    myMd5_Digest = md5.hexdigest()
    return myMd5_Digest

 # accuarcy
def AccuarcyCompute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    #     print(pred.shape(),label.shape())
    test_np = (np.argmax(pred, 1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)

def read_data(data_file):
    f = open(os.path.join(DATASET_DIR, data_file), 'rb')
    Episodes_dataset = pkl.load(f)
    batch_obs, batch_actions = [],[]
    ep_steps = [[], [], [], []]
    for episode_counter in range(1):
        action_history = -np.ones((4, 6), dtype=np.int32)
        history_obs = -np.ones((4, S_statespace, 11, 11))
        ep_rewards = [[], [], [], []]
        one_Episode_step_record = Episodes_dataset[episode_counter]
        ep_step = 0
        one_Step_record = one_Episode_step_record[ep_step]
        one_Step_state = one_Step_record['state']
        # print(len(one_Step_state))
        one_Step_action = one_Step_record['action']
        ep_steps_total = len(one_Episode_step_record)
        # step_count = 0 the start of an episode of game
        for agent_nr in range(4):
            ep_steps[agent_nr].append(ep_steps_total - 1)
        ep_steps_done_flag = [0, 0, 0, 0]
        num = 0
        sample_bad = 0
        while one_Step_state[0]['step_count'] < ep_steps_total - 2:  # state
            num += 1
            prev_states = [None for _ in range(2)]
            if num >= 0:
              for agent_nr in range(4):
                if (agent_nr + 10) in one_Step_state[agent_nr]['alive']:
                    state_tmp = one_Step_state[agent_nr]
                    safe_actions = action_prune.get_filtered_actions(state_tmp, prev_states)
                    my_position = tuple(state_tmp['position'])
                    board = np.array(state_tmp['board'])
                    bombs = convert_bombs(np.array(state_tmp['bomb_blast_strength']))
                    enemies = [constants.Item(e) for e in state_tmp['enemies']]
                    ammo = int(state_tmp['ammo'])
                    blast_strength = int(state_tmp['blast_strength'])
                    items, dist, prev = djikstra(
                        board, my_position, bombs, enemies, depth=10)

                    if near_target(my_position,items,enemies,dist,prev,3) == None:
                        sample_bad+=1
                        print("Bad Sample! {0}/{1}={2}".format(sample_bad,num,sample_bad/num))
                        continue

                    else:
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
                        woods=[
                            constants.Item.Wood
                        ]
                        bombsValue = [
                            constants.Item.Bomb
                        ]
                        Flames = [
                            constants.Item.Flames
                        ]
                        item_direction = [0,0,0,0,0,0]
                        valid_direction = [0,0,0,0,0,0]
                        '''
                        1个距离内是否有(敌人, 物资, 木墙, 炸弹, 火焰)
                        '''
                        EP1 = nearest_position(dist,enemies,items,1)
                        GP1 = nearest_position(dist,goods,items,1)
                        WP1 = nearest_position(dist,woods,items,1)
                        BP1 = nearest_position(dist,bombsValue,items,1)
                        FP1 = nearest_position(dist,Flames,items,1)
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
                            item_direction[0] = get_direction_towards_position(my_position,EP3,prev).value
                        else:
                            EP3 = (-1,-1)
                        if GP3 != None:
                            item_direction[1] = get_direction_towards_position(my_position,GP3,prev).value
                        else:
                            GP3 = (-1,-1)
                        if WP3 != None:
                            item_direction[2] = get_direction_towards_position(my_position,WP3,prev).value
                        else:
                            WP3 = (-1,-1)
                        if BP3 != None:
                            item_direction[3] = get_direction_towards_position(my_position,BP3,prev).value
                        else:
                            BP3 = (-1,-1)
                        if FP3 != None:
                            item_direction[4] = get_direction_towards_position(my_position,FP3,prev).value
                        else:
                            FP3 = (-1,-1)
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
                            valid_direction[directions.index(d)] +=1
                        '''
                        当前安全的动作列表
                        '''
                        for s in safe_actions:
                            # print(s)
                            valid_direction[s] +=1

                        state_features = have_enemy + have_woodwall + \
                                          have_good + have_bomb + have_fire + \
                                          item_direction + valid_direction + \
                                          list(EP3) + list(GP3) + list(WP3) + \
                                          list(BP3) + list(FP3) + list(my_position)

                        # state_md5 = get_str_md5(str(state_features))
                        state_md5 =  str(state_features)
                        # #get_str_md5(str(state_features))
                        #print(state_features)
                        if state_md5 not in DataCheck:
                            batch_obs.append(state_features)
                            obs_action = one_Step_action[agent_nr]
                            batch_actions.append(obs_action)
                            DataCheck.append(state_md5)
                        else:
                            # print("DataCheck : ", len(DataCheck))
                            # print(np.array(state_features).shape)
                            # print(state_md5)
                            # print(DataCheck)
                            continue
                elif ep_steps_done_flag[agent_nr] == 0:
                    ep_steps[agent_nr][-1] = one_Step_state[0]['step_count'] + 1
                    ep_steps_done_flag[agent_nr] = 1
              ep_step += 1
              one_Step_record = one_Episode_step_record[ep_step]
              one_Step_state = one_Step_record['state']
              one_Step_action = one_Step_record['action']
    return batch_obs, batch_actions

if __name__ == '__main__':
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    imitation_targets = [0,]#, 1, 2, 3]  # only imitate the data from agent1 and agent3
    data_file_num = len(os.listdir(DATASET_DIR))
    count_num = 0
    batch_obs_all, batch_actions_all = [], []

    f = open("new_data/features_20_10.pkl", "rb")
    dataset = pkl.load(f)
    batch_actions_all = dataset[1]
    batch_obs_all = dataset[0]

    print("batch_actions_all : ", len(batch_actions_all))
    print("batch_obs_all : ", len(batch_obs_all))
    X_train, X_test, y_train, y_test = train_test_split(batch_obs_all, batch_actions_all, test_size=0.01, random_state=42)
    data_train = xgb.DMatrix(X_train,label=y_train)
    data_test = xgb.DMatrix(X_test)
    # data_train = xgb.DMatrix(batch_obs_all,label=batch_actions_all)
    param = {'max_depth': 6, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', "num_class":6}
    # bst = xgb.train(param, data_train, 999)
    bst = xgb.Booster(model_file='xgb.model')
    train_preds = bst.predict(data_test)
    # train_preds = bst.predict(X_test)
    train_predictions = [round(value) for value in train_preds]  # 进行四舍五入的操作--变成0.1(算是设定阈值的符号函数)
    print(train_predictions)
    print(len(train_predictions))
    train_accuracy = accuracy_score(y_test, train_predictions)  # 使用sklearn进行比较正确率
    print("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
    # bst.save_model('xgb.model')