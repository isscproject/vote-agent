from issc_agent import *
import pommerman
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
import sys
from dypmAgents.master_agent4 import MasterAgent as dypm_agent4
sys.setrecursionlimit(1000000)


class A3CWorker(mp.Process):
    def __init__(self, global_ep, res_dict, rew_queue,
                 agent_input_dict, agent_return_dict,
                 name):
        super(A3CWorker, self).__init__()
        self.agent_nr = name % 4
        self.teammate_nr = int((self.agent_nr + 2) % 4)
        print("{}th agent chosen!".format(self.agent_nr))
        self.name = 'worker_{}_{}'.format(name, self.agent_nr)
        self.g_ep, self.res_dict, self.rew_queue = global_ep, res_dict, rew_queue
        self.agent_input_dict, self.agent_return_dict = agent_input_dict, agent_return_dict
        self.agent_return_dict['game_result'] = []
        self.agent_return_dict['game_steps'] = []
        self.agent_return_dict['game_reward'] = []
        self.nth_pbt_process = 1
        self.share_count = 0
        self.all_epsr = []
        self.uniq_name = uniq_name

    def run(self):
        self.A3CAgent0 = ISSCAgent("model.pth", 'xgb.model')
        self.A3CAgent2 = ISSCAgent("model.pth", 'xgb.model')
        self.agentList = [dypm_agent4() for _ in range(4)]
        self.agentList[self.agent_nr] = self.A3CAgent0
        self.agentList[self.teammate_nr] = self.A3CAgent2
        self.env = pommerman.make('PommeRadioCompetition-v2', self.agentList)
        ep_r_list, r_list, step_count_list = [], [], []
        while self.g_ep['g_ep'] < 500:
            s_act = self.env.reset()
            agent_actions_list = []
            done = False
            ep_rs = 0
            self.A3CAgent0.clear_actions()
            self.A3CAgent2.clear_actions()
            while done is False:
                agent_actions = self.env.act(s_act)
                s_new, r, done, _ = self.env.step(agent_actions)
                agent_actions_list.append(agent_actions)
                # self.env.render()
                if done:
                    if s_new[self.agent_nr]['step_count'] > 799 or s_new[self.teammate_nr]['step_count'] > 799:
                        res = "Tie"
                    else:
                        if (self.agent_nr + 10) in s_new[self.agent_nr]['alive'] or self.teammate_nr + 10 in \
                                s_new[self.teammate_nr]['alive']:
                            res = "Win"
                        else:
                            res = "Lose"
                    self.record(self.g_ep, ep_rs, self.res_dict, self.rew_queue,
                                s_new[self.agent_nr]['step_count'], self.name, self.all_epsr, res)
                    ep_r_list.append(ep_rs)
                    r_list.append(res)
                    step_count_list.append(s_new[self.agent_nr]['step_count'])

                s_act = s_new
        self.agent_return_dict['game_reward'] = ep_r_list
        self.agent_return_dict['game_result'] = r_list
        self.agent_return_dict['game_steps'] = step_count_list

    @staticmethod
    def record(global_ep, ep_r, res_dict, rew_queue, nr_steps, name, eps_average_reward, result):
        eps_average_reward.append(ep_r)
        global_ep['g_ep'] += 1
        global_ep['g_sum_step'] += int(nr_steps)
        res_dict['res_q'].append(ep_r)
        rew_queue.put([ep_r, global_ep['g_ep']])
        print(
            name,
            "| Ep:", global_ep['g_ep'],
            "| Avg Steps: %.2f" % (float(global_ep['g_sum_step']) / float(global_ep['g_ep'])),
            "| Ep_r: %.2f" % ep_r,
            "| Cur Steps: %d" % nr_steps,
            "| Ep_r / Steps: %.2f" % (ep_r / nr_steps),
            "| Result:", result
        )

    def reward_shaping(self, obs, act, obs_next, terminal, current_step, agent_actions_list, pos_list):
        final_res = [0,0,0,0]
        return np.array(final_res)

class TestFunc():
    def __init__(self, uniq_name, nth_pbt_process, input_dict, return_dict):
        super(TestFunc, self).__init__()
        self.nth_pbt_process = nth_pbt_process
        self.input_dict = input_dict
        self.return_dict = return_dict
        self.uniq_name = uniq_name

    def run(self):
        m = Manager()
        global_ep, res_dict = m.dict(), m.dict()
        global_ep['g_ep'] = 0
        global_ep['g_sum_step'] = 0
        res_dict['res_q'] = []
        rew_queue = m.Queue()
        agent_input_dict_list = m.list([self.input_dict for _ in range(1)])
        agent_return_dict_list = m.list([m.dict() for _ in range(1)])

        a3c_workers = [A3CWorker(global_ep, res_dict, rew_queue,
                                 agent_input_dict_list[i], agent_return_dict_list[i], i) for i in range(12)]

        [w.start() for w in a3c_workers]
        [w.join() for w in a3c_workers]

        res = res_dict['res_q']
        print('game_result:', res)
        for agent_return_dict in agent_return_dict_list:
            print(agent_return_dict)

        win_rate, tie_rate, lose_rate, step_game = self.calculate_statistics(agent_return_dict_list)
        print(win_rate, tie_rate, lose_rate, step_game)
        self.return_dict[int(self.nth_pbt_process)] = [win_rate, step_game]

    @staticmethod
    def calculate_statistics(agent_return_dict_list):
        count_win = 0
        count_tie = 0
        count_lose = 0
        count_sum_game = 0
        count_sum_step = 0
        # calculate the count number of games wins or loses, and the count number of steps,
        # and return the average number of them
        for agent_return_dict in agent_return_dict_list:
            for idx in range(len(agent_return_dict['game_result'])):
                count_sum_game += 1
                count_sum_step += agent_return_dict['game_steps'][idx]
                if agent_return_dict['game_result'][idx] == "Win":
                    count_win += 1
                if agent_return_dict['game_result'][idx] == "Tie":
                    count_tie += 1
                if agent_return_dict['game_result'][idx] == "Lose":
                    count_lose += 1

        assert count_sum_game != 0, 'No Game Counted!!!'
        print("Count Sum games : ", count_sum_game)
        return count_win / count_sum_game, count_tie / count_sum_game, count_lose / count_sum_game, count_sum_step / count_sum_game


if __name__ == '__main__':
    mp.set_start_method("spawn")
    manager = Manager()
    return_dict = manager.dict()
    input_dict = manager.dict()
    uniq_name = ""#sys.argv[1]
    instance = TestFunc(uniq_name, 0, input_dict, return_dict)
    instance.run()
