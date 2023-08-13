import os
import json
import copy
import torch
import collections
import numpy as np
import laserhockey.hockey_env as h_env

from agent import DDPG
from logger import Logger
from evaluation import evaluate
from utils import PrioritizedOpponentBuffer

class Trainer:
    def __init__(self):
        # code directory
        self.CODE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
        # output directory
        self.OUT_DIR_PATH = os.path.join(self.CODE_DIR_PATH, 'out')
        os.makedirs(self.OUT_DIR_PATH, exist_ok=True)
        # training log file which contains info about the trainings which were done
        # in the past
        self.TRAIN_LOG_FILE = os.path.join(self.OUT_DIR_PATH, 'train_logs.json')
        if not os.path.exists(self.TRAIN_LOG_FILE):
            log_file = open(self.TRAIN_LOG_FILE, 'w')
            json.dump([], log_file, indent=1)
            log_file.close()
        # get the number of the next training run
        training_dirs = [f.name for f in os.scandir(self.OUT_DIR_PATH) if f.is_dir()]
        current_nums = []
        for td in training_dirs:
            if len(td) < 4: continue
            if td.startswith('OUT-') and td[-4:].isnumeric():
                current_nums.append(int(td[-4:]))
        if len(current_nums) < 1: self.CURRENT_INDEX = 0
        else: self.CURRENT_INDEX = int(np.max(current_nums) + 1)
        if self.CURRENT_INDEX == 9999:
            raise Exception('Limit of 9999 files in out reached!')

        
    def train(self, config):
        # initialize the environment
        env_modes = [h_env.HockeyEnv.TRAIN_DEFENSE,
                     h_env.HockeyEnv.TRAIN_SHOOTING,
                     h_env.HockeyEnv.NORMAL]
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)

        # initialize the metrics
        episode_returns = collections.deque([0] * 10000, 10000)

        # initialize the configs
        num_episodes = config['num_episodes']
        num_game_ts = config['num_game_ts']
        num_episodes_autosave = config['num_episodes_autosave']
        num_opponent_update = config['num_opponent_update']
        num_beginner_games = config['num_beginner_games']
        wr_opponent_thresh = config['wr_opponent_thresh']
        min_buffer_size_training = config['min_buffer_size_training']
        evaluation_interval = config['evaluation_interval']
        evaluation_ts = config['evaluation_ts']
        print_interval = config['print_interval']
        agent_use_resets = config['agent_use_resets']
        
        # initialize the main player
        main_player = DDPG(int(env.action_space.shape[0] / 2), 
                           int(env.observation_space.shape[0]), 
                           env.action_space.high[:4],
                           use_mirror=config['use_mirror'],
                           lr_actor=config['lr_actor'],
                           lr_critic=config['lr_critic'],
                           discount_rate=config['discount_rate'],
                           batch_size=config['batch_size'],
                           max_buffer_size=config['max_buffer_size'],
                           soft_update_ts=config['soft_update_ts'],
                           tau=config['tau'],
                           dr3_coeff=config['dr3_coeff'],
                           use_resets=agent_use_resets,
                           save_path=self.CURRENT_MODELS_DIR,
                           logger=self.CURRENT_LOG
                        )

        # initialize the opponent buffer
        weak_opponent = h_env.BasicOpponent(weak=True)
        strong_opponent = h_env.BasicOpponent(weak=False) 
        POB = PrioritizedOpponentBuffer(B=1)
        POB.add_opponent(weak_opponent)

        # initialize the opponent
        opponent_idx, opponent = 0, weak_opponent

        # episodes loop
        for episode in range(1, num_episodes + 1):
            # change env mode if necessary
            if episode < 2 * num_beginner_games:
                if episode == num_beginner_games:
                    env = h_env.HockeyEnv(mode=env_modes[1])
            else:
                m = np.random.randint(3)
                env = h_env.HockeyEnv(mode=env_modes[m])
                
            # update opponent if necessary
            if POB.K > 1 and episode % num_opponent_update == 0:
                POB.register_outcome(opponent_idx,
                                    -np.mean(list(episode_returns)[-num_opponent_update:]))
                opponent_idx, opponent = POB.get_opponent()
            
            s1, info = env.reset()
            s2 = env.obs_agent_two()
            episode_return = 0

            t = 0
            for t in range(num_game_ts):
                a1 = main_player.act(s1, 1)
                a2 = opponent.act(s2)
                s_prime, r, done, _, info = env.step(np.hstack([a1, a2]))
                
                main_player.replay_buffer.put((s1, a1, r, s_prime, done))
                episode_return += r
                s1 = s_prime
                s2 = env.obs_agent_two()
                if done: break

            episode_returns.append(episode_return)

            if main_player.replay_buffer.size > min_buffer_size_training: 
                main_player.train()
                
            if episode % print_interval == 0 and episode != 0:
                avg_return = np.mean(np.array(list(episode_returns)[-print_interval:]))
                self.CURRENT_LOG.register(
                    "# of episode :{}, avg returns : {:.1f}\n".format(episode, avg_return))
                self.CURRENT_LOG.update_metric('average_returns', episode, avg_return)

            if episode % evaluation_interval == 0 and episode != 0:
                wr_weak = evaluate(main_player, weak_opponent, evaluation_ts, True, self.CURRENT_LOG)
                self.CURRENT_LOG.update_metric('win_rate_weak', episode, wr_weak)
                update_opponent = wr_weak >= wr_opponent_thresh
                if POB.K > 1 and update_opponent:
                    for idx, op in enumerate(POB.opponents[1:]):
                        curr_wr = evaluate(main_player, op, evaluation_ts, True)
                        if idx == 0:
                            self.CURRENT_LOG.update_metric('win_rate_strong', episode, curr_wr)
                        else:
                            self.CURRENT_LOG.update_metric('win_rate_L{idx}', episode, curr_wr)
                        update_opponent &= (curr_wr >= wr_opponent_thresh)
                        if not update_opponent: break
                # add new opponent if the win rate is over some threshold
                # for all current opponents
                if update_opponent:
                    self.CURRENT_LOG.register(f"Episode {episode}: Adding new opponent!\n")
                    if POB.K == 1:
                        POB.add_opponent(strong_opponent)
                    else:
                        POB.add_opponent(copy.deepcopy(main_player))
                    main_player.save_model(f'level-{POB.K - 1}', self.CURRENT_MODELS_DIR)
                
            if episode % num_episodes_autosave == 0:
                main_player.save_model('checkpoint-' + str(episode),
                                       self.CURRENT_MODELS_DIR)

            env.close()

    def start_new_training(self, config):
        # create the directory and update the training log
        new_dir_name = 'OUT-' + ('0000' + str(self.CURRENT_INDEX))[-4:]
        self.CURRENT_TRAIN_DIR = os.path.join(self.OUT_DIR_PATH, new_dir_name)
        os.makedirs(os.path.join(self.CURRENT_TRAIN_DIR))
        new_entry = {"id": self.CURRENT_INDEX,
                     "config": config }
        json_file_reader = open(self.TRAIN_LOG_FILE, 'r')
        list_of_trainings = json.load(json_file_reader)
        json_file_reader.close()
        list_of_trainings.append(new_entry)
        json_file_writer = open(self.TRAIN_LOG_FILE, 'w')
        json.dump(list_of_trainings, json_file_writer, indent=1)
        json_file_writer.close()
        # initialize the log of the new training
        self.CURRENT_LOG = Logger(self.CURRENT_TRAIN_DIR)
        # initialize the models directory
        self.CURRENT_MODELS_DIR = os.path.join(self.CURRENT_TRAIN_DIR, 'models')
        # start training
        self.train(config)
        self.CURRENT_LOG.close()
