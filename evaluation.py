import time
import numpy as np
import laserhockey.hockey_env as h_env

from optparse import OptionParser
from agent import DDPG

parser = OptionParser()
parser.add_option("-c", "--critic1", dest="critic1_path", default=None, help="add the critic network of the first player", metavar="C1_PATH")
parser.add_option("--critic2", dest="critic2_path", default="./tournament_models/critic/checkpoint-194000.pt",
                  help="add the critic network of the second player", metavar="C2_PATH")
parser.add_option("-a", "--actor1", dest="actor1_path", default=None, help="add the critic network of the first player", metavar="A1_PATH")
parser.add_option("--actor2", dest="actor2_path", default="./tournament_models/actor/checkpoint-194000.pt",
                  help="add the critic network of the second player", metavar="A2_PATH")
parser.add_option("-w", action="store_true", dest="watch")
parser.add_option("-n", "--number_of_duels", dest="num_duels", default=5, help="set the number of duels")


def watch_duels(p1, p2, n_duels):
    for test_games in range(n_duels):
        env = h_env.HockeyEnv()
        
        s1, info = env.reset()
        s2 = env.obs_agent_two()
        _ = env.render()

        for t in range(600):
            time.sleep(.025)
            env.render(mode="human")
            a1 = p1.act(s1, 0)
            a2 = p2.act(s2, 0)   
            s1, r, done, t, info = env.step(np.hstack([a1, a2])) 
            s2 = env.obs_agent_two()
            if info['winner'] != 0:
                print(info['winner'])
            if done: break
        env.close()

def evaluate(p1, p2, n_duels=1000, use_log=False, logger=None):
    winners = []
    for test_game in range(n_duels):
        env = h_env.HockeyEnv()

        s1, info = env.reset()
        s2 = env.obs_agent_two()

        for t in range(250):
            a1 = p1.act(s1)
            a2 = p2.act(s2)
            s1, r, done, t, info = env.step(np.hstack([a1, a2]))
            s2 = env.obs_agent_two()
            if done:
                winners.append(info['winner'])
                break
        env.close()
    win_rate = (np.array(winners) == 1).astype(int).sum() / n_duels
    loss_rate = (np.array(winners) == -1).astype(int).sum() / n_duels
    if use_log:
        if logger == None: return win_rate
        logger.register("Win rate : {:.2f}%".format(win_rate * 100))
        # logger.register("Tie rate : {:.2f}%".format((1 - win_rate - loss_rate) * 100))
        # logger.register("Loss rate : {:.2f}%".format(loss_rate * 100))
    else:
        print("Win rate : {:.2f}%".format(win_rate * 100))
        print("Tie rate : {:.2f}%".format((1 - win_rate - loss_rate) * 100))
        print("Loss rate : {:.2f}%".format(loss_rate * 100))
    return win_rate
        
def main():
    (options, args) = parser.parse_args()
    critic1_path = options.critic1_path
    critic2_path = options.critic2_path
    actor1_path = options.actor1_path
    actor2_path = options.actor2_path
    num_duels = int(options.num_duels)
    watch = False or options.watch

    as_dim = 4 
    os_dim = 18
    as_max = np.array([1., 1., 1., 1.])
    
    player1 = DDPG(as_dim, os_dim, as_max)
    player2 = DDPG(as_dim, os_dim, as_max)
    
    # load player 1
    if critic1_path == None or actor1_path == None:
        raise Exception('You have to specify the paths for the actor and critic of the first player!')

    print(actor1_path)
    player1.load_models(actor1_path, critic1_path)

    # load player 2
    if actor2_path == 'weak' or critic2_path == 'weak':
        player2 = h_env.BasicOpponent(weak=True)
    elif actor2_path == 'strong' or critic2_path == 'strong':
        player2 = h_env.BasicOpponent(weak=False)
    else:
        player2.load_models(actor2_path, critic2_path)

    if watch:
        # watch duels
        watch_duels(player1, player2, num_duels)
    else:
        evaluate(player1, player2, num_duels)

if __name__ == "__main__":
    main()
