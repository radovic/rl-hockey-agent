from optparse import OptionParser
from trainer import Trainer
import json

parser = OptionParser()
parser.add_option("--num_episodes", dest="num_episodes", default=25000, help="number of episodes", metavar="N")
parser.add_option("--num_game_ts", dest="num_game_ts", default=250, help="number of maximum time steps per episode", metavar="N")
parser.add_option("--num_episodes_autosave", dest="num_episodes_autosave", default=1000, help="number of episodes after which autosave is performed", metavar="N")
parser.add_option("--num_opponent_update", dest="num_opponent_update", default=50, help="number of episodes after which we query the POB for the new opponent", metavar="N")
parser.add_option("--num_beginner_games", dest="num_beginner_games", default=2500, help="number of initial episodes in which it is trained using a simple curriculum", metavar="N")
parser.add_option("--wr_opponent_thresh", dest="wr_opponent_thresh", default=0.95, help="number of maximum time steps per episode", metavar="TRSH")
parser.add_option("--min_buffer_size_training", dest="min_buffer_size_training", default=2000, help="the size of replay buffer required to start training", metavar="N")
parser.add_option("--evaluation_interval", dest="evaluation_interval", default=500, help="number of episodes after which the evaluation on weak agent should start", metavar="N")
parser.add_option("--evaluation_ts", dest="evaluation_ts", default=200, help="number of time steps per evaluation", metavar="N")
parser.add_option("--print_interval", dest="print_interval", default=100, help="number of episodes after which the average returns in last N episodes should be printed", metavar="N")

# agent params
parser.add_option("--lr_actor", dest="lr_actor", default=3e-4, help="learning rate for the actor", metavar="LR")
parser.add_option("--lr_critic", dest="lr_critic", default=3e-4, help="learning rate for the critic", metavar="LR")
parser.add_option("--discount_rate", dest="discount_rate", default=0.95, help="discount rate for the problem", metavar="LR")
parser.add_option("--batch_size", dest="batch_size", default=256, help="batch size of the replay buffer samples", metavar="BS")
parser.add_option("--max_buffer_size", dest="max_buffer_size", default=1000000, help="maximum buffer size", metavar="MBS")
parser.add_option("--soft_update_ts", dest="soft_update_ts", default=1, help="fit iterations berween soft updates", metavar="N")
parser.add_option("--tau", dest="tau", default=.005, help="soft update coefficient", metavar="F")
parser.add_option("--dr3_coeff", dest="dr3_coeff", default=0, help="coefficient of DR3 regularizer", metavar="F")
parser.add_option("--disable_mirror", dest="disable_mirror", action="store_true", default=False, help="disable mirror exploit")
parser.add_option("--agent_use_resets", dest="use_resets", action="store_false", default=True, help="use resets")


def main():
    (options, args) = parser.parse_args()
    config = {
        "num_episodes": int(options.num_episodes),
        "num_game_ts": int(options.num_game_ts),
        "num_episodes_autosave": int(options.num_episodes_autosave),
        "num_opponent_update": int(options.num_opponent_update),
        "num_beginner_games": int(options.num_beginner_games),
        "wr_opponent_thresh": float(options.wr_opponent_thresh),
        "min_buffer_size_training": int(options.min_buffer_size_training),
        "evaluation_interval": int(options.evaluation_interval),
        "evaluation_ts": int(options.evaluation_ts),
        "print_interval": int(options.print_interval),
        "replay_ratio": int(options.replay_ratio),
        "use_mirror": options.disable_mirror,
        "use_resets": options.use_resets,
        "lr_actor": float(options.lr_actor),
        "lr_critic": float(options.lr_critic),
        "discount_rate": float(options.discount_rate),
        "batch_size": int(options.batch_size),
        "max_buffer_size": int(options.max_buffer_size),
        "soft_update_ts": int(options.soft_update_ts),
        "tau": float(options.tau),
        "dr3_coeff": float(options.dr3_coeff)
        }
    trainer = Trainer()
    trainer.start_new_training(config)


if __name__ == "__main__":
    main()
