import gymnasium as gym
from agent import DDPG

env = gym.make('Pendulum-v1', max_episode_steps=200, autoreset=True)

score = 0.0
print_interval = 20

agent = DDPG(env.action_space.shape[0], 
             env.observation_space.shape[0], 
             env.action_space.high,
             batch_size=256,
             dr3_coeff=1e-3,
             use_mirror=False)

for n_epi in range(1000):
    s, _ = env.reset()
    done = False

    count = 0
    while count < 200 and not done:
        a = agent.act(s, 1)
        s_prime, r, done, truncated, info = env.step(a)
        agent.replay_buffer.put((s, a, r/100.0, s_prime, done))
        score += r
        s = s_prime
        count += 1

    if agent.replay_buffer.size > 2000: agent.train()

    if n_epi%print_interval==0 and n_epi!=0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
        score = 0.0

env.close()
