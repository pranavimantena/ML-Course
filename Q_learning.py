import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make('MountainCar-v0')
env.reset()
def QLearn(env, learn, discount, epsilon, min_eps, epis):
    num_states = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1
    Q = np.random.uniform(low = -1, high = 1, 
                          size = (num_states[0], num_states[1], 
                                  env.action_space.n))
    reduc = (epsilon - min_eps)/epis
    reward_list = []
    total_reward_list = []
    for i in range(epis):
    
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
    
        while done != True:   
            
            if i >= (epis - 20):
                env.render()
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]]) 
            else:
                action = np.random.randint(0, env.action_space.n)
                
            
            state2, reward, done, info = env.step(action) 
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                
            
            else:
                delta = learn*(reward + 
                                 discount*np.max(Q[state2_adj[0], 
                                                   state2_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
                                     
           
            tot_reward += reward
            state_adj = state2_adj
        
        
        if epsilon > min_eps:
            epsilon -= reduc
        
        
        reward_list.append(tot_reward)
        
        if (i+1) % 100 == 0:
            total_reward = np.sum(reward_list)
            total_reward_list.append(total_reward)
            reward_list = []
            
    env.close()
    
    return total_reward_list


rewards = QLearn(env, 0.2,1, 0.8, 0, 8000)
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward vs Episodes')  
plt.savefig('Plot.jpg')
plt.close() 
