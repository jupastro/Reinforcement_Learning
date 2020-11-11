def create_random_policy(grid):
##### Inputs:
# A grid_world class created by the grid = GridWorld(p) where p is the probability of a non-deterministic action
##### Output:
# Creates a random_policy assigning 0.25 probability to each option
  possible_actions=['nr','ea','so', 'we']# it is possible to move up,left,right and south
  n=len(possible_actions)
  probability=1/n #probability equal to all the values
  policy = np.zeros((len(grid.locs),4))
  for key in range(0, len(grid.locs)):
    p = np.zeros((1,4))
    for action in range(0, n):
      policy[key,action] =probability
  return policy
  
def firsvisit_td_onpolicy(grid,iterations,epsilon,alpha,gamma,probal=1,draw=False):
  #SARSA on-policy method for calculating the optimal Q(s,a) and policy, to obtain the estimated optimal Value state function you must np.sum(Q*P,axis=1)
  ####Input:
  #grid:gridworld object
  #iterations:number of complete episodes to perform
  #epsilon: factor used for computing a epsilon-greedy policy useful to explore different paths
  #alpha:weighting factor of the new_samples, a good way of initialising is to use 1/iterations
  #gamma:decaying factor of our returns
  #draw: indicated whether to plot or not the learning curve 
  
  ####Output:
  #Q,policy,Returns,EpisodeQ,EpisodeP
  #Q: estimated optimal state,action function
  #policy:estimated optimal policy
  #Returns:list of  total returns per episode
  #EpisodeQ: Q function obtained after each episode
  #EpisodeP: P optimal policy obtained after each episode
  Q=np.zeros((29,4))#arbitrary Q(s,a)
  EpisodeQ=np.zeros((29,4,iterations))
  EpisodeP=np.zeros((29,4,iterations))
  Returns=[]
  

  possible_actions=['nr','ea','so', 'we']

  policy=create_random_policy(grid)# arbitrary random policy to start with

  for x in range(0,iterations):#Number of episodes
    states=[]
    rewards=[]
    actions=[]
    current_loc=random.choice(grid.locs)#random initialization
    current_state=grid.loc_to_state(current_loc,grid.locs)
    states.append(current_state)
    finish=False
    action=np.random.choice(range(0,4),p=policy[current_state,:])#first choice of the action
    R=0
    alpha=alpha*0.99 #Decaying alpha factor
    epsilon=0.99*epsilon #Decaying epsilon factor
    while finish==False:
      k=0
      action_name=possible_actions[action]
      others=np.delete(possible_actions,action)
      if np.random.choice([0,1],p=[probal,1-probal])!=0:# for non-deterministic environment  action chosen different result than expected
        action_name=np.random.choice(others)
      next_loc = grid.get_neighbour(current_loc,action_name)
      next_state=grid.loc_to_state(next_loc,grid.locs)
      actions.append(action_name)
      states.append(next_state)
      next_action=np.random.choice(range(0,4),p=policy[next_state,:])
      #Stopping criteria:
      if next_loc in grid.absorbing_locs:
        if next_loc==(1,2):
          rewards.append(10)
          finish=True
        else:
          rewards.append(-100)
          finish=True
      elif k==250:
        finish=True
      else:
        rewards.append(-1)
        k+=1
      #Upload Q(s,a)value with the one-step ahead predictor for each step
      Q[current_state,action]=Q[current_state,action]+alpha*(rewards[k-1]+gamma*Q[next_state,next_action]-Q[current_state,action])
      

      #Upload the policy in each step
      probs=np.zeros((1,4))
      others=epsilon/4
      prob=1-epsilon+others

      max_Q=max(Q[current_state,:])
      A_star=0
      for m in range(0,4):
        if Q[current_state,m]==max_Q:
            A_star=m
        policy[current_state,:]=np.ones_like((1,1,1,1))*others
        policy[current_state,A_star]=prob
      action=next_action
      current_loc=next_loc
      current_state=next_state
    m=0
    R=0
    for w in np.flip(rewards):
      R+=(gamma**(m))*w# sum up all the next values but not the past ones
      m+=1
    Returns.append(R)
    EpisodeQ[:,:,x]=Q
    EpisodeP[:,:,x]=policy
  if draw==True:
    import matplotlib.pyplot as plt

    plt.plot(range(0,len(Returns)),Returns)
    plt.ylabel('Total Reward')
    plt.xlabel('Episodes')
    plt.show()
  return Q,policy,Returns,EpisodeQ,EpisodeP
def average_td_learning(iterations,epsilon,alpha,gamma,repetitions,probal=1):
####Input:
#repetitions:number of runs to average with
#grid:gridworld object
#iterations:number of complete episodes to perform
#epsilon: factor used for computing a epsilon-greedy policy useful to explore different paths
#alpha:weighting factor of the new_samples, a good way of initialising is to use 1/iterations
#gamma:decaying factor of our returns
  ####Output:
  #Q,policy,Returns,EpisodeQ,EpisodeP
  #Q: estimated optimal state,action function
  #policy:estimated optimal policy
  #Returns:list of  total returns per episode
  #EpisodeQ: Q function obtained after each episode
  #EpisodeP: P optimal policy obtained after each episode
    grid=GridWorld(p1=0.4)
    avR=np.zeros((repetitions,iterations))
    avQ=np.zeros((29,4,repetitions))
    avPolicy=np.zeros((29,4,repetitions))
    EpQav=np.zeros((29,4,iterations))
    EpPav=np.zeros((29,4,iterations))
    for i in range(0,repetitions):
     if (i + 1) % 5 == 0:
             print("\rEpisode {}/{}.".format(i + 1, repetitions), end="")
             sys.stdout.flush()
     avQ[:,:,i],avPolicy[:,:,i],avR[i,:],EpisodeQ,EpisodeP=firsvisit_td_onpolicy(grid,iterations,epsilon,alpha,gamma,probal)
     EpQav+=EpisodeQ/repetitions
     EpPav+=EpisodeP/repetitions
    R=np.mean(avR,axis=0) 
    stdR=np.std(avR,axis=0)
    Q=np.mean(avQ,axis=2)
    policy=np.mean(avPolicy,axis=2)
    import matplotlib.pyplot as plt

    plt.plot( range(0,len(R)), R, color='skyblue',label='average',linewidth=2)
    plt.plot( range(0,len(R)), R+stdR, color='r',label='+std',linewidth=0.5,linestyle='dashed')
    plt.plot( range(0,len(R)), R-stdR, color='r',label='-std',linewidth=0.5,linestyle='dashed')
    plt.legend()
    plt.title('Average learning curve with N='+str(repetitions)+'\n alpha='+str(alpha)+'\n epsilon='+str(epsilon))
    plt.ylabel('Total Averaged Reward')
    plt.xlabel('Episodes')
    plt.show() 
    return R,policy,Q,EpQav,EpPav
