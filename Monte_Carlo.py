
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

def random_trace_generation(grid,policy,probal=1):
# Creates a complete episode from a given policy 
##### Inputs:
# grid:Gridworld object
# policy: a pre-defined policy containing the probabilities of each state to perform a certain action, the dimensions must be (number_states,number_actions)
# probal:factor affecting the deterministic election of the chosen action, if it's set to 1 fully deterministic environment, for p<1 non-deterministic inducing noise in the environment
#### Outputs:states,rewards,actions,elected_actions
# states: list of states observed during the trace
# rewards: list of rewards obtained during the trace(is n-1 size than the states(n) list)
# actions: list of actions performed during the trace
# elected_actions: list of elected_actions according to our policy, for probal=1 elected_actions=actions but otherwise it doesn't have to

  possible_actions=['nr','ea','so', 'we']
  states=[]
  rewards=[]
  actions=[]
  elected_actions=[]
  current_loc=random.choice(grid.locs)# random initialization of the first state
  current_state=grid.loc_to_state(current_loc,grid.locs)#changing the state matrix description to a single number related to its position e.g (1,2)=state 2
  states.append(current_state)# we add it to our trace
  finish=True
  k=0
  while finish: # keep iterating if we don't get to a final or absorbing state 
    chosen=np.random.choice(range(0,4),p=policy[current_state,:])# perform an action based in the probabilities of our policy
    action=possible_actions[chosen]
    elected_actions.append(action)
    others=np.delete(possible_actions,chosen)
    if np.random.choice([0,1],p=[probal,1-probal])!=0:#if our system is non-deterministic (p<1)the action_chosen would not always be performed 
      action=np.random.choice(others)
    next_loc = grid.get_neighbour(current_loc,action)
    current_loc=next_loc
    current_state=grid.loc_to_state(next_loc,grid.locs)
    actions.append(action)
    states.append(current_state)
  
    if next_loc in grid.absorbing_locs:#Modify in order to include your own absorbing_states and rewards
      if next_loc==(1,2):
        rewards.append(10)
        finish=False
      else:
        rewards.append(-100)
        finish=False
    elif k==2000:
     finish=False
    else:
      rewards.append(-1) #penalisation for movement
      k+=1
  return states,rewards,actions,elected_actions
  
def firsvisit_mc_onpolicy(grid,iterations,epsilon,alpha,gamma,probal=1,draw=False):
  #Monte-Carlo on-policy method for calculating the optimal Q(s,a) and policy, to obtain the estimated optimal Value state function you must np.sum(Q*P,axis=1)
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
  policy=create_random_policy(grid)# arbitrary soft policy to start with
  for x in range(0,iterations):#Repeat forever
    #alpha=alpha*0.99 #Uncomment this line to perform a decaying alpha(not very useful in MC as its convergence is more far-sighted and the decaying factor would injure the long term learning)
    epsilon=epsilon*0.99 #decaying epsilon factor to end up in an almost totally greedy policy
    states,rewards,real_actions,actions=random_trace_generation(grid,policy,probal)
    #generate a a trace of x_samples(episodes)using policy
    trace=[]
    for i in range(0,len(actions)):
      trace.append((states[i],possible_actions.index(actions[i])))#pair state,action
    #for each state appearing for the first time
    seen_states=[]
    k=0
    R=0
    m=0
    for w in np.flip(rewards[:]):
      R=(gamma**(m))*w+R# sum up all the next values, total reward computed as the backwards discounted Return per episode
      m+=1
    Returns.append(R)
    for i in trace:
     
      if i not in seen_states:
        seen_states.append(i)
        #Calculate G as the Qtemp
        G=0
        n=k
        for r in rewards[k:]:
          G=(gamma**(n-k))*r+G# sum up all the next values but not the past ones
          n+=1
        #Upload Q(s,a)value 
        Q[i]=Q[i]+alpha*(G-Q[i])
      k+=1
    EpisodeQ[:,:,x]=Q
    EpisodeP[:,:,x]=policy 
  #Upload the policy

    probs=np.zeros((29,4))
    others=epsilon/4
    prob=1-epsilon+others
    
    for s in range(0,len(Q)):
    
      max_Q=max(Q[s,:])
      max_Q
      for m in range(0,4):
        if Q[s,m]==max_Q:
           A_star=m
      probs[s,:][:]=np.ones_like((1,1,1,1))*others
      probs[s,A_star]=prob
     
    policy=probs
   
  if draw==True:
    import matplotlib.pyplot as plt
    plt.plot(range(0,len(Returns)),Returns)
    plt.ylabel('Total Discounted Reward')
    plt.xlabel('Episodes')
    plt.title('Learning Curve of the MC agent')
    plt.show()
  return Q,policy,Returns,EpisodeQ,EpisodeP
  
def average_montecarlo_learning(iterations,epsilon,alpha,gamma,probal,repetitions):
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
    
    if (i + 1) % 2 == 0:
            print("\r Iteration {}/{}.".format(i + 1, repetitions), end="")
            sys.stdout.flush()
    avQ[:,:,i],avPolicy[:,:,i],avR[i,:],EpisodeQ,EpisodeP=firsvisit_mc_onpolicy(grid,iterations,epsilon,alpha,gamma,probal)
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
  

