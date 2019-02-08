import tensorflow as tf
import numpy as np
import collections
from collections import namedtuple
import itertools
def preproc(I):
    """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
    I = I[35:185] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
    I = I[::2,::2,0] # downsample by factor of 2.
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
    return I.astype(np.float).ravel() # ravel flattens an array and collapses it into a column vector

class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, env, learning_rate=0.001, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(np.float32, shape=[None] + list(preproc(env.observation_space.sample()).shape))
            self.action = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
            self.target = tf.placeholder(dtype=tf.float32, shape=[None], name="target")
            
            n_hidden = 200
            # This is just table lookup estimator
            self.hidden_layer = tf.contrib.layers.fully_connected(
                inputs = self.state,
                num_outputs=n_hidden,
                weights_initializer=tf.contrib.layers.xavier_initializer() ) 
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs = self.hidden_layer,
                num_outputs=2, #only use action 2 and 3 (up and down)
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer())

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.concated = tf.concat([self.action_probs, tf.reshape(tf.cast(self.action - 2, tf.float32),[-1,1])],axis=1)
            self.picked_action_prob = tf.map_fn(lambda t: tf.gather(t,tf.cast(tf.gather(t,2),tf.int32)),self.concated)
            mean, var = tf.nn.moments(self.target, [0], keep_dims=True)
            self.batch_norm = tf.div(tf.subtract(self.target, mean), tf.sqrt(var))
            # Loss and train op
            self.loss = -tf.reduce_mean(tf.log(self.picked_action_prob) * self.batch_norm)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,decay=0.99)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = np.expand_dims(state,0)
        state = state.astype(np.float32, copy=False)
        return sess.run(self.action_probs, { self.state: state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session() 
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        print(sess.run(self.target,feed_dict))
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, env, learning_rate=0.001, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(np.float32, shape=[None] + list(preproc(env.observation_space.sample()).shape))
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            n_hidden = 200
            # This is just table lookup estimator
            self.hidden_layer = tf.contrib.layers.fully_connected(
                inputs = self.state,
                num_outputs=n_hidden,
                #activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            
            # This is just table lookup estimator
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.hidden_layer,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.reduce_mean(tf.squared_difference(self.value_estimate, self.target))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = np.expand_dims(state,0)
        state = state.astype(np.float32, copy=False)
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss




def reinforce(env, policy_estimator, value_estimator, num_episodes, discount_factor=1.0):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    def discount_rewards(r):
        """ take 1D float array of rewards and compute discounted reward """
        """ this function discounts from the action closest to the end of the completed game backwards
        so that the most recent action has a greater weight """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)): # xrange is no longer supported in Python 3
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * discount_factor + r[t]
            discounted_r[t] = running_add
        return discounted_r
    
    # Keeps track of useful statistics
    EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    saver = tf.train.Saver()
    sess = tf.get_default_session()
    model = './monte_carlo_image_batch_RMS_batch_norm.ckpt'
    #saver.restore(sess,model)
    states = []
    actions = []
    rewards_ = []
    baseline_values = []

    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        cur_x = preproc(state)
        prev_x =  np.zeros(cur_x.shape)
        state = cur_x - prev_x
        episode = []
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            action_probs = policy_estimator.predict(state)
            action = np.random.choice([2,3], p=action_probs)
            observation, reward, done, _ = env.step(action)
            #env.render()
            # preprocess the observation, set input to network to be difference image
            cur_x = preproc(observation)
            # we take the difference in the pixel input, since this is more likely to account for interesting information
            # e.g. motion
            next_state = cur_x - prev_x
            #print(next_state.shape)
            prev_x = cur_x
            
           # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
            # sys.stdout.flush()

            if done:
                break
                
            state = next_state
    
        # Go through the episode and make policy updates
        #for t, transition in enumerate(episode):
        #    # The return after this timestep
        #    total_return = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
        #    # Calculate baseline/advantage
        #    baseline_value = estimator_value.predict(transition.state)            
        #    advantage = total_return - baseline_value
        #    # Update our value estimator
        #    estimator_value.update(transition.state, total_return)
        #    # Update our policy estimator
        #    estimator_policy.update(transition.state, advantage, transition.action)
        states += list(map(lambda t: t.state,episode))
        actions += list(map(lambda t: t.action,episode))
        rew_ = list(map(lambda t: t.reward,episode))
        rewards_ += list(discount_rewards(r=np.array(rew_)))
        
        batch_size =  10
        if i_episode % batch_size == 0 and i_episode>0:
            # Print out which step we're on, useful for debugging.
            #print("\rStep {} @ Episode {}/{} ({})".format(
            #    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
            # Update our policy estimator
            #advantage = rewards_ - baseline_values
            #advantages = [a_i - b_i for a_i, b_i in zip(rewards_, baseline_values)]
            #loss_policy = estimator_policy.update(states, advantages, actions)
            loss_policy = policy_estimator.update(states, rewards_, actions)
            #loss_values = value_estimator.update(states, rewards_)
            states = []
            actions = []
            rewards_ = []
            baseline_values = []
            print("loss policy: {}".format(loss_policy))
               
        saver.save(sess,model)
        
    return stats

if __name__ == "__main__":
    import gym
    tf.reset_default_graph()
    env = gym.make('Pong-v0')

    global_step = tf.Variable(0, name="global_step", trainable=False)
    policy_estimator = PolicyEstimator(env,learning_rate=0.01)
    #value_estimator = ValueEstimator(env,learning_rate=0.01)
    value_estimator = None
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        stats = reinforce(env, policy_estimator, value_estimator, 10000, discount_factor=.99)
