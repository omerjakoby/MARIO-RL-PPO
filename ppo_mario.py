
import tensorflow as tf
import gym
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation,FrameStack,ResizeObservation
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,BatchNormalization, Normalization
from keras import layers
from keras import initializers
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.utils import plot_model
import random
from keras.models import Model  # For building the model (if not already built)
from keras.layers import Input, Dense  # Example layers (replace with your actual layers)

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions[0], observation_dimensions[1], observation_dimensions[2]), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1


    def finish_trajectory(self, last_value=0):
            # Finish the trajectory by computing advantage estimates and rewards-to-go


            # Get the slice of the trajectory from the starting index to the current pointer
            path_slice = slice(self.trajectory_start_index, self.pointer)

            # Append the last value (estimated value of the next state) to the rewards and values buffers
            rewards = np.append(self.reward_buffer[path_slice], last_value)
            values = np.append(self.value_buffer[path_slice], last_value)

            # Compute the temporal differences (TD) between the rewards and values
            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]



            # Compute the advantage estimates using the discounted cumulative sums of the deltas
            # The advantages are calculated using the Generalized Advantage Estimation (GAE) method
            # GAE combines the TD residuals with a discount factor (gamma) and a lambda parameter
            self.advantage_buffer[path_slice] = discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )


            # Compute the rewards-to-go (discounted cumulative rewards) for each step in the trajectory
            # The rewards-to-go are the target values for training the value function
            self.return_buffer[path_slice] = discounted_cumulative_sums(
                rewards, self.gamma
            )[:-1]


            # Update the trajectory starting index to the current pointer
            # This prepares the buffer for the next trajectory
            self.trajectory_start_index = self.pointer


    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / (advantage_std+1e-8)
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def cnn(x, num_actions,activation=tf.tanh):
    x = x/255.0
    x = layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation= activation,padding = 'same')(x)
    x = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation= activation,padding = 'same')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation= activation,padding = 'same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation= activation)(x)
    logits = layers.Dense(num_actions, name='policy_logits',activation= None)(x)
    value = layers.Dense(1, name='value', activation= None)(x)
    return logits, value

def cnn_bn(x, num_actions,activation='linear'):
    x= x/255.0
    x = layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation= activation,padding = 'same')(x)
    #x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation= activation,padding = 'same')(x)
    #x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation= activation,padding = 'same')(x)
    #x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation= 'relu')(x)
    logits = layers.Dense(num_actions, name='policy_logits',activation= None)(x)
    value = layers.Dense(1, name='value', activation= None)(x)
    return logits, value


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Sample action from actor
@tf.function
def sample_action(observation, seed=42):
    observation = tf.expand_dims(observation, axis=0)  # Add batch dimension
    logits = actor(observation)
    #print(logits)
    action = tf.squeeze(tf.random.categorical(logits, 1, seed = seed), axis=1)
   # print(action)
    return logits, action

@tf.function
def train_value_and_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer,return_buffer, epoch_index):
    with tf.GradientTape() as tape:

         # Calculate the value loss using the critic network
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)

        # Get the policy logits from the actor network
        actor_policy = actor(observation_buffer)

        # Calculate the ratio of new probabilities to old probabilities
        ratio = tf.exp(
            logprobabilities(actor_policy, action_buffer)
            - logprobability_buffer
        )
        # Calculate the linearly decaying clip ratio based on the current epoch
        clip_ratio_linear_decay = clip_ratio * (1-epoch_index/ epochs)

        # Calculate the clipped advantage using the linearly decaying clip ratio
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio_linear_decay) * advantage_buffer,
            (1 - clip_ratio_linear_decay) * advantage_buffer,
        )

        # Calculate the policy loss using the clipped advantage
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )

        # Calculate the entropy loss to encourage exploration
        entropy_loss = -tf.reduce_sum(- tf.nn.softmax(actor_policy) * tf.nn.log_softmax(logits), axis=1)

        # Calculate the total loss as a weighted sum of value loss, policy loss, and entropy loss
        loss = 0.01* value_loss + policy_loss + 0.01*entropy_loss

    # Get the trainable variables from both the critic and actor networks
    variables = (critic.trainable_variables + actor.trainable_variables)

    # Calculate the gradients of the loss with respect to the variables
    grads = tape.gradient(loss, variables)

    # Clip the gradients by global norm to prevent exploding gradients
    clipped_grads, _ = tf.clip_by_global_norm(grads, 10.0)

    # Apply the clipped gradients to update the network variables using the policy optimizer
    policy_optimizer.apply_gradients(zip(clipped_grads, variables))

#Hyperparameters of the PPO algorithm
steps_per_epoch = 4480
batch_size = 320
epoch_start=250
epochs = 1250
num_actors = 8
save_interval = 50
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 2.5e-5
policy_final_learning_rate = 0
train_policy_iterations = 5
train_value_iterations = 5
lam = 0.95
target_kl = 0.015 # was-0.01
img_size= 96
# True if you want to render the environment
render = True

ENV_NAME = 'SuperMarioBros-1-1-v0'
env = []
for num_actor in range(num_actors):
    env_ = gym.make(ENV_NAME)
    env_ = JoypadSpace(env_, COMPLEX_MOVEMENT)
    env_.seed(seed =(num_actor+7)*43)
    env_ = GrayScaleObservation(env_, keep_dim=True)
    env_ = ResizeObservation(env_, shape=img_size)
    env_ = FrameStack(env_, num_stack=4)
    num_actions = env_.action_space.n
    env.append(env_)


observation_dimensions = (img_size, img_size,4)
buffer = Buffer(observation_dimensions, steps_per_epoch*num_actors)
# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(img_size, img_size,4), dtype=tf.float32)
logits, value = cnn_bn(observation_input, num_actions)
value= tf.squeeze(value,axis=1)


actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
# Initialize the policy and the value function optimizers
lr = policy_learning_rate *(1- epoch_start/epochs*(policy_learning_rate - policy_final_learning_rate))

policy_optimizer = keras.optimizers.Adam(learning_rate=lr)
value_optimizer = keras.optimizers.Adam(learning_rate=lr)
actor.compile(optimizer=policy_optimizer)
critic.compile(optimizer=policy_optimizer)

##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################
#actor= keras.models.load_model(r"actor_model_v550") # actor path absolute
#critic = keras.models.load_model(r"critic_model_v550")#  critic path
##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################


observation = env[0].reset()
done = False

train = "0"
while train != "1" and train != "2":
    print(" To run the model without training press 1")
    print(" To train a model press 2")
    train = input()


if train == "1":
    while True:
        if done:
            env[0].reset()
        observation = tf.transpose(observation, perm=[1, 2, 0, 3])  # Permute dimensions to (height, width, frames, channels)
        observation = tf.reshape(observation, shape=(img_size, img_size, 4))
        logits, action = sample_action(observation)
        observation, reward, done, info = env[0].step(action[0].numpy())
        env[0].render()

# Initialize the observation
#observation = env.reset()

# Iterate over the number of epochs
if train =="2" :

    for epoch in range(epoch_start,epochs):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch

        num_episodes = 0
        max_episode_return = 0
        episode_return_copy = 0
        done_counter = 0
        max_x_location = 0
        total_sum_return = 0

        print(policy_optimizer.learning_rate)


        
        total_sum_return = 0
        for num_actor in range(num_actors):
            sample_action_seed = random.seed(num_actor+ epoch)
            observation = env[num_actor].reset()
            episode_return = 0
            episode_return_copy = 0
            episode_length = 0
            done_counter = 0
            sum_return = 0
            sum_length = 0

            # Iterate over the steps of each epoch
            for t in range(steps_per_epoch):
                if render and num_actor ==0 :
                    env[0].render()
                # Get the logits, action, and take one step in the environment
                observation = tf.transpose(observation, perm=[1, 2, 0, 3])  # Permute dimensions to (height, width, frames, channels)
                observation = tf.reshape(observation, shape=(img_size, img_size, 4))
                logits, action = sample_action(observation, seed = sample_action_seed)

                observation_new, reward, done, info = env[num_actor].step(action[0].numpy())
                reward = reward * 0.05
                episode_return += reward
                episode_return_copy += reward
                max_episode_return = max(max_episode_return,episode_return_copy)
                max_x_location = max(max_x_location, info.get('x_pos'))

                if done == True:
                    episode_return_copy = 0
                    done_counter +=1
                    reward = reward - (15*0.05)

                episode_length += 1

                # Get the value and log-probability of the action

                value_t = critic(tf.expand_dims(observation, axis=0))

                logprobability_t = logprobabilities(logits, action)

                # Store obs, act, rew, v_t, logp_pi_t
                buffer.store(observation, action, reward, value_t, logprobability_t)

                # Update the observation

                observation = observation_new

                # Finish trajectory if reached to done state or an end of episode t == steps_per_epoch - 1

                observation_mem = observation
                terminal = done
                if terminal or (t == steps_per_epoch - 1):
                    observation = tf.transpose(observation, perm=[1, 2, 0, 3])  # Permute dimensions to (height, width, frames, channels)
                    observation = tf.reshape(observation, shape=(img_size, img_size, 4))
                    last_value = -15*0.05 if done else critic(tf.expand_dims(observation, axis=0))
                    buffer.finish_trajectory(last_value)
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    observation, episode_return, episode_length =  env[num_actor].reset(),0, 0

            total_sum_return += (sum_return / (done_counter+1))

        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()



        num_samples = steps_per_epoch*num_actors
        indices = list(range(num_samples))

        for e in range(train_policy_iterations):
            random.shuffle(indices)
            ii = 0
            kl = 0
            while ii < num_samples:
                observation_buffer_batch = []
                action_buffer_batch = []
                advantage_buffer_batch = []
                return_buffer_batch = []
                logprobability_buffer_batch = []

                for b in range(batch_size):
                    index = indices[ii]
                    observation_buffer_batch.append(observation_buffer[index])
                    action_buffer_batch.append(action_buffer[index])
                    advantage_buffer_batch.append(advantage_buffer[index])
                    return_buffer_batch.append(return_buffer[index])
                    logprobability_buffer_batch.append(logprobability_buffer[index])
                    ii += 1

                observation_buffer_batch = np.array(observation_buffer_batch)
                action_buffer_batch = np.array(action_buffer_batch)
                advantage_buffer_batch = np.array(advantage_buffer_batch)
                logprobability_buffer_batch = np.array(logprobability_buffer_batch)
                return_buffer_batch = np.array(return_buffer_batch)
                # Update the policy and implement early stopping using KL divergence

                #print(observation_buffer_batch.shape, action_buffer_batch.shape,logprobability_buffer_batch.shape, advantage_buffer_batch.shape, return_buffer_batch.shape)
                new_lr = policy_learning_rate *(1- float(epoch)/float(epochs)*(policy_learning_rate - policy_final_learning_rate))
                #print(new_lr)
                keras.backend.set_value(policy_optimizer.learning_rate, new_lr)
                train_value_and_policy(observation_buffer_batch, action_buffer_batch, logprobability_buffer_batch, advantage_buffer_batch,return_buffer_batch, epoch)

        # Print mean return and length for each epoch
        print(f" Epoch: {epoch + 1}. Done counter: {done_counter} Max Return : {max_episode_return} Max X location = {max_x_location} Mean Return: {total_sum_return / num_actors}. Mean Length: {sum_length / ((done_counter+1)*num_actors)}")


