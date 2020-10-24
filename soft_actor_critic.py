# The code is modified from rail-berkeley/softlearning repo https://github.com/rail-berkeley/softlearning

from copy import deepcopy
from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def td_targets(rewards, discounts, next_values):
    return rewards + discounts * next_values

def compute_Q_targets(next_Q_values,
                      next_log_pis,
                      rewards,
                      terminals,
                      discount,
                      entropy_scale,
                      reward_scale):
    next_values = next_Q_values - entropy_scale * next_log_pis
    terminals = tf.cast(terminals, next_values.dtype)

    Q_targets = td_targets(
        rewards=reward_scale * rewards,
        discounts=discount,
        next_values=(1.0 - terminals) * next_values)

    return Q_targets


def heuristic_target_entropy(action_space):
    heuristic_target_entropy = -np.prod(action_space.shape)

    return heuristic_target_entropy


class SAC:
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            policy,
            Qs,
            policy_optimizer,
            q_optimizers,
            action_space,
            plotter=None,
            policy_lr=3e-4,
            Q_lr=3e-4,
            alpha_lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            save_full_state=False,
            Q_targets=None,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
        """

        self._policy = policy

        self._Qs = Qs

        if Q_targets is not None:
            self._Q_targets = Q_targets
        else:
            self._Q_targets = tuple(deepcopy(Q) for Q in Qs)
            self._update_target(tau=tf.constant(1.0))

        self._plotter = plotter

        self._policy_lr = policy_lr
        self._Q_lr = Q_lr
        self._alpha_lr = alpha_lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            heuristic_target_entropy(action_space)
            if target_entropy == 'auto'
            else target_entropy)

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval

        self._save_full_state = save_full_state

        self._Q_optimizers = q_optimizers
        self._policy_optimizer = policy_optimizer

        self._log_alpha = tf.Variable(0.0, dtype=tf.float16)
        self._alpha = tfp.util.DeferredTensor(self._log_alpha, tf.exp)

        self._alpha_optimizer = tf.optimizers.Adam(
            self._alpha_lr, name='alpha_optimizer')

    def _compute_Q_targets(self, batch):
        next_observations = batch['next_observations']
        rewards = batch['rewards']
        terminals = batch['terminals']

        entropy_scale = self._alpha
        reward_scale = self._reward_scale
        discount = self._discount

        next_actions, next_log_pis = self._policy.actions_and_log_probs(
            next_observations)
        next_Qs_values = tuple(
            # Q.values(next_observations, next_actions) for Q in self._Q_targets)
            Q(tf.concat((next_observations, next_actions), axis=-1)) for Q in self._Q_targets)
        next_Q_values = tf.reduce_min(next_Qs_values, axis=0)

        Q_targets = compute_Q_targets(
            next_Q_values,
            next_log_pis,
            rewards,
            terminals,
            discount,
            entropy_scale,
            reward_scale)

        return tf.stop_gradient(Q_targets)

    def _update_critic(self, batch):
        """Update the Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_targets = self._compute_Q_targets(batch)
        Q_targets = tf.expand_dims(Q_targets, axis=-1)

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        rewards = tf.expand_dims(rewards, axis=-1)

        # tf.debugging.assert_shapes((
        #     (Q_targets, ('B', 1)), (rewards, ('B', 1))))

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values = Q(tf.concat((observations, actions), axis=-1))
                Q_losses = 0.5 * (
                    tf.losses.MSE(y_true=Q_targets, y_pred=tf.expand_dims(Q_values, axis=-1)))
                Q_loss = tf.nn.compute_average_loss(Q_losses)

            optimizer(tape, Q_loss)
            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)

        return Qs_values, Qs_losses

    def _update_actor(self, batch):
        """Update the policy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        observations = batch['observations']

        with tf.GradientTape() as tape:
            actions, log_pis = self._policy.actions_and_log_probs(observations)

            Qs_log_targets = tuple(
                # Q.values(observations, actions) for Q in self._Qs)
                Q(tf.concat((observations, actions), axis=-1)) for Q in self._Qs)
            Q_log_targets = tf.reduce_min(Qs_log_targets, axis=0)
            policy_losses = self._alpha * log_pis - Q_log_targets
            policy_loss = tf.nn.compute_average_loss(policy_losses)

        return policy_losses

    # @tf.function(experimental_relax_shapes=True)
    def _update_alpha(self, batch):
        if not isinstance(self._target_entropy, Number):
            return 0.0

        observations = batch['observations']

        actions, log_pis = self._policy.actions_and_log_probs(observations)

        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (
                self._alpha * tf.stop_gradient(log_pis + self._target_entropy))

            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        alpha_gradients = tape.gradient(alpha_loss, [self._log_alpha])

        return alpha_losses

    def _update_target(self, tau):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

    def _do_updates(self, states, actions, rewards, dones):
        """Runs the update operations for policy, Q, and alpha."""
        batch = OrderedDict((
            ('observations', states[:-1]),
            ('next_observations', states[1:]),
            ('rewards', rewards[:-1]),
            ('terminals', dones[:-1]),
            ('actions', actions[:-1])
        ))
        Qs_values, Qs_losses = self._update_critic(batch)
        policy_losses = self._update_actor(batch)
        alpha_losses = self._update_alpha(batch)

        diagnostics = OrderedDict((
            ('Q_value-mean', tf.reduce_mean(Qs_values)),
            ('Q_loss-mean', tf.reduce_mean(Qs_losses)),
            ('policy_loss-mean', tf.reduce_mean(policy_losses)),
            ('alpha', tf.convert_to_tensor(self._alpha)),
            ('alpha_loss-mean', tf.reduce_mean(alpha_losses)),
        ))
        return diagnostics

    def _do_training(self, iteration, states, actions, rewards, dones):
        training_diagnostics = self._do_updates(states, actions, rewards, dones)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target(tau=tf.constant(self._tau))

        return training_diagnostics

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as an ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """
        diagnostics = OrderedDict((
            ('alpha', self._alpha.numpy()),
            ('policy', self._policy.get_diagnostics_np(batch['observations'])),
        ))

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_alpha': self._alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
