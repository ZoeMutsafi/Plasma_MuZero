import datetime
import itertools
import os

import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (2, 3, 3)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(TOTAL_OPTIONS))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 1000  # Maximum number of moves if game is not finished before
        self.num_simulations = 25  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = PlasmaRecursive()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.
    
        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                row = int(
                    input(
                        f"Enter the row (1, 2 or 3) to play for the player {self.to_play()}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (1, 2 or 3) to play for the player {self.to_play()}: "
                    )
                )
                choice = (row - 1) * 3 + (col - 1)
                if (
                    choice in self.legal_actions()
                    and 1 <= row
                    and 1 <= col
                    and row <= 3
                    and col <= 3
                ):
                    break
            except:
                pass
            print("Wrong input, try again")
        return choice

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        
        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        row = action_number // 3 + 1
        col = action_number % 3 + 1
        return f"Play row {row}, column {col}"


MAX_TARGET_NUMBER = 99
TOTAL_OPTIONS = 6
VALIDATOR = False  # Should be set to True only on test, Adding the deterministic validator to the test execution


class PlasmaRecursive:
    def __init__(self):
        self.board = None
        self.actions = None
        self.correct_sequences = None
        self.total_correct_sequences = 0
        self.real_target = None
        self.real_options = None
        self.round = 0
        self.player = 0
        self.new_level()

    def to_play(self):
        return 0

    def new_level(self):
        # Generate new Plasma board, init state to game start
        self.board = numpy.zeros((4, 3), dtype="int32")
        self.correct_sequences = set()
        self.round = 0

        target_number = numpy.random.randint(3, MAX_TARGET_NUMBER + 1)
        self.real_target = target_number
        first_row = numpy.zeros((1, 3), dtype="int32")
        first_row[0][0] = self.round
        first_row[0][1] = target_number % 10
        sol1 = numpy.random.randint(0, target_number - 2) + 1
        sol3 = target_number - sol1
        options = [sol1, sol3]
        while len(options) < TOTAL_OPTIONS:
            next_number = numpy.random.randint(1, target_number)
            if next_number * 2 != target_number:
                options.append(next_number)
        np_options = numpy.array(options)
        numpy.random.shuffle(np_options)
        padding = [0] * (9 - TOTAL_OPTIONS)
        np_options_parsed = numpy.array(list(np_options) + padding).reshape(3,3)
        self.real_options = np_options_parsed
        self.board = numpy.concatenate((first_row, np_options_parsed % 10), axis=0)
        self.total_correct_sequences = self.count_all_combinations()

        self.reset_actions()

    def reset(self):
        self.new_level()
        return self.get_observation()

    def step(self, action):
        row = action // 3 + 1
        col = action % 3
        self.actions[row, col] = 1

        sequence_found, sequence_correct, sequence_new, has_another_seq, distance_from_target = self.game_result()

        reward = 0
        game_finished = False
        if sequence_found:
            if not sequence_correct:
                reward = -1 * 5 * distance_from_target  # we will receive greater punishment if we were distant from the target number
                game_finished = True
            elif not has_another_seq:
                reward = 100  # Finished finding all sequences! new game
                game_finished = True
            else:
                reward = 5 if sequence_new else -1  # -1 or 5 if we saw this correct combination or not
                if sequence_new:
                    self.round += 1  # new round
                    self.board[0][0] = self.round
                    if VALIDATOR:
                        # Execute the deterministic validator to check if suggested sequence is correct
                        if numpy.where(self.actions[1:] == 1, self.real_options, 0).sum() == self.real_target:
                            print("real_options: ", self.real_options)
                            print("real_target: ", self.real_target)
                            reward = 200
                            game_finished = True
                self.reset_actions()  # We can select new sequence from scratch

        return self.get_observation(), reward, game_finished

    def get_observation(self):
        return numpy.array([self.actions[:3], self.board[:3]], dtype="int32")

    def legal_actions(self):
        legal = []
        for i in range(TOTAL_OPTIONS):
            row = i // 3 + 1
            col = i % 3
            if self.actions[row, col] == 0:
                legal.append(i)
        return legal

    def count_all_combinations(self):
        target_number = self.board[0][1]
        data = self.board[1:, :]
        count = 0
        i = 2
        for combinations in itertools.combinations(data.reshape(1,-1)[0], i):
            if (0 not in combinations) and (sum(combinations) in [target_number, target_number + 10]):
                count += 1
        return count

    def game_result(self):
        # return 5 tuple:
        # ** sequence_found - Is a sequence found
        # ** sequence_correct - Is the sequence reach the sum
        # ** sequence_new - Is the sequence was seen before
        # ** has_another_seq - Is there another sequence
        # ** distance_from_target - the distance of the found sequence from the target
        selected_actions = numpy.where(self.actions == 1, self.board, 0)
        sum_actions = selected_actions.sum()
        count_actions = numpy.where(self.actions == 1, self.actions, 0).sum()
        target_number = self.board[0][1]
        game_won = sum_actions in [target_number, target_number + 10]
        game_finished = ((sum_actions >= target_number + 10) or count_actions == 2)
        distance_from_target = abs(target_number - sum_actions)
        sequence_new = False
        has_more_seqs = True
        if game_finished and game_won:
            action_tuple = tuple([tuple(action) for action in numpy.argwhere(selected_actions)])
            existing_sequence = action_tuple in self.correct_sequences
            sequence_new = False
            if not existing_sequence:
                self.correct_sequences.add(action_tuple)
                sequence_new = True  # check if already found this sequence
            has_more_seqs = len(self.correct_sequences) < self.total_correct_sequences
        return game_finished, game_won, sequence_new, has_more_seqs, distance_from_target

    def reset_actions(self):
        self.actions = numpy.zeros((4, 3))

    def expert_action(self):
        raise Exception("Not Implemented")

    def render(self):
        print("Target Number: ", self.board[0][1])
        print("Selected Sum: ", numpy.where(self.actions == 1, self.board, 0).sum())
        print("current board: ")
        print(self.board)
        print("selected cells: ")
        print(self.actions)
