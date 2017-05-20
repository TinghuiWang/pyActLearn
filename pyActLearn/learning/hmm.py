import pickle
import numpy as np
from hmmlearn.hmm import MultinomialHMM


class HMM:
    r"""Hidden Markov Model

    This HMM class implements solution of two problems:
    # Supervised learning problem
    # Decode Problem

    Supervised learning:
    
    Given a sequence of observation O: O1, O2, O3, ... and corresponding state sequence Q: Q1, Q2, Q3, ...
    Update probability of state transition matrix A, and observation probability matrix B.
    
    The value of both A and B are estimated based on the state transition and observation in training data,
    so that the joint probability of X (the given observation) and Y (the given state sequence) is maximized.

    Decode Problem:
    
    Given state transition matrix A, and observation probability matrix B, and a sequence of observation
    O: O1, O2, O3, ... Find the most probable state sequence Q: Q1, Q3, Q3, ...
    
    In this code, Vertibi algorithm is implemented to solve the decode problem.

    Args:
        num_states (:obj:`int`): Size of state space
        num_observations (:obj:`int`): Size of observation space
        
    Attributes:
        num_states (:obj:`int`): Size of state space
        num_observations (:obj:`int`): Size of observation space
        A (:obj:`numpy.ndarray`): Transition matrix of size (num_states, num_states), where :math:`a_{ij}` is the 
            probability of transition from :math:`q_i` to :math:`q_j`
        B (:obj:`numpy.ndarray`): Emission matrix of size (num_states, num_observation), where :math:`b_{ij}` is the 
            probability of observing :math:`o_j` from :math:`q_i`
        total_learned_events (:obj:`int`): Number of events learned so far - used for incremental learning
    """
    def __init__(self, num_states, num_observations):
        self.num_states = num_states
        self.num_observations = num_observations
        self.A_count = np.ones((num_states, num_states))
        self.B_count = np.ones((num_states, num_observations))
        self.A = self.A_count / np.sum(self.A_count, axis=1, keepdims=True)
        self.B = self.B_count / np.sum(self.B_count, axis=1, keepdims=True)

    def decode(self, observations, init_probability):
        """Vertibi Decode Algorithm

        Parameters
        ----------
        observations : np.array
            A sequence of observation of size (T, ) O: O_1, O_2, ..., O_T
        init_probability : np.array
            Initial probability of states represented by an array of size (N, )

        Vertibi algorithm is composed of three parts: Initialization, Recursion and Termination

        Temporary Parameters
        --------------------
        trellis : np.array
            trellis stores the best scores so far (or Vertibi path probility)
            It is an array of size (N, T), where N is the number of states, and T is the length of observation sequence
            trellis_{jt} = max_{q_0, q_1, ..., q_{t-1}} P(q_0, q_1, ..., q_t, o_1, o_2, ..., o_n, q_t=j | \lambda)
        back_trace : np.array
            back_trace is an array that stores the most possible path that corresponds to the best scores in trellis
            It is an array of size (N, T) as well
        """
        # Allocate temporary arrays
        T = observations.shape[0]
        trellis = np.zeros((self.num_states, T))
        back_trace = np.ones((self.num_states, T)) * -1

        # Initialization
        # trellis_{i0} = \pi_{i} * B{i,O_0}
        # In Probability Term:
        # P(O_0, q_0=k) = P(O_0 | q_0=k) * P(q_0 = k)
        trellis[:, 0] = np.squeeze(init_probability * self.B[:, observations[0]])

        # Recursion
        # If the end state is q_T = k, find the q_{T-1} so that the likelihood to q_T = k is maximized
        # And store that maximum likelihood in trellis for future use
        # trellis_{k, T} = max_{x} P(O_0, O_1, ..., O_T, q_0, q_1, ..., q_{T-1}=x, q_T=k)
        #                = max_{x} P(O_0, O_1, ..., O_{T-1}, q_0, q_1, ..., q_{T-1}=x) * P(q_T=k|q_{T-1}=x) * P(O_T|q_T=k)
        #                = max_{x} trellis_{x, T-1} * A_{x,k} * B(k, O_T)
        for i in range(1, T):
            trellis[:, i] = (trellis[:, i-1, None].dot(self.B[:, observations[i], None].T) * self.A).max(0)
            back_trace[:, i] = (np.tile(trellis[:, i-1, None], [1, self.num_states]) * self.A).argmax(0)

        # Termination - back trace
        tokens = [trellis[:, -1].argmax()]
        for i in range(T-1, 0, -1):
            tokens.append(int(back_trace[tokens[-1], i]))
        return tokens[::-1]

    def learn(self, observations, states):
        """Update transition matrix A and emission matrix B with training sequence composed of a sequence of 
        observations and corresponding states.
        """
        # Make sure that states and observations equal each other
        if observations.shape[0] == states.shape[0]:
            T = observations.shape[0]
        else:
            return -1
        # Update Counts
        for i in range(1, T):
            if states[i] >= self.num_states or states[i-1] >= self.num_states or \
                            observations[i] >= self.num_observations:
                return -2
            # Update Emission Count (Skip the first one)
            self.B_count[states[i], observations[i]] += 1
            # Update State Transition
            self.A_count[states[i-1], states[i]] += 1
        # Update Probability Matrix
        self.A = self.A_count / np.sum(self.A_count, axis=1, keepdims=True)
        self.B = self.B_count / np.sum(self.B_count, axis=1, keepdims=True)

    def save(self, filename):
        """Pickle the model to file.
        
        Args:
            filename (:obj:`str`): The path of the file to store the model parameters.
        """
        f = open(filename, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def predict(self, x, init_prob=None, method='hmmlearn', window=-1):
        """Predict result based on HMM
        """
        if init_prob is None:
            init_prob = np.array([1/self.num_states for i in range(self.num_states)])
        if method == 'hmmlearn':
            model = MultinomialHMM(self.num_states, n_iter=100)
            model.n_features = self.num_observations
            model.startprob_ = init_prob
            model.emissionprob_ = self.B
            model.transmat_ = self.A
            if window == -1:
                result = model.predict(x)
            else:
                result = np.zeros(x.shape[0], dtype=np.int)
                result[0:window] = model.predict(x[0:window])
                for i in range(window, x.shape[0]):
                    result[i] = model.predict(x[i-window+1:i+1])[-1]
        else:
            if window == -1:
                result = self.decode(x, init_prob)
            else:
                result = np.zeros(x.shape[0], dtype=np.int)
                result[0:window] = self.decode(x[0:window], init_prob)
                for i in range(window, x.shape[0]):
                    result[i] = self.decode(x[i-window+1:i+1], init_prob)[-1]
        return result

    def predict_prob(self, x, init_prob=None, window=-1):
        """Predict the probability
        """
        if init_prob is None:
            init_prob = np.array([1/self.num_states for i in range(self.num_states)])
        model = MultinomialHMM(self.num_states)
        model.n_features = self.num_observations
        model.startprob_ = init_prob
        model.emissionprob_ = self.B
        model.transmat_ = self.A
        return model.predict_proba(x)
