import logging

logger = logging.getLogger(__name__)


class ConstIterations:
    """Stopping Criterion: After certain iterations

    Args:
        num_iters (:obj:`int`): Number of iterations

    Attributes:
        num_iters (:obj:`int`): Number of iterations
        cur_iter (:obj:`int`): Current number of iterations
    """
    def __init__(self, num_iters):
        self.num_iters = num_iters
        self.cur_iter = 0

    def reset(self):
        """Reset internal iteration counter
        """
        self.cur_iter = 0

    def continue_learning(self):
        """Determine whether learning should continue
        If so, return True, otherwise, return False.
        """
        if self.cur_iter < self.num_iters:
            self.cur_iter += 1
            return True
        else:
            return False


class MonitorBased:
    """Stop training based on the return of a monitoring function.

    If the monitoring result keep improving within past n_steps, keep learning.
    Otherwise, stop.
    If the monitoring result is the best at the moment, call the parameter save
    function.
    Once it is done, the parameters saved last is the training results.

    Args:
        n_steps (:obj:`int`): The amount of steps to look for improvement
        monitor_fn: Parameter monitor function.
        monitor_fn_args (:obj:`tuple`): Argument tuple (arg1, arg2, ...) for monitor function.
        save_fn: Parameter save function.
        save_fn_args (:obj:`tuple`): Argument tuple (arg1, arg2, ...) for save function.

    Attributes:
        n_steps (:obj:`int`): The amount of steps to look for improvement
        monitor_fn: Parameter monitor function.
        monitor_fn_args (:obj:`tuple`): Argument tuple (arg1, arg2, ...) for monitor function.
        save_fn: Parameter save function.
        save_fn_args (:obj:`tuple`): Argument tuple (arg1, arg2, ...) for save function.
        step_count (:obj:`int`): Number of steps that the parameter monitored is worse than the best value.
        best_value: Best value seen so far.
    """
    def __init__(self, n_steps, monitor_fn, monitor_fn_args, save_fn, save_fn_args):
        self.n_steps = n_steps
        self.monitor_fn = monitor_fn
        self.monitor_fn_args = monitor_fn_args
        self.save_fn = save_fn
        self.save_fn_args = save_fn_args
        self.step_count = 0
        self.best_value = None

    def reset(self):
        """Reset internal step count
        """
        self.step_count = 0
        self.best_value = None

    def continue_learning(self):
        """Determine whether learning should continue
        If so, return True, otherwise, return False.
        """
        param = self.monitor_fn(*self.monitor_fn_args)
        if self.best_value is None:
            self.best_value = param
            self.save_fn(*self.save_fn_args)
        if param > self.best_value:
            self.step_count = 0
            self.best_value = param
            self.save_fn(*self.save_fn_args)
            logger.info('New Best: %g' % self.best_value)
        else:
            self.step_count += 1
            if self.step_count > self.n_steps:
                return False
        return True
