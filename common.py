import os
import csv

class LearningChecker:
    def __init__(self,
                 max_iteration,
                 check_every=50,
                 max_no_improvements_iteration=150,
                 min_loss_improvement=1e-4,
                 tr_max_worst_performance=0.03):

        self.max_iteration = max_iteration
        self.max_no_improvements_iteration = max_no_improvements_iteration
        self.epsilon_loss = min_loss_improvement
        self.check_every = check_every
        self.tr_max_worst_performance = tr_max_worst_performance

        self.best_sl = 0
        self.best_info = None
        self.no_improvements = 0
        self.new_best = False

        self.previous_info = None

    def has_to_check(self, it):
        return it % self.check_every == 0 and it > 0

    def reset(self):
        self.best_info = None
        self.previous_info = None
        self.best_sl = 0

    def check(self, iteration, tr_loss, tr_perf, sel_perf, print_log=True):
        """
          Returns:
              - boolean: has to stop
              - boolean: is the best so far
        """

        def _print(str):
            if print_log:
                print str
        log=""
        if print_log:
            log = "{:d}:\t {:.2f} {:.2f} | Loss: {:.6f}".format(iteration, tr_perf, sel_perf, tr_loss)

        if self.best_info is None or (sel_perf > self.best_sl + self.epsilon_loss and tr_perf >= sel_perf - self.tr_max_worst_performance):
            self.best_sl = sel_perf
            self.best_info = iteration, tr_loss, tr_perf, sel_perf
            _print(log+" B")
            self.new_best = True
        else:
            self.new_best = False
            _print(log)

        self.previous_info = iteration, tr_loss, tr_perf, sel_perf

        if iteration > self.max_iteration:
            _print("STOP: max iteration!\n")
            return True  # stop due to iterations
        elif iteration - self.best_info[0] > self.max_no_improvements_iteration:
            _print("STOP: no improvement!\n")
            return True  # early stop, no improvements in the last checks
        else:
            return False  # keep iterating


def csv_array_reader(filename, base_path=None):
    try:
        with open(os.path.join(base_path, filename)) as f:
            for x in csv.DictReader(f):
                yield x
    except Exception as e:
        print ("Failed to load" + str(filename) + " \n" + e.message)
