import tensorflow as tf
import csv
import numpy as np
import time
import os
import datetime
import random
import itertools
import multiprocessing

from common import LearningChecker, csv_array_reader

features = ["LF", "SDNN", "SD1", "SD2", "SVI", "averageNN", "RMSSD", "SDSD", "pNN50", "HF"]

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('action', None, """select | eval""")
tf.app.flags.DEFINE_string('prefix', "elm-bin", """prefix for results""")
tf.app.flags.DEFINE_boolean('retrain', True, """whether to retrain, only valid for eval action""")
tf.app.flags.DEFINE_boolean('verbose', False, "")
tf.app.flags.DEFINE_boolean('debug', False, "")


class Ranker:
    def __init__(self,
                 input_features_name,
                 ELM_Size, ELM_init_sd, READ_init_sd,
                 checkpoint_name):

        self.checkpoint_name = checkpoint_name
        self.features = input_features_name
        self.ELM_Size = ELM_Size
        self.ELM_init_sd = ELM_init_sd
        self.READ_init_sd = READ_init_sd

        self.variable_to_save = []
        self.weight_cost = 0.0
        self.network_built = False


    @staticmethod
    def _load_file(filename, features):
        pref = ["pre_"+f for f in features]
        postf = ["post_" + f for f in features]

        reader = csv_array_reader(filename, ".")

        pre, post = zip(*[ ([float(row[k1]) for k1 in pref], [float(row[k2]) for k2 in postf]) for row in reader])
        return pre, post

    def _network(self, input1_t, input2_t, linout=False):
        size = self.ELM_Size
        reuse = True if self.network_built else False

        input_t = tf.concat([input1_t, input2_t], axis=1)

        with tf.variable_scope("ELM", reuse=reuse):
            Welm = tf.get_variable("WELM", initializer=tf.truncated_normal([len(self.features)*2, size], mean=0, stddev=self.ELM_init_sd, dtype=tf.float32), trainable=False, dtype=tf.float32)
            Belm = tf.get_variable("BELM", initializer=tf.truncated_normal([size], mean=0, stddev=self.ELM_init_sd, dtype=tf.float32), trainable=False, dtype=tf.float32)
            encoding = tf.nn.tanh(tf.matmul(input_t, Welm) + Belm)

        with tf.variable_scope("READ", reuse=reuse):
            WR = tf.get_variable("WR", initializer=tf.truncated_normal([size, 1], stddev=self.READ_init_sd, dtype=tf.float32), dtype=tf.float32)
            BR = tf.get_variable("BR", initializer=tf.truncated_normal([1], stddev=self.READ_init_sd, dtype=tf.float32), dtype=tf.float32)

            outl = tf.matmul(encoding, WR) + BR
            out = tf.nn.tanh(outl)

        self.weight_cost = tf.nn.l2_loss(WR)

        if not reuse:
            self.variable_to_save = [Welm, Belm, WR, BR]

        self.network_built = True
        return outl if linout else out

    def _loss(self, pre, post, regularization):
        return \
            tf.reduce_mean((2 - (self._network(pre, post))) ** 2) + \
            tf.reduce_mean((-2 - (self._network(post, pre))) ** 2) + \
            self.weight_cost * regularization

    def _performance(self, pre, post):
        return \
            (tf.reduce_mean(tf.cast(tf.equal(tf.sign((self._network(pre, post))), 1), tf.float32)) + \
             tf.reduce_mean(tf.cast(tf.equal(tf.sign((self._network(post, pre))), -1), tf.float32))) / 2.0

    def train(self, tr_matrix, sl_matrix,
              batch_size,
              learning_rate,
              momentum,
              regularization):

        global_step = tf.Variable(initial_value=0, trainable=False)

        tr = tf.constant(tr_matrix, dtype=tf.float32)
        tr_batch = tf.slice(tr, [tf.minimum(tf.mod(global_step*batch_size, tr_matrix.shape[0]), tr_matrix.shape[0]-batch_size-1),0], [batch_size,20], name="TR_BATCH")
        tr_pre = tf.slice(tr_batch, [0,0], [batch_size, 10])
        tr_post = tf.slice(tr_batch, [0,10], [batch_size, 10])

        sl = tf.constant(sl_matrix, dtype=tf.float32)
        sl_size = 1000
        sl_batch = tf.slice(sl, [tf.minimum(tf.mod(tf.mod(global_step, 25) * sl_size, sl_matrix.shape[0]), sl_matrix.shape[0] - sl_size - 1), 0],[sl_size, 20], name="SL_BATCH")
        sl_pre = tf.slice(sl_batch, [0,0], [sl_size, 10])
        sl_post = tf.slice(sl_batch, [0,10], [sl_size, 10])

        loss = self._loss(tr_pre, tr_post, regularization)

        train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step)

        tr_perf = self._performance(tr_pre, tr_post)
        sl_perf = self._performance(sl_pre, sl_post)

        checker = LearningChecker(
            max_iteration=6000,
            check_every=25,
            max_no_improvements_iteration=250,
            min_loss_improvement=2.5e-3,
            tr_max_worst_performance=0.02
            )

        with tf.Session() as S:
            saver = tf.train.Saver(self.variable_to_save)

            S.run(tf.global_variables_initializer())

            stop = False
            it = 0

            while not stop:
                S.run(train_step)

                if checker.has_to_check(it):

                    l, p_tr, p_sl = S.run([loss, tr_perf, sl_perf])
                    stop = checker.check(it, l, p_tr, p_sl, print_log=FLAGS.verbose)

                    if checker.new_best:
                        saver.save(S, self.checkpoint_name)

                it += 1

            return checker.best_info

    def eval(self, data):
        d = tf.constant(data, dtype=tf.float32)
        pre = tf.slice(d, [0, 0], [data.shape[0], 10])
        post = tf.slice(d, [0, 10], [data.shape[0], 10])
        perf = self._performance(pre, post)
        saver = tf.train.Saver(self.variable_to_save)

        with tf.Session() as S:
            saver.restore(S, self.checkpoint_name)
            return S.run(perf)

    def get_ranking(self, csv_file):
        pre, post = Ranker._load_file(csv_file, self.features)
        all = pre + post
        p1, p2 = random.sample(all, 2)
        p1_t = tf.tile(tf.reshape(tf.convert_to_tensor(p1), shape=[1, 10]), [len(all), 1])
        p2_t = tf.tile(tf.reshape(tf.convert_to_tensor(p2), shape=[1, 10]), [len(all), 1])
        p1_out = self._network(p1_t, all, True)
        p2_out = self._network(p2_t, all, True)

        saver = tf.train.Saver(self.variable_to_save)
        with tf.Session() as S:
            saver.restore(S, self.checkpoint_name)
            return S.run([p1_out, p2_out])


def worker_initializer(_q, _tr, _sl, _ts, result_file_path, _result_file_header):
    global q, tr, sl, ts, result_writer, result_file_header
    q = _q
    tr = _tr
    sl = _sl
    ts = _ts

    result_writer = csv.DictWriter(open(os.path.join(result_file_path,str(os.getpid())+'.csv'), "w"), _result_file_header)
    result_file_header = _result_file_header


def eval_model(m):
    global q, tr, sl, ts, result_file_header, result_writer

    runs = []

    t = time.time()
    total_run = 10

    for run in range(total_run):
        tf.reset_default_graph()
        np.random.shuffle(tr)
        np.random.shuffle(sl)

        R = Ranker(features,
                   ELM_Size=m['elm_size'],
                   ELM_init_sd=m['elm_init_sd'],
                   READ_init_sd=m['read_init_sd_m'] / m['elm_size'],
                   checkpoint_name='ckp/temp_' + str(os.getpid()) + '.ckp'
                   )

        r = R.train(
            tr_matrix=tr,
            sl_matrix=sl,
            batch_size=150,
            learning_rate=m['learning_rate'],
            momentum=m['momentum'],
            regularization=m['reg']
        )

        ts_res = R.eval(ts)

        runs.append([x for x in r] + [ts_res])

    avgs = np.mean(runs, axis=0)
    stds = np.std(runs, axis=0)

    keys = result_file_header
    row = [m[k] for k in m.keys()] + [
        avgs[0], avgs[2], avgs[3], avgs[4],
        stds[0], stds[2], stds[3], stds[4]]
    res = dict(zip(keys, row))
    result_writer.writerow(res)

    q.put([res, time.time() - t])


def model_selection(prefix):
    global features

    # preload all data
    cols = ['pre_' + f for f in features] + ['post_' + f for f in features]
    tr = np.array(
        [[float(row[k]) for k in cols] for row in csv_array_reader('data/tr.csv', ".")])
    sl = np.array(
        [[float(row[k]) for k in cols] for row in csv_array_reader('data/sl.csv', ".")])

    ts = np.array([[float(row[k]) for k in cols] for row in csv_array_reader('data/ts.csv', ".")])

    pars = {
        'elm_size': [250, 500, 1000, 2500, 5000, 7500, 10000],
        'elm_init_sd': [1.0, 2.5, 5.0],
        'read_init_sd_m': [1.0, 10.0, 50.0],
        'learning_rate': [0.01, 1e-3, 1e-4, 1e-5],
        'momentum': [0, 0.3, 0.6],
        'reg': [0, 1e-2, 1e-3, 1e-4]
    }

    named_pars = {k: [(k,v) for v in pars[k]] for k in pars.keys()}
    all_model = map(dict, itertools.product(*named_pars.values()))

    results_col = ['avg_best_it', 'avg_tr_perf', 'avg_sl_perf', 'avg_ts_perf', 'sd_best_it', 'sd_tr_perf', 'sd_sl_perf', 'sd_ts_perf']
    header = pars.keys()+results_col

    base_results = os.path.join("results/", prefix)
    os.makedirs(base_results)

    with open(os.path.join(base_results, 'header.csv'), 'w') as f:
        f.write(",".join(header) + "\n")

    q = multiprocessing.Queue()
    if FLAGS.debug:
        worker_initializer(q, tr, sl, ts, base_results, header)
        res = map(eval_model, all_model)
    else:
        pool = multiprocessing.Pool(multiprocessing.cpu_count() / 2, initializer=worker_initializer, initargs=(q, tr, sl, ts, base_results, header))
        res = pool.map_async(eval_model, all_model)

    computed_model = 0
    t0 = time.time()
    while not res.ready() or not q.empty():
        if not q.empty():
            model, t = q.get(timeout=10)
            computed_model += 1
            delta_t = time.time() - t0
            eta = datetime.timedelta(seconds=(delta_t / computed_model) * (len(all_model) - computed_model))
            print "Computed {0} model of {1}. Last took {2:.2f} seconds. [{3} - ETA: {4}]".format(
                computed_model, len(all_model), t, str(datetime.timedelta( seconds=delta_t)), str(eta))
        else:
            time.sleep(10)


def main(argv=None):
    FLAGS.prefix += '-' + str(int(time.time()))

    if FLAGS.action == "select":
        model_selection(FLAGS.prefix)
    elif FLAGS.action == "eval":
        R = Ranker(features,
                   ELM_Size=10000,
                   ELM_init_sd=5,
                   READ_init_sd=1e-3,
                   checkpoint_name='ckp/elm-bin.ckp'
                   )
        if FLAGS.retrain:

            # preload all data
            cols = ['pre_' + f for f in features] + ['post_' + f for f in features]
            tr = np.array(
                [[float(row[k]) for k in cols] for row in csv_array_reader('data/tr.csv', ".")])
            sl = np.array(
                [[float(row[k]) for k in cols] for row in csv_array_reader('data/sl.csv', ".")])

            R.train(
                tr_matrix=tr,
                sl_matrix=sl,
                batch_size=150,
                learning_rate=0.0001,
                momentum=0.6,
                regularization=0.01
                )

        ranking = R.get_ranking('data/ts.csv')
        z = zip(ranking[0], ranking[1])
        m = map((lambda x: str(x[0][0])+','+str(x[1][0])), z)
        j = "\n".join(m)
        with open(os.path.join("results", FLAGS.prefix+"_ts-ranking.csv"), 'wb') as f:
            f.write("pivot1,pivot2\n")
            f.write(j)
    else:
        print "Invalid action"

if __name__ == "__main__":
    tf.app.run()