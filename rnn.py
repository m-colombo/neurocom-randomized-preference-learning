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

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('action', None, """select | eval""")
tf.app.flags.DEFINE_string('prefix', "rnn", """prefix for results""")
tf.app.flags.DEFINE_integer('nprocess', 4, "number of parallel process for model selection")
tf.app.flags.DEFINE_boolean('debug', False, "")

tr_mean, tr_sd = 0.81548140, 0.17976908     # training data stat


class Ranker:
    def __init__(self, checkpoint_name,  rnn_cell):
        self.rnn_cell = rnn_cell
        self.checkpoint_name = checkpoint_name

        self.network_built = False
        self.weight_loss = 0
        self.variable_to_save = []

    def fetch_variable_to_save(self):
        with tf.variable_scope("Network", reuse=True):
            self.variable_to_save = [tf.get_variable(n) for n in [
                "Embedding/basic_rnn_cell/weights",
                "Embedding/basic_rnn_cell/biases",
                "Proj/kernel",
                "Proj/bias"
            ]]

    def _embedding(self, input_t, length_t):
        """ input_t: [ rr_max_length x batch ]
            length_t: [ batch ]
         """
        with tf.variable_scope("Embedding", reuse=self.network_built) as vs:
            cell = self.rnn_cell
            (outs, _) = tf.nn.dynamic_rnn(cell=cell, inputs=input_t, sequence_length=tf.squeeze(length_t),
                                          dtype=tf.float32, scope=vs, time_major=True, swap_memory=True)

            batch_idx = tf.squeeze(tf.range(length_t.get_shape().as_list()[0]))
            # gather last non-zero outputs
            # this work for non-time-major
            # last_outputs = tf.gather_nd(outs, tf.stack([batch_idx, tf.squeeze(length_t - 1)], axis=1))

            last_outputs = tf.gather_nd(outs, tf.stack([tf.squeeze(length_t - 1),batch_idx], axis=1))

            return last_outputs

    def _network_ws(self, pre, post, linout=False):

        with tf.variable_scope("Network", reuse=self.network_built) as vs:

            pre_emb = self._embedding(pre[0], pre[1])
            pre_proj = tf.layers.dense(pre_emb,
                                       units=1,
                                       activation=None if linout else tf.tanh,
                                       name = "Proj"
                                       )
            vs.reuse_variables()

            post_emb = self._embedding(post[0], post[1])
            post_proj = tf.layers.dense(post_emb,
                                       units=1,
                                       activation=None if linout else tf.tanh,reuse=True,
                                        name="Proj"
                                       )
            self.network_built = True

            return pre_proj - post_proj

    def _loss(self, pre, post, reg):
        n = self._network_ws(pre, post)
        self.fetch_variable_to_save()
        with tf.variable_scope("Loss"):
            self.weight_loss = tf.nn.l2_loss(self.variable_to_save[0]) + tf.nn.l2_loss(self.variable_to_save[2])
            return tf.reduce_mean((2 - (n)) ** 2) + self.weight_loss * reg

    def _performance(self, pre, post):
        n = self._network_ws(pre, post)
        with tf.variable_scope("Performance"):
            return tf.reduce_mean(tf.cast(tf.equal(tf.sign((n)), 1), tf.float32))

    @staticmethod
    def pre_load_csv(file_name, limit=None, limit_rr=60):
        g = csv_array_reader(file_name, '.')
        if limit:
            g = itertools.islice(g, limit)

        def rrr(row_len):
            all_rr = (row_len - 2) / 4
            return min(all_rr,limit_rr) if limit_rr is not None else all_rr

        data = [
            (([map(float, [row['pre_l'+str(i)], row['pre_g'+str(i)], row['post_l'+str(i)], row['post_g'+str(i)]]) for i in range(rrr(len(row)))]),
             [min(rrr(len(row)),int(row['length_pre'])), min(rrr(len(row)),int(row['length_post']))])
         for row in g]

        return data

    @staticmethod
    def _build_batch(data, size, global_step, every_n_steps=1):
        np.random.shuffle(data)
        rrs, lenghts = zip(*data)

        counter = global_step / every_n_steps

        rrs_t = tf.constant(rrs, dtype=tf.float32)
        rrs_t = tf.transpose(rrs_t, [1,0,2]) # time major

        lenghts_t = tf.constant(lenghts, dtype=tf.int32)
        data_size = rrs_t.shape[1]._value

        start = tf.minimum(tf.mod(counter * size, data_size), data_size - size - 1)
        rrs_pre = tf.slice(rrs_t, begin=[0, start, 0], size=[-1, size, 2])
        rrs_post = tf.slice(rrs_t, begin=[0, start, 2], size=[-1, size, 2])

        length_pre = tf.slice(lenghts_t, begin=[start, 0], size=[size, 1])
        length_post = tf.slice(lenghts_t, begin=[start, 1], size=[size, 1])

        return (rrs_pre, length_pre), (rrs_post, length_post)

    def train(self, tr_data, sl_data, learning_rate, momentum, reg, train_batch_size=150, check_every=25):
        global_step = tf.Variable(initial_value=0, trainable=False)

        with tf.variable_scope('TR_BATCH'):
            tr = Ranker._build_batch(tr_data, size=train_batch_size, global_step=global_step)
        with tf.variable_scope("SL_BATCH"):
            sl = Ranker._build_batch(sl_data, size=500, global_step=global_step, every_n_steps=check_every)

        loss = self._loss(tr[0], tr[1], reg)

        with tf.variable_scope("Trainer"):
            train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step)

        tr_perf = self._performance(tr[0], tr[1])
        sl_perf = self._performance(sl[0], sl[1])

        checker = LearningChecker(
            max_iteration=6000,
            check_every=check_every,
            max_no_improvements_iteration=250,
            min_loss_improvement=2.5e-3,
            tr_max_worst_performance=0.02
        )

        with tf.Session() as S:
            self.fetch_variable_to_save()
            saver = tf.train.Saver(self.variable_to_save)

            S.run(tf.global_variables_initializer())

            stop = False
            it = 0

            while not stop:
                S.run(train_step)

                if checker.has_to_check(it):

                    l, p_tr, p_sl = S.run([loss, tr_perf, sl_perf])
                    stop = checker.check(it, l, p_tr, p_sl, print_log=False)

                    if checker.new_best:
                        saver.save(S, self.checkpoint_name)

                it += 1

            return checker.best_info

    def eval(self, data):
        rrs, lenghts = zip(*data)
        rrs_t = tf.constant(rrs, dtype=tf.float32)
        lenghts_t = tf.constant(lenghts, dtype=tf.int32)
        rrs_t = tf.transpose(rrs_t, [1, 0, 2])  # time major

        rrs_pre = tf.slice(rrs_t, begin=[0, 0, 0], size=[-1, -1, 2])
        rrs_post = tf.slice(rrs_t, begin=[0, 0, 2], size=[-1, -1, 2])

        length_pre = tf.slice(lenghts_t, begin=[0, 0], size=[-1, 1])
        length_post = tf.slice(lenghts_t, begin=[0, 1], size=[-1, 1])

        pre, post = (rrs_pre, length_pre), (rrs_post, length_post)

        perf = self._performance(pre, post)
        self.fetch_variable_to_save()
        saver = tf.train.Saver(self.variable_to_save)

        with tf.Session() as S:
            saver.restore(S, self.checkpoint_name)
            return S.run(perf)

    def get_rank(self, data):
        rrs, lenghts = zip(*data)
        rrs_t = tf.constant(rrs, dtype=tf.float32)
        lenghts_t = tf.constant(lenghts, dtype=tf.int32)
        rrs_t = tf.transpose(rrs_t, [1, 0, 2])  # time major

        rrs_pre = tf.slice(rrs_t, begin=[0, 0, 0], size=[-1, -1, 2])
        rrs_post = tf.slice(rrs_t, begin=[0, 0, 2], size=[-1, -1, 2])

        length_pre = tf.slice(lenghts_t, begin=[0, 0], size=[-1, 1])
        length_post = tf.slice(lenghts_t, begin=[0, 1], size=[-1, 1])

        with tf.variable_scope("Network", reuse=True):
            emb = self._embedding(tf.concat([rrs_pre,rrs_post], axis=1), tf.concat([length_pre, length_post], axis=0))
            proj = tf.layers.dense(emb, units=1, activation=None, name="Proj")
        self.fetch_variable_to_save()
        saver = tf.train.Saver(self.variable_to_save)

        with tf.Session() as S:
            saver.restore(S, self.checkpoint_name)
            return S.run(proj)


def worker_initializer(q, tr, sl, ts):
    global queue, tr_data, sl_data, ts_data
    queue = q
    tr_data = tr
    sl_data = sl
    ts_data = ts


def eval_model(m):
    global queue, tr_data, sl_data, ts_data

    runs = []
    t = time.time()
    total_run = 10

    for run in range(total_run):
        tf.reset_default_graph()

        R = Ranker(
            checkpoint_name='ckp/temp_' + str(os.getpid()) + '.ckp',
            rnn_cell=tf.contrib.rnn.BasicRNNCell(num_units=m['rnn_size'])
           )

        r = R.train(tr_data, sl_data, m['learning_rate'], m['momentum'], m['reg'])

        ts_r = R.eval(ts_data)
        runs.append([x for x in r] + [ts_r])

    avgs = np.mean(runs, axis=0)
    stds = np.std(runs, axis=0)

    keys = m.keys() + ['avg_best_it', 'avg_tr_perf', 'avg_sl_perf', 'avg_ts_perf', 'sd_best_it', 'sd_tr_perf', 'sd_sl_perf',
                   'sd_ts_perf']
    row = [m[k] for k in m.keys()] + [
        avgs[0], avgs[2], avgs[3], avgs[4],
        stds[0], stds[2], stds[3], stds[4]]
    res = dict(zip(keys, row))

    queue.put([res, time.time() - t])


def model_selection((tr, sl, ts)):

    pars = {
        'rnn_size': [5, 10, 25, 50, 100, 150],
        'learning_rate': [0.01, 1e-3, 1e-4, 1e-5],
        'momentum': [0.0, 0.3, 0.6],
        'reg': [0, 1e-2, 1e-3, 1e-4, 1e-5]
    }

    named_pars = {k: [(k, v) for v in pars[k]] for k in pars.keys()}
    all_model = map(dict, itertools.product(*named_pars.values()))
    random.shuffle(all_model, lambda: 0.32) # to get a less biased ETA

    results_col = ['avg_best_it', 'avg_tr_perf', 'avg_sl_perf', 'avg_ts_perf', 'sd_best_it', 'sd_tr_perf', 'sd_sl_perf',
                   'sd_ts_perf']
    header = pars.keys() + results_col

    results_root = os.path.join("results/", FLAGS.prefix)
    os.makedirs(results_root)

    q = multiprocessing.Queue()

    with open(os.path.join(results_root, 'all.csv'), 'w') as f:
        result_writer = csv.DictWriter(f, header)
        result_writer.writeheader()

        if FLAGS.debug:
            worker_initializer(q, tr, sl, ts)
            map(eval_model, all_model)
        else:
            process = FLAGS.nprocess
            pool = multiprocessing.Pool(process, initializer=worker_initializer,
                                        initargs=(q, tr, sl, ts))

            res = pool.map_async(eval_model, all_model)

        computed_model = 0
        t0 = time.time()
        while not res.ready() or not q.empty():
            if not q.empty():
                eval_res, t = q.get(timeout=10)
                result_writer.writerow(eval_res)
                computed_model += 1
                delta_t = time.time() - t0
                eta = datetime.timedelta(seconds=(delta_t/computed_model) * (len(all_model) - computed_model))
                print "Computed {0} model of {1}. Last took {2:.2f} seconds. [{3} - ETA: {4}]".format(computed_model,
                                                                                           len(all_model), t, str(
                        datetime.timedelta(seconds=delta_t)), str(eta))
            else:
                time.sleep(10)


def main(argv=None):
    FLAGS.prefix += '-' + str(int(time.time()))

    # parse all only once - should fit in memory
    limit = None  # for fast loading while testing - None to load all
    tr_file, sl_file, ts_file = "data/tr-rr.csv", "data/sl-rr.csv", "data/ts-rr.csv"
    tr_data = Ranker.pre_load_csv(tr_file, limit, limit_rr=60)
    sl_data = Ranker.pre_load_csv(sl_file, limit, limit_rr=60)
    ts_data = Ranker.pre_load_csv(ts_file, limit, limit_rr=60)

    if FLAGS.action == 'select':
        model_selection((tr_data, sl_data, ts_data))
    elif FLAGS.action == 'eval':

        c = tf.contrib.rnn.BasicRNNCell(150)
        r = Ranker('ckp/temp_rnn.ckp', c)

        r.train(tr_data, sl_data, 0.001, 0.6, 0.001)

        # concordance
        # tf.reset_default_graph()
        outs = r.get_rank(ts_data)

        with open("results/rnn_concordance.csv", 'wb') as fout:
            fout.write("\n".join(map(lambda x: str(x[0]),outs)))
    else:
        print "invalid action"


if __name__ == "__main__":
    tf.app.run()
