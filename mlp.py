import tensorflow as tf
import csv
import numpy as np
import time
import os
import datetime
import itertools
import multiprocessing
import random

from common import LearningChecker, csv_array_reader

features = ["LF", "SDNN", "SD1", "SD2", "SVI", "averageNN", "RMSSD", "SDSD", "pNN50", "HF"]

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('action', None, """select | eval""")
tf.app.flags.DEFINE_string('prefix', "mlp", """prefix for results""")
tf.app.flags.DEFINE_boolean('retrain', True, """whether to retrain, only valid for eval action""")
tf.app.flags.DEFINE_boolean('verbose', False, "")
tf.app.flags.DEFINE_boolean('debug', False, "")
tf.app.flags.DEFINE_boolean('ws', True, "Weight sharing")

class Ranker:
    def __init__(self,
                 input_features_name,
                 layers,
                 checkpoint_name,
                 ws
                 ):

        self.checkpoint_name = checkpoint_name
        self.features = input_features_name
        self.layers=layers

        self.variable_to_save = []
        self.weight_cost = 0.0
        self.network_built = False
        self.ws = ws

    def _network_bin(self, input1_t, input2_t, linout=False):

        reuse = True if self.network_built else False

        i = 0
        inp = tf.concat([input1_t, input2_t], axis=1)
        for l in self.layers + [1]:
            i += 1
            out = tf.contrib.layers.fully_connected(
                inputs=inp,
                num_outputs=l,
                activation_fn=(lambda x: x) if i == len(self.layers) + 1 and linout else tf.tanh,
                reuse=reuse,
                scope="NET" + str(i),
                variables_collections=[tf.GraphKeys.WEIGHTS]
            )
            inp = out

        regularizer = tf.contrib.layers.l2_regularizer(1.0)
        weights = tf.get_default_graph().get_collection(tf.GraphKeys.WEIGHTS)

        if not reuse:
            self.weight_cost += sum(map(regularizer, weights))
            self.variable_to_save += weights

        self.network_built = True
        return out

    def _network_ws(self, input1_t, input2_t, linout=False):

        reuse = True if self.network_built else False

        def n(i_t, r):
            i = 0
            for l in self.layers + [1]:
                i += 1
                out = tf.contrib.layers.fully_connected(
                    inputs=i_t,
                    num_outputs=l,
                    activation_fn=(lambda x: x) if i == len(self.layers) + 1 and linout else tf.tanh,
                    reuse=r,
                    scope="NET" + str(i),
                    variables_collections=[tf.GraphKeys.WEIGHTS]
                )
                i_t = out
            return out

        out = (n(input1_t, reuse) - n(input2_t, True))

        regularizer = tf.contrib.layers.l2_regularizer(1.0)
        weights = tf.get_default_graph().get_collection(tf.GraphKeys.WEIGHTS)

        if not reuse:
            self.weight_cost += sum(map(regularizer, weights))
            self.variable_to_save += weights

        self.network_built = True
        return out

    def _loss(self, *args):
        if self.ws:
            return self._loss_ws(*args)
        else:
            return self._loss_bin(*args)

    def _performance(self, *args):
        if self.ws:
            return self._performance_ws(*args)
        else:
            return self._performance_bin(*args)

    def _loss_bin(self, pre, post, regularization):
        # MSE
        return \
            (tf.reduce_mean((2 - (self._network_bin(pre, post))) ** 2) +
                tf.reduce_mean((-2 - (self._network_bin(post, pre))) ** 2)) / 2.0 + self.weight_cost * regularization

    def _performance_bin(self, pre, post):
        # accuracy
        return \
            (tf.reduce_mean(tf.cast(tf.equal(tf.sign((self._network_bin(pre, post))), 1), tf.float32)) +\
            tf.reduce_mean(tf.cast(tf.equal(tf.sign((self._network_bin(post, pre))), -1), tf.float32))) /2.0

    def _loss_ws(self, pre, post, regularization):
        return tf.reduce_mean((2 - (self._network_ws(pre,post))) ** 2) + self.weight_cost * regularization

    def _performance_ws(self, pre, post):
        return tf.reduce_mean(tf.cast(tf.equal(tf.sign((self._network_ws(pre,post))), 1), tf.float32))

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

    @staticmethod
    def _load_file(filename, features):
        pref = ["pre_"+f for f in features]
        postf = ["post_" + f for f in features]

        reader = csv_array_reader(filename, ".")

        pre, post = zip(*[ ([float(row[k1]) for k1 in pref], [float(row[k2]) for k2 in postf]) for row in reader])
        return pre, post

    def get_ranking(self, csv_file, ws=True):
        pre, post = Ranker._load_file(csv_file, self.features)
        all = pre + post
        p1, p2 = random.sample(all, 2)
        p1_t = tf.tile(tf.reshape(tf.convert_to_tensor(p1), shape=[1, 10]), [len(all), 1])
        p2_t = tf.tile(tf.reshape(tf.convert_to_tensor(p2), shape=[1, 10]), [len(all), 1])

        if ws:
            p1_out = self._network_ws(p1_t, all, True)
            p2_out = self._network_ws(p2_t, all, True)
        else:
            p1_out = self._network_bin(p1_t, all, True)
            p2_out = self._network_bin(p2_t, all, True)

        saver = tf.train.Saver(self.variable_to_save)
        with tf.Session() as S:
            saver.restore(S, self.checkpoint_name)
            return S.run([p1_out, p2_out])


def worker_initializer(_q, _tr, _sl, _ts, result_file_path, _result_file_header, _ws):
    global q, tr, sl, ts, result_writer, result_file_header, ws
    q = _q
    tr = _tr
    sl = _sl
    ts = _ts
    ws = _ws

    result_writer = csv.DictWriter(open(os.path.join(result_file_path, str(os.getpid())+'.csv'), "w"), _result_file_header)
    result_file_header = _result_file_header


def eval_model(m):
    global q, tr, sl, ts, result_file_header, ws
    runs = []

    t = time.time()
    total_run = 10

    for run in range(total_run):
        tf.reset_default_graph()
        np.random.shuffle(tr)
        np.random.shuffle(sl)

        R = Ranker(features,
                   layers=m['mlp_size'],
                   checkpoint_name='ckp/temp_' + str(os.getpid()) + '.ckp',
                   ws=ws
                   )

        r = R.train(
            tr_matrix=tr,
            sl_matrix=sl,
            batch_size=150,
            learning_rate=m['learning_rate'],
            momentum=m['momentum'],
            regularization=m['reg']
        )

        ts_res= R.eval(ts)
        runs.append([x for x in r] + [ts_res])

    avgs = np.mean(runs, axis=0)
    stds = np.std(runs, axis=0)

    keys = result_file_header
    row = [m[k] for k in m.keys()] + [
        avgs[0], avgs[2], avgs[3], avgs[4],
        stds[0], stds[2], stds[3], stds[4]]

    res = dict(zip(keys, row))

    q.put([res, time.time() - t])


def model_selection(prefix, ws):
    global features

    # preload all data
    cols = ['pre_' + f for f in features] + ['post_' + f for f in features]
    tr = np.array(
        [[float(row[k]) for k in cols] for row in csv_array_reader('data/tr.csv', ".")])
    sl = np.array(
        [[float(row[k]) for k in cols] for row in csv_array_reader('data/sl.csv', ".")])

    ts = np.array([[float(row[k]) for k in cols] for row in csv_array_reader('data/ts.csv', ".")])

    pars = {
        'mlp_size': [[250,50],[100,10],[5],[10],[25],[50],[100],[250],[10,5], [25,10], [50,10]],
        'learning_rate': [0.01, 1e-3, 1e-4, 1e-5, 1e-6],
        'momentum': [0, 0.3, 0.6],
        'reg': [0, 1e-2, 1e-3, 1e-4, 1e-5]
    }

    named_pars = {k: [(k,v) for v in pars[k]] for k in pars.keys()}
    all_model = map(dict, itertools.product(*named_pars.values()))

    results_col = ['avg_best_it', 'avg_tr_perf', 'avg_sl_perf', 'avg_ts_perf', 'sd_best_it', 'sd_tr_perf', 'sd_sl_perf', 'sd_ts_perf']
    header = pars.keys()+results_col

    base_results = os.path.join("results/", prefix)
    os.makedirs(base_results)

    q = multiprocessing.Queue()

    with open(os.path.join(base_results, 'result.csv'), 'w') as f:
        result_writer = csv.DictWriter(f, header)
        result_writer.writeheader()

        if FLAGS.debug:
            worker_initializer(q, tr, sl, ts, base_results, header, ws)
            res = map(eval_model, all_model)
        else:
            pool = multiprocessing.Pool(multiprocessing.cpu_count() / 2, initializer=worker_initializer, initargs=(q, tr, sl, ts, base_results, header, ws))
            res = pool.map_async(eval_model, all_model)

        computed_model = 0
        t0 = time.time()
        while not res.ready() or not q.empty():
            if not q.empty():
                eval_res, t = q.get(timeout=10)
                result_writer.writerow(eval_res)
                computed_model += 1
                delta_t = time.time() - t0
                eta = datetime.timedelta(seconds=(delta_t / computed_model) * (len(all_model) - computed_model))
                print "Computed {0} model of {1}. Last took {2:.2f} seconds. [{3} - ETA: {4}]".format(computed_model, len(all_model), t, str(datetime.timedelta(seconds=delta_t)), str(eta))
            else:
                time.sleep(10)


def main(argv=None):
    ws_str = '-ws' if FLAGS.ws else '-bin'
    FLAGS.prefix += ws_str
    FLAGS.prefix += '-' + str(int(time.time()))

    if FLAGS.action == "select":
        model_selection(FLAGS.prefix, FLAGS.ws)
    elif FLAGS.action == "eval":
        cols = ['pre_' + f for f in features] + ['post_' + f for f in features]
        tr = np.array(
            [[float(row[k]) for k in cols] for row in csv_array_reader('data/tr.csv', ".")])
        sl = np.array(
            [[float(row[k]) for k in cols] for row in csv_array_reader('data/sl.csv', ".")])

        if FLAGS.ws:
            R = Ranker(features,
                       layers=[100,10],
                       checkpoint_name='ckp/mlp'+ws_str+'.ckp',
                       ws=True
                       )

            r = R.train(
                tr_matrix=tr,
                sl_matrix=sl,
                batch_size=150,
                learning_rate=1e-2,
                momentum=0.3,
                regularization=1e-4
            )
        else:
            R = Ranker(features,
                       layers=[100, 10],
                       checkpoint_name='ckp/mlp' + ws_str + '.ckp',
                       ws=True
                       )

            r = R.train(
                tr_matrix=tr,
                sl_matrix=sl,
                batch_size=150,
                learning_rate=1e-2,
                momentum=0.3,
                regularization=1e-4
            )

        ranking = R.get_ranking('data/ts.csv', FLAGS.ws)
        z = zip(ranking[0], ranking[1])
        m = map((lambda x: str(x[0][0]) + ',' + str(x[1][0])), z)
        j = "\n".join(m)
        with open("results/"+FLAGS.prefix+"_ts-ranking.csv", 'wb') as f:
            f.write("p1out,p2out\n")
            f.write(j)
    else:
        print "Invalid action"

if __name__ == "__main__":
    tf.app.run()
