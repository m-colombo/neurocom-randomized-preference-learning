import tensorflow as tf
import csv
import numpy as np
import time
import os
import datetime
import random

import itertools
import multiprocessing

from common import csv_array_reader

from ESN import esn_cell

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('action', None, """select | eval""")
tf.app.flags.DEFINE_string('prefix', "esn", """prefix for results""")

tr_mean, tr_sd = 0.81548140, 0.17976908     # training data stat


class Ranker:
    def __init__(self, tr_data , sl_data, ts_data, **esn_cell_args):

        self.beta = tf.placeholder(tf.float32)
        self.esn_cell_args = esn_cell_args
        self.wout = None

        with tf.variable_scope('TR_DATA'):
            tr = Ranker._build_data_tensor(tr_data)
        with tf.variable_scope("SL_DATA"):
            sl = Ranker._build_data_tensor(sl_data)
        with tf.variable_scope("TS_DATA"):
            ts = Ranker._build_data_tensor(ts_data)

        with tf.variable_scope("Network") as vs:
            pre_emb = self._embedding(tr[0][0], tr[0][1])
            vs.reuse_variables()
            post_emb = self._embedding(tr[1][0], tr[1][1])

            y_trg = tf.concat([tf.tile([1.0], [len(tr_data)]), tf.tile([-1.0], [len(tr_data)])], axis=0)
            x_d1 = pre_emb - post_emb
            x_d2 = -x_d1
            x_d = tf.concat([x_d1, x_d2], axis=0)

            yx = tf.matmul(tf.reshape(y_trg, [1, -1]), x_d)
            xx = tf.matmul(x_d, x_d, transpose_a=True)
            reg = self.beta * tf.eye(pre_emb.shape.dims[1]._value)
            inv = tf.matrix_inverse(xx + reg)
            self.wout = tf.matmul(yx, inv, name="Wout")

        self.tr_performance = self._performance(tr)
        self.sl_performance = self._performance(sl)
        self.ts_performance = self._performance(ts)
        self.S = None

    def _embedding(self, input_t, length_t):
        """ input_t: [ rr_max_length x batch ]
            length_t: [ batch ]
         """
        with tf.variable_scope("Embedding") as vs:
            cell = esn_cell.ESNCell(**self.esn_cell_args)
            (outs, _) = tf.nn.dynamic_rnn(cell=cell, inputs=input_t, sequence_length=tf.squeeze(length_t),
                                          dtype=tf.float32, scope=vs, time_major=True)

            # gather last non-zero outputs
            batch_idx = tf.squeeze(tf.range(length_t.get_shape().as_list()[0]))
            last_outputs = tf.gather_nd(outs, tf.stack([tf.squeeze(length_t - 1),batch_idx], axis=1))

            return last_outputs

    def continuous_prediction(self, rr_streak):
        global tr_mean, tr_sd
        rr = np.array(rr_streak)
        gnorm = tf.convert_to_tensor((rr - tr_mean) / tr_sd, dtype=tf.float32)
        lnorm = tf.convert_to_tensor((rr - np.mean(rr)) / np.std(rr), dtype=tf.float32)
        stacked = tf.stack([lnorm,gnorm], axis=1)
        rr_t=tf.expand_dims(stacked, axis=1)

        with tf.variable_scope("Network", reuse=True) as vs:
            with tf.variable_scope("Embedding", reuse=True) as vs1:
                cell = esn_cell.ESNCell(**self.esn_cell_args)
                (outs, _) = tf.nn.dynamic_rnn(cell=cell, inputs=rr_t, sequence_length=[len(rr_streak)],
                                              dtype=tf.float32, scope=vs1, time_major=True)
            return tf.matmul(self.wout, tf.squeeze(outs), transpose_b=True)

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

    @staticmethod
    def _build_data_tensor(data):
        np.random.shuffle(data)
        rrs, lenghts = zip(*data)

        rrs_t = tf.constant(rrs, dtype=tf.float32)
        rrs_t = tf.transpose(rrs_t, [1, 0, 2])  # time major

        lenghts_t = tf.constant(lenghts, dtype=tf.int32)

        rrs_pre = tf.slice(rrs_t, begin=[0, 0, 0], size=[-1, -1, 2])
        rrs_post = tf.slice(rrs_t, begin=[0, 0, 2], size=[-1, -1, 2])

        length_pre = tf.slice(lenghts_t, begin=[0, 0], size=[-1, 1])
        length_post = tf.slice(lenghts_t, begin=[0, 1], size=[-1, 1])

        return (rrs_pre, length_pre), (rrs_post, length_post)

    def _performance(self, data):
        with tf.variable_scope("Network", reuse=True) as vs:
            pre_emb = self._embedding(data[0][0], data[0][1])
            post_emb = self._embedding(data[1][0], data[1][1])
            x_d = pre_emb - post_emb
            return tf.reduce_mean(tf.cast(tf.equal(tf.sign(tf.matmul(self.wout, x_d, transpose_b=True)), 1), tf.float32))

    def _out(self, rrs, lenghts):
        rrs = tf.transpose(tf.convert_to_tensor(rrs, dtype=tf.float32), perm=[1,0,2])

        lenghts = tf.convert_to_tensor(lenghts, dtype=tf.int32)

        with tf.variable_scope("Network", reuse=True) as vs:
            emb = self._embedding(rrs, lenghts)
            return tf.matmul(self.wout, emb, transpose_b=True)

    def train_oneshot(self, regularizations, keep_ses=False):

        self.S = tf.Session()
        self.S.run(tf.global_variables_initializer())
        results = [self.S.run([self.tr_performance,self.sl_performance,self.ts_performance], feed_dict={self.beta:b}) for b in regularizations]

        if not keep_ses:
            self.S.close()
            self.S = None

        return results

    # def eval(self, data):
    #     rrs, lenghts = zip(*data)
    #     rrs_t = tf.constant(rrs, dtype=tf.float32)
    #     lenghts_t = tf.constant(lenghts, dtype=tf.int32)
    #     rrs_t = tf.transpose(rrs_t, [1, 0, 2])  # time major
    #
    #     rrs_pre = tf.slice(rrs_t, begin=[0, 0, 0], size=[-1, -1, 2])
    #     rrs_post = tf.slice(rrs_t, begin=[0, 0, 2], size=[-1, -1, 2])
    #
    #     length_pre = tf.slice(lenghts_t, begin=[0, 0], size=[-1, 1])
    #     length_post = tf.slice(lenghts_t, begin=[0, 1], size=[-1, 1])
    #
    #     pre, post = (rrs_pre, length_pre), (rrs_post, length_post)
    #
    #     perf = self._performance(pre, post)
    #     self.fetch_variable_to_save()
    #     saver = tf.train.Saver(self.variable_to_save)
    #
    #     with tf.Session() as S:
    #         saver.restore(S, self.checkpoint_name)
    #         return S.run(perf)


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

    tf.reset_default_graph()
    R = Ranker(
        tr_data, sl_data, ts_data,
        num_units=m['esn_size'],
        wr2_scale=m['esn_w2'],
        connectivity=m['esn_connectivity'],
        leaky=m['esn_leaky'],
        win_init=tf.random_normal_initializer(stddev=m['esn_win_sd']))

    for run in range(total_run):
        r = R.train_oneshot(regularizations=m['beta'])
        runs.append(r)

    avgs = np.mean(runs, axis=0)
    stds = np.std(runs, axis=0)

    mk = m.keys()
    mk.remove('beta')

    rows = [
        dict([('beta', m['beta'][b])] + [(k, m[k]) for k in mk] +
            zip(['avg_tr_perf', 'avg_sl_perf', 'avg_ts_perf', 'sd_tr_perf', 'sd_sl_perf', 'sd_ts_perf'],[
                avgs[b][0], avgs[b][1], avgs[b][2],
                stds[b][0], stds[b][1], stds[b][2]
            ]))
        for b in range(len(m['beta']))
    ]

    queue.put([rows, time.time() - t])


def model_selection((tr, sl, ts)):

    pars = {
        'esn_size': [5, 10, 25, 50, 100, 200, 300],
        'esn_connectivity': [0.1],
        'esn_leaky': [0.1, 0.2, 0.3, 0.4, 0.5], # 0.1 -> 0.5
        'esn_w2': [0.85, 0.9, 0.95, 1.0],
        'esn_win_sd': [0.2, 0.4, 0.6],
        'beta': [[0] + [10 ** (-e) for e in range(1,5)]]
    }

    named_pars = {k: [(k, v) for v in pars[k]] for k in pars.keys()}
    all_model = map(dict, itertools.product(*named_pars.values()))
    random.shuffle(all_model, lambda:.3) # to get a less biased ETA

    results_col = ['avg_tr_perf', 'avg_sl_perf', 'avg_ts_perf',
                   'sd_tr_perf', 'sd_sl_perf', 'sd_ts_perf']
    header = pars.keys() + results_col

    results_root = os.path.join("results/", FLAGS.prefix)
    os.makedirs(results_root)

    with open(os.path.join(results_root, 'header.csv'), 'w') as f:
        f.write(",".join(header) + "\n")

    q = multiprocessing.Queue()

    with open(os.path.join(results_root, 'all.csv'), 'w') as f:
        result_writer = csv.DictWriter(f, header)
        result_writer.writeheader()

        process = multiprocessing.cpu_count() / 4
        # process = 4
        pool = multiprocessing.Pool(process, initializer=worker_initializer,
                                    initargs=(q, tr, sl, ts))

        res = pool.map_async(eval_model, all_model)

        computed_model = 0
        t0 = time.time()
        while not res.ready() or not q.empty():
            if not q.empty():
                eval_res, t = q.get(timeout=10)
                result_writer.writerows(eval_res)
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

    limit = None  # for fast loading while testing - None to load all
    tr_file, sl_file, ts_file = "data/tr-rr.csv", "data/sl-rr.csv", "data/ts-rr.csv"
    tr_data = Ranker.pre_load_csv(tr_file, limit)
    sl_data = Ranker.pre_load_csv(sl_file, limit)
    ts_data = Ranker.pre_load_csv(ts_file, limit)

    if FLAGS.action == 'select':
        model_selection((tr_data, sl_data, ts_data))
    elif FLAGS.action == 'eval':
        R = Ranker(
            tr_data, sl_data, ts_data,
            num_units=300,
            wr2_scale=0.95,
            connectivity=0.1,
            leaky=0.1,
            win_init=tf.random_normal_initializer(stddev=0.4))

        r = R.train_oneshot(regularizations=[0.1], keep_ses=True)

        # concordance
        (pre, post) = Ranker._build_data_tensor(ts_data)
        all_rr = tf.concat([pre[0], post[0]], axis=1)
        lenghts = tf.concat([pre[1], post[1]], axis=0)

        with tf.variable_scope("Network", reuse=True) as vs:
            emb = R._embedding(all_rr, lenghts)
            outs = R.S.run(tf.matmul(R.wout, emb, transpose_b=True), feed_dict={R.beta: 0.1})

        with open("results/esn_concordance.csv", 'wb') as fout:
            fout.write("\n".join(map(str,outs[0])))

    else:
        print "Invalid action"


if __name__ == "__main__":
    tf.app.run()
