"""
Does the embedding batch size change the selections?
====================================================
The instructed embeddings of GSM8K pools 0-29 used batch 64 and those of pools 30-99 used batch 8. One pool is embedded at
batch 8 and estimated twice from fresh copies, once from the batch 8 and once from the stored batch 64 embeddings. The batch 64 run is a control: it must reproduce the stored selections.

  python analysis/embed_instructed.py --pool-start 25 --pool-end 26 --cache-root ROOT/b8/cache-gsm8k --batch-size 8
  then estimate_pool of scripts/run_gsm8k_instructed.sh under ROOT/b8/cache-gsm8k and ROOT/b64/cache-gsm8k

Usage: python analysis/embed_batch_check.py --root ROOT [--pool 25]
Output: analysis/embed_batch_check.json
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import plot_gsm8k as P  # noqa: E402

EMB = "google__gemma-2-9b-it_qwenins_embeddings.pkl"


def key(x):
    return x[0], tuple(sorted(str(s) for s in x[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pool", type=int, default=25)
    a = ap.parse_args()
    pool = f"pool{a.pool}"
    stored = os.path.join(ROOT_DIR, "cache-gsm8k", pool)
    old, new = pickle.load(open(os.path.join(stored, EMB), "rb")), pickle.load(open(os.path.join(a.root, "b8", "cache-gsm8k", pool, EMB), "rb"))
    cos, absd, gram, n = [], [], [], 0
    for sT in old:
        for pt, d in old[sT].items():
            e = new[sT][pt]
            assert np.array_equal(d["u_sents"], e["u_sents"]), (sT, pt)
            x, y = np.asarray(d["embeddings"], dtype=np.float64), np.asarray(e["embeddings"], dtype=np.float64)
            cos.append((x * y).sum(1)); absd.append(np.abs(x - y).max(1)); gram.append(np.abs(x @ x.T - y @ y.T).max())
            n += len(x)
    c, m = 1 - np.concatenate(cos), np.concatenate(absd)
    out = dict(pool=a.pool, embeddings=dict(strings=n, identical_rows=int((m == 0).sum()), one_minus_cos_max=float(c.max()), coordinate_diff_max=float(m.max()), gram_entry_diff_max=float(np.max(gram))))
    df = P.load_gsm8k_dataset().head(P.N_QUESTIONS)
    gold = {P.build_math_prompt(r["question"]): r["answer"] for _, r in df.iterrows()}
    metrics = P.make_metrics(gold)
    S = os.path.join(stored, "est10qwenins")
    for lab, other in (("control_batch64", os.path.join(a.root, "b64", "cache-gsm8k", pool, "est10qwenins")), ("batch8", os.path.join(a.root, "b8", "cache-gsm8k", pool, "est10qwenins"))):
        out[lab] = {}
        for method in sorted(os.path.basename(d) for d in glob.glob(f"{S}/0/*")):
            same = tot = 0
            dy, dx = [], []
            for name in sorted({os.path.basename(p) for p in glob.glob(f"{S}/*/{method}/*.pkl")}):
                ya, xa, yb, xb = [], [], [], []
                for p in glob.glob(f"{S}/*/{method}/{name}"):
                    q = p.replace(S, other)
                    if not os.path.exists(q):
                        continue
                    ra, rb = pickle.load(open(p, "rb")), pickle.load(open(q, "rb"))
                    u, v = dict(map(key, ra)), dict(map(key, rb))
                    tot += len(u)
                    same += sum(u[k] == v.get(k) for k in u)
                    ma, mb = metrics(ra), metrics(rb)
                    ya.append(ma[0]); xa.append(ma[1]); yb.append(mb[0]); xb.append(mb[1])
                if ya:
                    dy.append(np.mean(yb) - np.mean(ya)); dx.append(np.mean(xb) - np.mean(xa))
            out[lab][method.rsplit("-", 1)[1]] = dict(records=tot, same_selection=same / max(tot, 1), max_abs_d_pass=float(np.max(np.abs(dy))), mean_d_pass=float(np.mean(dy)), max_abs_d_logdet=float(np.max(np.abs(dx))), mean_d_logdet=float(np.mean(dx)))
            print(lab, method.rsplit("-", 1)[1], out[lab][method.rsplit("-", 1)[1]], flush=True)
    dest = os.path.join(SCRIPT_DIR, "embed_batch_check.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print("Saved", dest)


if __name__ == "__main__":
    main()
