import pickle
import argparse
import time


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', default="", type=str)
    parser.add_argument('--truths', default=None, type=str)
    parser.add_argument('--raw', default=False, action="store_true")
    option = parser.parse_args()
    print(option)    
    start = time.time()

    if not option.raw:
        truths = pickle.load(open(option.truths, "r"))
        query_heads, query_tails = truths.values()
    
    hits10, hits5, hits1 = 0, 0, 0
    ranks, rranks = 0., 0.

    lines = [l.strip().split(",") for l in open(option.preds).readlines()]
    line_cnt = len(lines)

    for l in lines:
        assert(len(l) > 3)
        q, h, t = l[0:3]
        this_preds = l[3:]
        assert(h == this_preds[-1])
        hitted10, hitted5, hitted1 = 0., 0., 0.

        if not option.raw:
            if q.startswith("inv_"):
                q_ = q[len("inv_"):]
                also_correct = query_heads[(q_, t)]
            else:
                also_correct = query_tails[(q, t)]
            also_correct = set(also_correct)
            assert(h in also_correct)
            #this_preds_filtered = [j for j in this_preds[:-1] if not j in also_correct] + this_preds[-1:]
            this_preds_filtered = set(this_preds[:-1]) - also_correct
            this_preds_filtered.add(this_preds[-1])
            if len(this_preds_filtered) <= 10:
                hitted10 = 1.
            if len(this_preds_filtered) <= 5:
                hitted5 = 1.
            if len(this_preds_filtered) <= 1:
                hitted1 = 1.
            rank = len(this_preds_filtered)
        else:
            if len(this_preds) <= 10:
                hitted10 = 1.
            if len(this_preds) <= 5:
                hitted5 = 1.
            if len(this_preds) <= 1:
                hitted1 = 1.
            rank = len(this_preds)
        
        hits10 += hitted10
        hits5 += hitted5
        hits1 += hitted1
        ranks += rank
        rranks += 1. / rank

    print("Hits at top 10 is %0.4f" % (hits10 / line_cnt))
    print("Hits at top 5 is %0.4f" % (hits5 / line_cnt))
    print("Hits at top 1 is %0.4f" % (hits1 / line_cnt))

    print("Mean rank %0.2f" % (1. * ranks / line_cnt))
    print("Mean Reciprocal Rank %0.4f" % (1. * rranks / line_cnt))

    print("Time %0.3f mins" % ((time.time() - start) / 60.))
    print("="*36 + "Finish" + "="*36)


if __name__ == "__main__":
    evaluate()
