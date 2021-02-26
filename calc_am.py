import argparse
import logging
import string
import numpy as np
import codecs
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--hyp_file", type=str, help="path to hypothesis file")
parser.add_argument("--ref_file", type=str, help="path to reference file")
parser.add_argument("--model_path", type=str, help="am_model_path")
parser.add_argument("--num_test", type=int, help="total number of test cases")
parser.add_argument("--batch_size", type=int, help="batch size", default=32)
parser.add_argument("--save_path", type=str, help="path to save system level score")
args = parser.parse_args()

def calc_am_single(hyp_list, ref_list):
    score_mat = cosine_similarity(np.array(hyp_list), np.array(ref_list))
    return score_mat.diagonal()


if __name__=='__main__':
    model = SentenceTransformer(args.model_path)

    logging.info("Loading hypothesis file -------------------------------------------------------")
    with codecs.open(args.hyp_file, mode='r', encoding='utf-8') as rf:
        hyp_lines = rf.readlines()
    hyp_lines = [l.strip() for l in hyp_lines]
    hyp_array = []
    for i in range(0, len(hyp_lines), args.batch_size):
        sentences = hyp_lines[i:i+args.batch_size]
        embeddings = model.encode(sentences)
        hyp_array.extend(embeddings)
    
    assert len(hyp_array) == args.num_test, "wrong number of hypotheses, please check the number of references"

    logging.info("Loading reference file -------------------------------------------------------")
    with codecs.open(args.ref_file, mode='r', encoding='utf-8') as rf:
        ref_lines = rf.readlines()
    ref_line = [l.strip() for l in ref_lines]
    ref_array = []
    for i in range(0, len(ref_lines), args.batch_size):
        sentences = ref_lines[i:i+args.batch_size]
        embeddings = model.encode(sentences)
        ref_array.extend(embeddings)

    assert len(ref_array) == args.num_test, "wrong number of inferences, please check the number of references"
    
    # calculate AM score
    logging.info("Computing am score ---------------------------------------------------")
    full_scores = []
    hyp_arr = np.array(hyp_array)
    ref_arr = np.array(ref_array)
    scores = calc_am_single(hyp_arr, ref_arr).tolist()

    with codecs.open(args.save_path, mode='w', encoding='utf-8') as wf:
        for s in scores:
            wf.write(str(s) + '\n')
    logging.info("Done saving am score to {} ---------------------------------------------------".format(args.save_path))
    
    
