import copy
import os
import re
import time
import json
import math
from tqdm import tqdm
import argparse
from datasets import load_dataset
import itertools

from retrievers import calculate_retrieval_metrics
import functools
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics','theoremqa_questions', "theoremqa_theorems",
                                 'stackoverflow','sustainable_living','aops','leetcode'])
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--score_file', type=str, default=None)
    parser.add_argument('--input_k', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--reasoning', type=str, default=None)
    args = parser.parse_args()

    # if os.path.exists(args.rerank_score_file):
    #     print(f"Rerank score file {args.rerank_score_file} already exists.")
    #     exit()

    if args.reasoning is not None:
        raw_examples = load_dataset('xlangai/bright', f"{args.reasoning}_reason", cache_dir=args.cache_dir)[args.task]
    else:
        raw_examples = load_dataset('xlangai/bright', 'examples',cache_dir=args.cache_dir)[args.task]

    # raw_examples = load_dataset('xlangai/bright', 'examples', cache_dir=args.cache_dir)[args.task]
    examples = {}
    for e in raw_examples:
        examples[e['id']] = e
    if args.long_context:
        doc_pairs = load_dataset('xlangai/bright', 'long_documents', cache_dir=args.cache_dir)[args.task]
    else:
        doc_pairs = load_dataset('xlangai/bright', 'documents', cache_dir=args.cache_dir)[args.task]
    documents = {}
    for d in doc_pairs:
        documents[d['id']] = d['content']
    with open(args.score_file) as f:
        all_scores = json.load(f)
    new_scores = copy.deepcopy(all_scores)

    with open("rank1_sustainable_living_llama_score_rank_results.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    # import pdb; pdb.set_trace()

    # extract doc order
    dids = []
    for qid,scores in tqdm(all_scores.items()):
        docs = []
        sorted_scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)[:args.input_k]
        for did, _ in sorted_scores:
            docs.append([did, documents[did]])
        dids.append([did for did, _ in sorted_scores])
    # import pdb; pdb.set_trace()

    # for binarizing llama
    # for qid,scores in tqdm(all_scores.items()):
    #     for i, output in enumerate(data[int(qid)]['outputs']):
    #         if output.rfind("Relevance judgement: Yes") != -1:
    #             data[int(qid)]['ranking'][dids[int(qid)][i]] = 1.0
    #         else:
    #             data[int(qid)]['ranking'][dids[int(qid)][i]] = 0.0

    # import pdb; pdb.set_trace()
    # for combing with BM25
    # for qid,scores in tqdm(all_scores.items()):
    #     for i, _ in enumerate(data[int(qid)]['scores']):
    #         data[int(qid)]['ranking'][dids[int(qid)][i]] = data[int(qid)]['ranking'][dids[int(qid)][i]]*100 + scores[dids[int(qid)][i]]

    for qid,scores in tqdm(all_scores.items()):
        new_scores[qid] = data[int(qid)]['ranking']

    # import pdb; pdb.set_trace()

    # code for averaging across multiple rerankings
    # zero the values for every entry in the value dicts
    # for qid, scores in new_scores.items():
    #     for doc_id in scores:
    #         new_scores[qid][doc_id] = 0

    # for idxi in range(5):
    #     with open(f"r1_score_5x_{idxi}_rank_r_results.jsonl", "r") as f:
    #         data = [json.loads(line) for line in f]
    #         for qid,scores in tqdm(new_scores.items()):
    #             for doc_id in scores:
    #                 new_scores[qid][doc_id] += max(data[int(qid)]['ranking'][doc_id], 0.0)
    # for qid,scores in tqdm(new_scores.items()):
    #     for doc_id in scores:
    #         new_scores[qid][doc_id] /= 5
    #         # combine with BM25
    #         new_scores[qid][doc_id] = new_scores[qid][doc_id]*100 + all_scores[qid][doc_id]

    # oracle best performance
    # for qid, scores in new_scores.items():
    #     for doc_id in scores:
    #         new_scores[qid][doc_id] = 0

    # for idxi in range(5):
    #     with open(f"r1_score_5x_{idxi}_rank_r_results.jsonl", "r") as f:
    #         data = [json.loads(line) for line in f]
    #         for qid,scores in tqdm(new_scores.items()):
    #             for doc_id in scores:
    #                 new_scores[qid][doc_id] += max(data[int(qid)]['ranking'][doc_id], 0.0)
    # for qid,scores in tqdm(new_scores.items()):
    #     for doc_id in scores:
    #         new_scores[qid][doc_id] /= 5
    #         # combine with BM25
    #         new_scores[qid][doc_id] = new_scores[qid][doc_id]*100 + all_scores[qid][doc_id]
                  
        


    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in raw_examples:
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        for i in e["excluded_ids"]:
            if i in documents:
                ground_truth[e['id']][i] = 0

    results = calculate_retrieval_metrics(results=new_scores, qrels=ground_truth)
    import pdb; pdb.set_trace()
    # with open(args.rerank_score_file.replace(".json", "_results.json"), 'w') as f:
    #     json.dump(results, f, indent=2)
