import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from tenacity import retry, stop_after_attempt, wait_random_exponential

from models.wrappers import (
    call_openai_chat,
    call_anthropic_claude,
    call_gemini,
    call_local_model,
)


# Argument Parsing
parser = argparse.ArgumentParser(description="cycle")
parser.add_argument('--model', type=str, default="gpt-3.5-turbo", help='Model name (e.g. gpt-4o, claude-3)')
parser.add_argument('--provider', type=str, default="openai", help='Provider: openai, anthropic, gemini, huggingface')
parser.add_argument('--mode', type=str, default="easy", help='Difficulty mode: easy, medium, hard')
parser.add_argument('--prompt', type=str, default="none", help='Prompting technique: CoT, PROGRAM, etc.')
parser.add_argument('--T', type=int, default=0, help='Temperature (default: 0)')
parser.add_argument('--token', type=int, default=400, help='Max tokens (default: 400)')
parser.add_argument('--SC', type=int, default=0, help='Use self-consistency (default: 0)')
parser.add_argument('--SC_num', type=int, default=5, help='Number of samples for self-consistency')
args = parser.parse_args()

assert args.prompt in [
    "CoT", "none", "0-CoT", "LTM", "PROGRAM", "k-shot", "Instruct",
    "Algorithm", "Recitation", "hard-CoT", "medium-CoT"
]

def translate(edge, n, args):
    Q = ''
    if args.prompt in ["CoT", "k-shot", "Instruct", "Algorithm", "Recitation", "hard-CoT", "medium-CoT"]:
        with open("NLGraph/cycle/prompt/" + args.prompt + "-prompt.txt", "r") as f:
            exemplar = f.read()
        Q += exemplar + "\n\n\n"

    Q += f"In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.\n"
    Q += f"The nodes are numbered from 0 to {n-1}, and the edges are:"
    for (u, v) in edge:
        Q += f" ({u},{v})"
    if args.prompt == "Instruct":
        Q += ". Let's construct a graph with the nodes and edges first."
    Q += "\n"

    if args.prompt == "Recitation":
        Q += f"Q1: Are node {edge[0][0]} and node {edge[0][1]} connected with an edge?\nA1: Yes.\n"
        u = -1
        for i in range(n):
            for j in range(n):
                if [i, j] not in edge:
                    u, v = i, j
                    break
            if u != -1:
                break
        Q += f"Q2: Are node {u} and node {v} connected with an edge?\nA2: No.\n"

    Q += "Q: Is there a cycle in this graph?\nA:"
    match args.prompt:
        case "0-CoT":
            Q += " Let's think step by step:"
        case "LTM":
            Q += " Let's break down this problem:"
        case "PROGRAM":
            Q += " Let's solve the problem by a Python program:"
    return Q

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
def predict(Q_list, args):
    temperature = 0.7 if args.SC else 0
    answer_list = []

    for prompt in Q_list:
        if args.provider == "openai":
            response = call_openai_chat(args.model, prompt, temperature, args.token)
        elif args.provider == "anthropic":
            response = call_anthropic_claude(args.model, prompt, temperature, args.token)
        elif args.provider == "gemini":
            response = call_gemini(args.model, prompt, temperature, args.token)
        elif args.provider == "huggingface":
            response = call_local_model(args.model, prompt)
        else:
            raise ValueError(f"Unsupported provider: {args.provider}")
        answer_list.append(response)
    return answer_list

def log(Q_list, res, answer, args):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    timestamp = bj_dt.now().strftime("%Y%m%d---%H-%M")
    folder = f'log/cycle/{args.model}-{args.mode}-{timestamp}-{args.prompt}'
    if args.SC:
        folder += "+SC"
    os.makedirs(folder, exist_ok=True)

    np.save(os.path.join(folder, "res.npy"), res)
    np.save(os.path.join(folder, "answer.npy"), answer)
    with open(os.path.join(folder, "prompt.txt"), "w") as f:
        for Q in Q_list:
            f.write(Q + "\n\n")
        f.write(f"Acc: {res.sum()}/{len(res)}\n")
        print(args, file=f)

def main():
    if 'OPENAI_API_KEY' not in os.environ:
        raise Exception("Missing OpenAI API Key!")

    res, answer = [], []

    match args.mode:
        case "easy":
            g_num = 3
        case "medium":
            g_num = 1
        case "hard":
            g_num = 1

    batch_num = 20
    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        G_list, Q_list = [], []
        for j in range(i * batch_num, min(g_num, (i + 1) * batch_num)):
            with open(f"NLgraph/cycle/graph/{args.mode}/standard/graph{j}.txt", "r") as f:
                n, m = map(int, f.readline().split())
                edges = [list(map(int, line.strip().split())) for line in f]
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for u, v in edges:
                    G.add_edge(u, v)
                Q = translate(edges, n, args)
                Q_list.append(Q)
                G_list.append(G)

        sc = args.SC_num if args.SC else 1
        sc_list = []
        for k in range(sc):
            print(f"Running call #{k + 1}...")
            answer_list = predict(Q_list, args)
            print(f"Finished call #{k + 1}")
            sc_list.append(answer_list)

        for j in range(len(Q_list)):
            vote = 0
            for k in range(sc):
                ans = sc_list[k][j].lower()
                answer.append(ans)
                p1 = ans.find("there is no cycle")
                p2 = ans.find("there is a cycle")
                p1 = 1000000 if p1 == -1 else p1
                p2 = 1000000 if p2 == -1 else p2
                idx = i * batch_num + j
                if (idx * 2 < g_num and p1 < p2) or (idx * 2 > g_num and p2 < p1):
                    vote += 1
            res.append(1 if vote * 2 >= sc else 0)

    res = np.array(res)
    answer = np.array(answer)
    log(Q_list, res, answer, args)
    print("Final Accuracy:", res.sum(), "/", len(res))

if __name__ == "__main__":
    main()
