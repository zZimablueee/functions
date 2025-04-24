#还是尝试适配csv文件的
import pandas as pd
import chess
import chess.engine
import math
import sys
import traceback

# ------------------ Helper Functions ------------------
def move_accuracy_percent(before, after):
    if after >= before:
        return 100.0
    win_diff = before - after
    raw = 103.1668100711649 * math.exp(-0.04354415386753951 * win_diff) - 3.166924740191411
    return max(min(raw + 1, 100), 0)


def winning_chances_percent(cp):
    m = -0.00368208
    chances = 2 / (1 + math.exp(m * cp)) - 1
    return 50 + 50 * max(min(chances, 1), -1)


def harmonic_mean(values):
    n = len(values)
    if n == 0:
        return 0
    recip = [1/x for x in values if x]
    return n / sum(recip) if recip else 0


def std_dev(seq):
    if not seq:
        return 0.5
    mean = sum(seq) / len(seq)
    var = sum((x - mean)**2 for x in seq) / len(seq)
    return math.sqrt(var)


def volatility_weighted_mean(accuracies, win_chances, is_white):
    weights = []
    for i in range(len(accuracies)):
        base = 2*i + (1 if is_white else 2)
        start = max(base - 2, 0)
        end = min(base + 2, len(win_chances)-1)
        sub = win_chances[start:end+1]
        w = max(min(std_dev(sub), 12), 0.5)
        weights.append(w)
    wsum = sum(a*w for a,w in zip(accuracies, weights))
    total = sum(weights)
    return wsum/total if total else 0

# ------------------ Core Processing ------------------
def process_moves(moves_list, engine, depth, is_verbose=False):
    board = chess.Board()
    accuracies_white, accuracies_black = [], []
    win_chances = []
    prev_eval = 17  # initial for win chances

    # record initial win chance
    win_chances.append(winning_chances_percent(prev_eval))

    move_number = 1
    for move_uci in moves_list:
        try:
            move = chess.Move.from_uci(move_uci)
            san = board.san(move)
        except Exception:
            if is_verbose:
                print(f"Invalid move UCI: {move_uci} at board {board.fen()}")
            break
        board.push(move)

        result = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = result['score'].white().score(mate_score=1000)

        before_w = winning_chances_percent(prev_eval)
        after_w = winning_chances_percent(score)
        win_chances.append(after_w)

        if board.turn == chess.WHITE:
            before = 100 - before_w
            after = 100 - after_w
        else:
            before = before_w
            after = after_w

        acc = move_accuracy_percent(before, after)

        if board.turn == chess.BLACK:
            cp_loss = max(prev_eval - score, 0)
            accuracies_white.append(acc)
        else:
            cp_loss = max(score - prev_eval, 0)
            accuracies_black.append(acc)

        if is_verbose:
            prefix = f"{move_number:3}." if board.turn == chess.BLACK else "    "
            print(f"{prefix} {san:<6} Eval: {score/100:.2f}, Loss: {cp_loss:3}, Acc: {acc:.1f}%, Win: {after_w:.1f}%")
        if board.turn == chess.WHITE:
            move_number += 1
        prev_eval = score

    # compute per-game statistics
    avg_cp_w = 0  # not tracked in this CSV mode
    avg_cp_b = 0
    hm_w = harmonic_mean(accuracies_white)
    vm_w = volatility_weighted_mean(accuracies_white, win_chances, True)
    final_w = (hm_w + vm_w) / 2
    hm_b = harmonic_mean(accuracies_black)
    vm_b = volatility_weighted_mean(accuracies_black, win_chances, False)
    final_b = (hm_b + vm_b) / 2

    return {
        'harmonic_white': hm_w,
        'weighted_white': vm_w,
        'accuracy_white': final_w,
        'harmonic_black': hm_b,
        'weighted_black': vm_b,
        'accuracy_black': final_b
    }

# ------------------ Main Entry ------------------
def analyze_csv(path_csv, engine_path, threads=1, depth=12, is_verbose=False, output_csv=None):
    # read CSV
    df = pd.read_csv(path_csv, sep='\t')  # or comma

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({'Threads': threads})

    results = []
    for idx, row in df.iterrows():
        moves = row['Moves'].split(',')
        stats = process_moves(moves, engine, depth, is_verbose)
        stats['index'] = idx
        results.append(stats)

    engine.quit()

    res_df = pd.DataFrame(results).set_index('index')
    final = pd.concat([df, res_df], axis=1)

    if output_csv:
        final.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    else:
        print(final)

    return final

# ------------------ Command Line ------------------
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python compute_acc_csv.py <csv_path> <engine_path> <depth> [threads] [-o output.csv] [-verbose]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    engine_path = sys.argv[2]
    depth = int(sys.argv[3])
    threads = 1
    output_csv = None
    is_verbose = False

    for arg in sys.argv[4:]:
        if arg.isdigit():
            threads = int(arg)
        elif arg.startswith('-o'):
            output_csv = arg.split('=',1)[1]
        elif arg == '-verbose':
            is_verbose = True

    analyze_csv(csv_path, engine_path, threads, depth, is_verbose, output_csv)


#E:\VSCodeProjects\crm\CreateDatabase\eda\InfoExtraction\csv2.py
#  python csv2.py C:\Users\Administrator\Desktop\findcode\output\master_games.pgn_processed.csv C:\Users\Administrator\Desktop\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe 12 2 -o=C:\Users\Administrator\Desktop\findcode\output\jieguo.csv -verbose

