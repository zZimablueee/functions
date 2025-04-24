#用claude
import io
import math
import sys
import traceback
import csv
import chess.engine
import chess


def get_eval_str(score, board):
    if score.is_mate():
        if score.relative.mate() > 0:
            mating_side = "White" if board.turn else "Black"
        else:
            mating_side = "Black" if board.turn else "White"
        return "Mate in " + str(abs(score.relative.mate())) + " for " + mating_side
    else:
        return str(score.white().score() / 100.0)


def move_accuracy_percent(before, after):
    if after >= before:
        return 100.0
    else:
        win_diff = before - after
        raw = 103.1668100711649 * math.exp(-0.04354415386753951 * win_diff) + -3.166924740191411
        return max(min(raw + 1, 100), 0)


def winning_chances_percent(cp):
    multiplier = -0.00368208
    chances = 2 / (1 + math.exp(multiplier * cp)) - 1
    return 50 + 50 * max(min(chances, 1), -1)


def harmonic_mean(values):
    n = len(values)
    if n == 0:
        return 0
    reciprocal_sum = sum(1 / x for x in values if x)
    return n / reciprocal_sum if reciprocal_sum else 0


def std_dev(seq):
    if len(seq) == 0:
        return 0.5  # Return the minimum weight if sub-sequence is empty
    mean = sum(seq) / len(seq)
    variance = sum((x - mean) ** 2 for x in seq) / len(seq)
    return math.sqrt(variance)


def volatility_weighted_mean(accuracies, win_chances, is_white):
    weights = []
    for i in range(len(accuracies)):
        base_index = i * 2 + 1 if is_white else i * 2 + 2
        start_idx = max(base_index - 2, 0)
        end_idx = min(base_index + 2, len(win_chances) - 1)

        sub_seq = win_chances[start_idx:end_idx+1]
        weight = max(min(std_dev(sub_seq), 12), 0.5)
        weights.append(weight)

    weighted_sum = sum(a*w for a,w in zip(accuracies,weights))
    total_weight = sum(weights)
    weighted_mean = weighted_sum / total_weight if total_weight else 0

    return weighted_mean


def process_csv(csv_file, engine, depth, is_verbose):
    all_games = []
    game_count = 0
    prev_evaluation = 17
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file, delimiter='\t')
        
        for row in csv_reader:
            game_count += 1
            if is_verbose:
                print(f'\n—— Analyzing Game {game_count} ——')
                print(f'White: {row["White"]} ({row["WhiteElo"]}), Black: {row["Black"]} ({row["BlackElo"]})')
            
            # 解析走法
            moves_str = row.get('Moves', '')
            if not moves_str:
                if is_verbose:
                    print(f'Game {game_count}: Skipped (no moves)')
                continue
                
            moves_list = moves_str.split(',')
            
            game_acc_white, game_acc_black = [], []
            game_cp_white, game_cp_black = 0, 0
            game_win_chances = []
            
            board = chess.Board()
            move_number = 1
            current_eval = prev_evaluation
            
            for move_idx, move_str in enumerate(moves_list):
                try:
                    move = chess.Move.from_uci(move_str)
                    if move not in board.legal_moves:
                        print(f'Illegal move {move} detected at {board.fen()}')
                        break
                        
                    san_move = board.san(move)
                    board.push(move)
                    
                    result = engine.analyse(board, chess.engine.Limit(depth=depth))
                    score = result["score"].white().score(mate_score=1000)
                    
                    win_before_white = winning_chances_percent(prev_evaluation)
                    win_after_white = winning_chances_percent(score)
                    game_win_chances.append(win_after_white)
                    
                    if board.turn == chess.WHITE:  # 黑方刚走完
                        win_before = 100 - win_before_white
                        win_after = 100 - win_after_white
                    else:  # 白方刚走完
                        win_before = win_before_white
                        win_after = win_after_white
                        
                    accuracy = move_accuracy_percent(win_before, win_after)
                    
                    if board.turn == chess.BLACK:  # 白方刚走完
                        cp_loss = 0 if score > current_eval else current_eval - score
                        game_cp_white += cp_loss
                        game_acc_white.append(accuracy)
                    else:  # 黑方刚走完
                        cp_loss = 0 if score < current_eval else score - current_eval
                        game_cp_black += cp_loss
                        game_acc_black.append(accuracy)
                        
                    current_eval = score
                    prev_evaluation = score
                    
                    if is_verbose:
                        color = "White" if move_idx % 2 == 0 else "Black"
                        move_number_str = f'{(move_idx // 2) + 1}.' if move_idx % 2 == 0 else "   "
                        print(
                            f'{move_number_str} {san_move:5} ({color}): Eval: {get_eval_str(result["score"], board):5}, '
                            f'Centipawn Loss: {cp_loss:3}, Accuracy %: {accuracy:3.0f}, Win %: {win_after_white:2.0f}')
                        
                    if board.turn == chess.WHITE:
                        move_number += 1
                        
                except Exception as e:
                    print(f"Error processing move {move_str}: {e}")
                    break
            
            # 计算游戏统计数据
            avg_cp_white = game_cp_white / len(game_acc_white) if game_acc_white else 0
            avg_cp_black = game_cp_black / len(game_acc_black) if game_acc_black else 0
            
            harmonic_white = harmonic_mean(game_acc_white) if game_acc_white else 0
            weighted_white = volatility_weighted_mean(game_acc_white, game_win_chances, True)
            final_acc_white = (harmonic_white + weighted_white) / 2
            
            harmonic_black = harmonic_mean(game_acc_black) if game_acc_black else 0
            weighted_black = volatility_weighted_mean(game_acc_black, game_win_chances, False)
            final_acc_black = (harmonic_black + weighted_black) / 2
            
            all_games.append((row["White"], row["Black"], avg_cp_white, final_acc_white, avg_cp_black, final_acc_black))
            
    return all_games


def analyze_csv(csv_file_path, engine_path, threads, depth, is_verbose):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": threads})
    
    try:
        all_games = process_csv(csv_file_path, engine, depth, is_verbose)
        
        # 输出结果
        total_games = len(all_games)
        total_avg_cp_white = total_avg_cp_black = 0.0
        total_acc_white = total_acc_black = 0.0
        
        print("\n===== 分析结果 =====")
        for game_idx, (white, black, avg_cp_white, acc_white, avg_cp_black, acc_black) in enumerate(all_games, 1):
            print(f'\n—— 对局 {game_idx}: {white} vs {black} ——')
            print(f'白方: {avg_cp_white:.1f}cp损失, {acc_white:.1f}%准确率')
            print(f'黑方: {avg_cp_black:.1f}cp损失, {acc_black:.1f}%准确率')
            
            total_avg_cp_white += avg_cp_white
            total_avg_cp_black += avg_cp_black
            total_acc_white += acc_white
            total_acc_black += acc_black
        
        if total_games > 0:
            print("\n===== 总体统计 =====")
            print(f'分析对局数: {total_games}')
            print(f'白方平均: {total_avg_cp_white/total_games:.1f}cp损失, {total_acc_white/total_games:.1f}%准确率')
            print(f'黑方平均: {total_avg_cp_black/total_games:.1f}cp损失, {total_acc_black/total_games:.1f}%准确率')
    
    finally:
        engine.quit()


def print_usage():
    print("用法: python chess_accuracy_csv.py [depth] [threads] [engine_path] -file=path_to_csv_file [-verbose]")
    print("示例:")
    print("    python chess_accuracy_csv.py 16 2 ./stockfish -file=games.csv")
    print("    python chess_accuracy_csv.py 16 2 ./stockfish -file=games.csv -verbose")


if __name__ == "__main__":
    is_verbose_arg = False
    try:
        if len(sys.argv) < 4:
            print("错误: 至少需要三个参数。")
            print_usage()
            sys.exit(1)

        depth_arg = int(sys.argv[1])
        threads_arg = int(sys.argv[2])
        engine_path_arg = sys.argv[3]
        csv_file_path = None

        for arg in sys.argv[4:]:
            if arg.startswith("-file="):
                csv_file_path = arg.split("=", 1)[1]
            elif arg == "-verbose":
                is_verbose_arg = True

        if not csv_file_path:
            print("错误: 未指定CSV文件路径。")
            print_usage()
            sys.exit(1)

        analyze_csv(csv_file_path, engine_path_arg, threads_arg, depth_arg, is_verbose_arg)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        if is_verbose_arg:
            traceback.print_exc()
        sys.exit(1)


#E:\VSCodeProjects\crm\CreateDatabase\eda\InfoExtraction\csv3.py

# python chess_accuracy_csv.py 16 2 ./C:\Users\Administrator\Desktop\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe -file=C:\Users\Administrator\Desktop\findcode\output\master_games.pgn_processed.csv -verbose

