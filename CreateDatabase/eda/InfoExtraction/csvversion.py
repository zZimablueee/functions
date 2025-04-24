#用来放我对compute_acc.py的适配

#环境配置
import csv
import math
import sys
import traceback
from io import TextIOWrapper
import chess.engine
import chess.pgn
import pandas as pd

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
    
def winning_chances_percent(score_obj):
    if isinstance(score_obj, chess.engine.PovScore):
        effective_score = score_obj.relative
    else:
        effective_score = score_obj

    if effective_score.is_mate():
        return 100.0 if effective_score.mate() > 0 else 0.0
    else:
        cp_value = effective_score.score()
        return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp_value)) - 1)


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
#以上都是说不用改直接用的

def process_csv(csv_path, engine, depth, is_verbose, board):
    all_games=[]
    win_before_adj=0.0
    win_after_adj=0.0

    df=pd.read_csv(csv_path)

    def get_score_value(score_obj):
        if isinstance(score_obj, chess.engine.PovScore):
            raw_score = score_obj.relative
        else:
            raw_score = score_obj
    
        if raw_score.is_mate():
            return 10000 if raw_score.mate() > 0 else -10000
        else:
            return raw_score.score() / 100.0  # ✅ 直接调用score()


    for idx,row in df.iterrows():
        board=chess.Board()
        init_result=engine.analyse(board,chess.engine.Limit(depth=1))
        prev_score_obj=init_result['score'].pov(chess.WHITE)
        game_acc_white,game_acc_black=[],[]
        game_cp_white,game_cp_black=0,0
        game_win_chances=[]
        prev_eval=0.17
        current_eval=prev_eval

        moves=row['Moves'].split(',')
        current_eval=prev_eval

        for move_num,san_move in enumerate(moves,1):
            san_move=san_move.strip()
            try:
                move=board.parse_san(san_move)
                if move not in board.legal_moves:
                    print(f'illegal move {san_move} @ number {move_num} step')
                    break
                board.push(move)
            except Exception as e:
                if is_verbose:
                    print(f'Parsing failed: {san_move} | Error: {str(e)}')
                break

            result=engine.analyse(board,chess.engine.Limit(depth=depth))
            score_obj=result['score'].pov(board.turn)
            current_score=get_score_value(score_obj)
            if board.turn==chess.BLACK:
                cp_loss=max(current_eval-current_score,0)
            else:
                cp_loss=max(current_score-current_eval,0)

            current_eval=current_score
            prev_eval=current_score

            win_before=winning_chances_percent(prev_score_obj.pov(board.turn))
            win_after=winning_chances_percent(score_obj.pov(board.turn))
            game_win_chances.append(win_after)

            if board.turn==chess.WHITE:
                win_before_adj=100-win_before
                win_after_adj=100-win_after
            else:
                win_before_adj=win_before
                win_after_adj=win_after
            accuracy=move_accuracy_percent(win_before_adj,win_after_adj)

            if board.turn==chess.BLACK:
                game_cp_white+=cp_loss
                game_acc_white.append(accuracy)
            else:
                game_cp_black+=cp_loss
                game_acc_black.append(accuracy)


            if is_verbose:
                turn_symbol='⚪' if board.turn == chess.WHITE else '⚫'
                print(
                    f"[{row['White']} vs {row['Black']}] {turn_symbol} "
                    f"{move_num:2}. {san_move:5}: Eval: {get_eval_str(score_obj, board):5} "
                    f"Loss: {cp_loss:3} | Acc: {accuracy:3.0f}% | Win%: {win_after:5.1f}"
                )

            prev_score_obj=score_obj
            avg_white = game_cp_white/len(game_acc_white) if game_acc_white else 0
            avg_black = game_cp_black/len(game_acc_black) if game_acc_black else 0
        
            harmonic_white = harmonic_mean(game_acc_white) if game_acc_white else 0
            weighted_white = volatility_weighted_mean(game_acc_white, game_win_chances, True)
            final_white = (harmonic_white + weighted_white) / 2
            
            harmonic_black = harmonic_mean(game_acc_black) if game_acc_black else 0
            weighted_black = volatility_weighted_mean(game_acc_black, game_win_chances, False)
            final_black = (harmonic_black + weighted_black) / 2

            all_games.append({
            'event': row['Event'],
            'white': row['White'],
            'black': row['Black'],
            'result': row['Result'],
            'accuracy_white': final_white,
            'accuracy_black': final_black,
            'cp_loss_white': avg_white,
            'cp_loss_black': avg_black})
    return pd.DataFrame(all_games)

def analyze_csv(csv_path, engine_path, threads, depth, is_verbose):
    # 初始化引擎
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": threads})
    board = chess.Board()

    try:
        # 处理CSV数据
        games_df = process_csv(csv_path, engine, depth, is_verbose, board)

        # —— 详细输出段 ——
        if is_verbose and not games_df.empty:
            required_columns = ['Event', 'White', 'Black', 'Termination',
                               'cp_loss_white', 'accuracy_white',
                               'cp_loss_black', 'accuracy_black']
            if all(col in games_df.columns for col in required_columns):
                for idx, row in games_df.iterrows():
                    print(f"\n—— Game {idx+1} ({row['Event']}) ——")
                    print(f"White [{row['White']}]: {row['cp_loss_white']:.0f}cp loss, {row['accuracy_white']:.0f}%")
                    print(f"Black [{row['Black']}]: {row['cp_loss_black']:.0f}cp loss, {row['accuracy_black']:.0f}%")
                    print(f"Termination: {row['Termination']}")
            else:
                print("[Warning] CSV缺少必要列，无法输出详细信息")

        # —— 统计计算 ——
        stats = {'total_games': 0}  # 默认值
        if not games_df.empty:
            stats = {
                'total_games': len(games_df),
                'avg_cp_loss_white': games_df.get('cp_loss_white', pd.Series()).mean(skipna=True),
                'avg_acc_white': games_df.get('accuracy_white', pd.Series()).mean(skipna=True),
                'avg_cp_loss_black': games_df.get('cp_loss_black', pd.Series()).mean(skipna=True),
                'avg_acc_black': games_df.get('accuracy_black', pd.Series()).mean(skipna=True)
            }
            
    finally:
        engine.quit()  # 确保引擎始终关闭

    return stats, games_df




def print_usage():
    print("Usage: python chess_accuracy.py [csv-path] [engine_path] [threads] [depth] [-verbose]")
    print("Example:")
    print("    python avg_cp_loss.py games.csv ./stockfish 4 18 -verbose")
    print("    python avg_cp_loss.py /path/to/data.csv ./stockfish-15 2 16")

if __name__ == "__main__":
    is_verbose_arg = False
    try:
        if len(sys.argv) < 5:
            print("Error: at least three arguments are required.")
            print_usage()
            sys.exit(1)

        csv_path_arg = sys.argv[1]         # 第一个参数为CSV路径
        engine_path_arg = sys.argv[2]      # 第二个参数为引擎路径
        threads_arg = int(sys.argv[3])     # 第三个参数为线程数
        depth_arg = int(sys.argv[4])       # 第四个参数为分析深度
        
        for arg in sys.argv[5:]:
            if arg == "-verbose":
                is_verbose_arg = True
            else:
                print(f'warning: dismiss unknown argument {arg}')

        stats, df = analyze_csv(
        csv_path=csv_path_arg,
        engine_path=engine_path_arg,
        threads=threads_arg,
        depth=depth_arg,
        is_verbose=is_verbose_arg
        )

        print("\nGlobal Statistics:")
        print(f"Total games analyzed: {stats['total_games']}")
        print(f"Average white CP loss: {stats['avg_cp_loss_white']:.1f}cp")
        print(f"Average black CP loss: {stats['avg_cp_loss_black']:.1f}cp")

    except Exception as e:
        print(f"致命错误: {str(e)}", file=sys.stderr)
        if is_verbose_arg:
            traceback.print_exc()
        sys.exit(1)

#python csvversion.py ^ "C:\Users\Administrator\Desktop\findcode\output\chess_com_games_2025-04-21.pgn_processed.csv" ^ "C:/Users/Administrator/Desktop/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe" ^ 6 18 -verbose
#E:\VSCodeProjects\crm\CreateDatabase\eda\InfoExtraction\csvversion.py