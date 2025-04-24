import math
import chess
import chess.engine
import pandas as pd
from tqdm import tqdm
import os
import asyncio
import sys


# ===== 配置路径 =====
csv_file = "C:\\Users\\Administrator\\Desktop\\simple eda\\testfunctions\\original_dataset.csv"
STOCKFISH_PATH = "C:\\Users\\Administrator\\Desktop\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"
STOCKFISH_DEPTH = 20
OUTPUT_CSV = "analyzed_games.csv"  # 输出路径

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def _calculate_delta(cp_before, cp_after):
    """计算胜率差"""
    def cp_to_win(cp):
        return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)
    return abs(cp_to_win(cp_before) - cp_to_win(cp_after))


def generate_mediate_df(csv_file, STOCKFISH_PATH='stockfish', depth=20):
    raw_df = pd.read_csv(csv_file, encoding='latin1')
    mediate_df = raw_df.copy()
    mediate_df['white_accuracy'] = None
    mediate_df['black_accuracy'] = None

    # 初始化引擎
    engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\Administrator\\Desktop\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe")


    try:
        with tqdm(total=len(mediate_df),
                  desc="🔄 棋局分析进度",
                  unit="局",
                  bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [剩余:{remaining}]",
                  dynamic_ncols=True) as pbar:

            for idx, row in mediate_df.iterrows():
                pbar.set_postfix_str(f"当前对局ID: {idx}")
                moves = str(row['Moves']).split(',')
                board = chess.Board()
                white_acc = []
                black_acc = []

                for move_uci in moves:
                    move_uci = move_uci.strip()
                    if not move_uci:
                        continue

                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move not in board.legal_moves:
                            break

                        info_before = engine.analyse(board, chess.engine.Limit(depth=depth))
                        cp_before = info_before['score'].relative.score(mate_score=10000)

                        board.push(move)

                        info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
                        cp_after = info_after['score'].relative.score(mate_score=10000)

                        moved_color = 'white' if board.turn == chess.BLACK else 'black'

                        delta = _calculate_delta(cp_before, cp_after)
                        acc = 103.1668 * math.exp(-0.04354 * delta) - 3.1669
                        clamped_acc = max(0.0, min(100.0, acc))

                        if moved_color == 'white':
                            white_acc.append(clamped_acc)
                        else:
                            black_acc.append(clamped_acc)

                    except Exception as e:
                        print(f"\n❌ 对局 {idx} 发生错误：{str(e)}")
                        break

                mediate_df.at[idx, 'white_accuracy'] = sum(white_acc)/len(white_acc) if white_acc else None
                mediate_df.at[idx, 'black_accuracy'] = sum(black_acc)/len(black_acc) if black_acc else None
                pbar.update(1)

    finally:
        engine.quit()

    return mediate_df


if __name__ == "__main__":
    result_df = generate_mediate_df(
        csv_file=csv_file,
        STOCKFISH_PATH=STOCKFISH_PATH,
        depth=STOCKFISH_DEPTH
    )
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 分析完成！结果已保存至: {OUTPUT_CSV}")
