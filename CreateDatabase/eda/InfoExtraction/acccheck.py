#测试我的精度函数是否正确的
import math
import chess
import os
import sys
import chess.pgn
import chess.engine
import csv
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import asyncio
from tqdm import tqdm
# 在代码头部添加（重要！）
from stockfish import Stockfish  # 确保此语句在调用Stockfish之前
# 🔧 关键：强制更换 event loop（解决部分 Windows 异步问题）
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



csv_file="C:\\Users\\Administrator\\Desktop\\projects\\test\\2\\original_dataset1.csv"
STOCKFISH_PATH ="C:/Users/Administrator/Desktop/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
STOCKFISH_DEPTH=12
OUTPUT_CSV = "set1_result.csv"  # 输出路径


# 计算胜率差（和原公式一致）
def _calculate_delta(cp_before, cp_after):
    def cp_to_win(cp):
        return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)
    return abs(cp_to_win(cp_before) - cp_to_win(cp_after))
# 辅助函数：从 Stockfish 的评估字典中提取 cp 分值
def get_cp(evaluation):
    # evaluation 形如 {'type': 'cp', 'value': 34} 或 {'type': 'mate', 'value': 2}
    if evaluation["type"] == "cp":
        return evaluation["value"]
    elif evaluation["type"] == "mate":
        # 如果是 mate，按正负10000 处理
        return 10000 if evaluation["value"] > 0 else -10000
    return 0

# 逐步分析棋局准确性（完全按照你原来的评估逻辑）
def generate_mediate_df(csv_file, STOCKFISH_PATH, depth=20):
    raw_df = pd.read_csv(csv_file, encoding='latin1')
    mediate_df = raw_df.copy()
    mediate_df['white_accuracy'] = None
    mediate_df['black_accuracy'] = None

    # 使用 stockfish 包来替代 chess.engine.SimpleEngine
    stockfish = Stockfish(path=STOCKFISH_PATH, depth=depth)
    # 若需要，可设置其他参数，例如线程数、hash 大小等

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
                        # 如果遇到非法走法则跳出本局分析
                        break

                    fen_before=board.fen()
                    stockfish.set_fen_position(fen_before)
                    eval_before=stockfish.get_evaluation()
                    cp_before=get_cp(eval_before)

                    board.push(move)

                    fen_after=board.fen()
                    stockfish.set_fen_position(fen_after)
                    eval_after=stockfish.get_evaluation()
                    cp_after=get_cp(eval_after)
                


                    # 判断本步走法是哪方走的：
                    # 注意：执行 push() 后，board.turn 表示下一步走棋的颜色，
                    # 所以走完走法前的局面时，当前走棋方为与 board.turn 相反的一方
                    moved_color = 'white' if board.turn == chess.BLACK else 'black'

                    # 根据原公式计算走法准确率
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

            # 记录本局的平均准确率
            mediate_df.at[idx, 'white_accuracy'] = sum(white_acc) / len(white_acc) if white_acc else None
            mediate_df.at[idx, 'black_accuracy'] = sum(black_acc) / len(black_acc) if black_acc else None
            pbar.update(1)

    return mediate_df

# ===== 主函数入口 =====
if __name__ == "__main__":
    mediate_df = generate_mediate_df(
        csv_file=csv_file,
        STOCKFISH_PATH=STOCKFISH_PATH,
        depth=STOCKFISH_DEPTH
    )
    mediate_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 分析完成！结果已保存至: {OUTPUT_CSV}")
    