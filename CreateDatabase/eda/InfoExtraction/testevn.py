import pandas as pd
import chess
import chess.engine
import chess.pgn
import subprocess
import time
from tqdm import tqdm

# ===== 配置区域 =====
CSV_FILE = "C:\\Users\\Administrator\\Desktop\\simple eda\\testfunctions\\original_dataset.csv"
STOCKFISH_PATH = "C:/Users/Administrator/Desktop/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
OUTPUT_CSV = "analysis_result.csv"  # 输出路径
STOCKFISH_DEPTH = 12  # 分析深度
TIMEOUT_LIMIT = 10  # 每一步最多等待10秒
# ===================

# 获取 Stockfish 评估函数（带超时机制）
def get_eval_from_stockfish(engine, board_fen, depth=12, timeout=10):
    engine.stdin.write(f"position fen {board_fen}\n".encode())
    engine.stdin.write(f"go depth {depth}\n".encode())
    engine.stdin.flush()

    score = None
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            print(f"⚠️ 超时！跳过该局面: {board_fen}")
            return 0

        line = engine.stdout.readline().decode("utf-8", errors="ignore")
        if "score cp" in line:
            parts = line.strip().split()
            try:
                idx = parts.index("cp")
                score = int(parts[idx+1])
            except:
                continue
        elif "mate" in line:
            return 10000
        elif "bestmove" in line:
            break

    return score if score is not None else 0

# 计算准确率（转为0~1的精度评分）
def centipawn_to_accuracy(score):
    score = abs(score)
    if score > 300:
        return 0.0
    return max(0.0, 1.0 - (score / 300))

# 主函数：读取数据、调用引擎、保存中间结果
def generate_mediate_df(csv_file, STOCKFISH_PATH, depth=12):
    df = pd.read_csv(csv_file)
    mediate_df = df.copy()
    mediate_df['centipawn'] = None
    mediate_df['white_accuracy'] = None
    mediate_df['black_accuracy'] = None

    engine = subprocess.Popen(
        STOCKFISH_PATH,
        universal_newlines=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    try:
        for idx, row in tqdm(mediate_df.iterrows(), total=len(mediate_df), desc="♟ 分析中"):
            moves_str = row['Moves']
            moves = moves_str.strip().split(',')
            board = chess.Board()

            evals = []
            for move in moves:
                try:
                    board.push_uci(move)
                    eval_score = get_eval_from_stockfish(engine, board.fen(), depth=depth, timeout=TIMEOUT_LIMIT)
                    evals.append(eval_score)
                except Exception as e:
                    print(f"❌ 无效走法跳过: {move} 错误: {e}")
                    evals.append(0)

            if evals:
                final_eval = evals[-1]
                mediate_df.at[idx, 'centipawn'] = final_eval
                mediate_df.at[idx, 'white_accuracy'] = centipawn_to_accuracy(final_eval)
                mediate_df.at[idx, 'black_accuracy'] = centipawn_to_accuracy(-final_eval)

    finally:
        engine.terminate()

    return mediate_df

# === 启动脚本 ===
if __name__ == "__main__":
    result_df = generate_mediate_df(
        csv_file=CSV_FILE,
        STOCKFISH_PATH=STOCKFISH_PATH,
        depth=STOCKFISH_DEPTH
    )
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 分析完成！结果已保存到: {OUTPUT_CSV}")
