
import math
import chess
import os
import sys
import chess.pgn
import chess.engine

STOCKFISH_PATH = "C:\\Users\\Administrator\\Desktop\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"
STOCKFISH_DEPTH = 15

GAME_META_DATA = "C:\\Users\\Administrator\\Desktop\\simple eda\\testfunctions\\original_dataset.csv"
MOVES_LOG = "C:\\Users\\Administrator\\Desktop\\simple eda\\testfunctions\\logs\\99_moves_log.txt"

log_files = {
    "MISSED_WIN_LOG": "C:\\Users\\Administrator\\Desktop\\simple eda\\testfunctions\\logs\\00_missed_wins_log.txt",  #“错失胜利”日志（键）  
    "BLUNDER_LOG": "C:\\Users\\Administrator\\Desktop\\simple eda\\testfunctions\\logs\\01_blunders_log.txt", #“大错”日志
    "MISTAKE_LOG": "C:\\Users\\Administrator\\Desktop\\simple eda\\testfunctions\\logs\\02_mistakes_log.txt", #“错误”日志
    "INACCURACY_LOG": "C:\\Users\\Administrator\\Desktop\\simple eda\\testfunctions\\logs\\03_inaccuracies_log.txt" #“不准确”日志
}
#log_files是一个字典，键是日志类型名称，值是对应日志文件的路径
#如访问 log_files["MISSED_WIN_LOG"] 将返回 "logs/00_missed_wins_log.txt"

thresholds = {
    "MISSED_WIN_THRESHOLD": 2000,
    "BLUNDER_THRESHOLD": 150,
    "MISTAKE_THRESHOLD": 80,
    "INACCURACY_LOG_THRESHOLD": 30
}
#定义判断错失错误着子类型的得分差异阈值
#MISSED_WIN_THRESHOLD: 如果某种评分或分数的差距大于等于 2000，则可能被认为是错失了明显的胜利机会。；BLUNDER_THRESHOLD: 如果差距大于等于 150，则可能被认为是一次严重的失误。
#MISTAKE_THRESHOLD: 如果差距大于等于 80，则可能被认为是一次一般的错误或失误。；INACCURACY_LOG_THRESHOLD: 如果差距大于等于 30，则可能被记录为一次不准确的行为。
MISSED_WIN_LOG = log_files["MISSED_WIN_LOG"]
MISSED_WIN_THRESHOLD = thresholds["MISSED_WIN_THRESHOLD"]

BLUNDER_LOG = log_files["BLUNDER_LOG"]
BLUNDER_THRESHOLD = thresholds["BLUNDER_THRESHOLD"]

MISTAKE_LOG = log_files["MISTAKE_LOG"]
MISTAKE_THRESHOLD = thresholds["MISTAKE_THRESHOLD"]

INACCURACY_LOG = log_files["INACCURACY_LOG"]
INACCURACY_LOG_THRESHOLD = thresholds["INACCURACY_LOG_THRESHOLD"]
#初始化日志和阈值变量


ACPL_RESULT = "C:\\Users\\Administrator\\Desktop\\simple eda\\testfunctions\\logs\\09_acpl.txt"
ACCURACY_RESULT = "C:\\Users\\Administrator\\Desktop\\simple eda\\testfunctions\\logs\\10_accuracy.txt"



def write_move_to_log(board, played_move, score_diff):
    """
    Writes the move to the appropriate log file based on the score difference.
    根据得分差异将走子记录到相应日志文件中

    Args:
        board (chess.Board): The current board position.当前棋盘位置
        played_move (chess.Move): The move played by the player.玩家的棋步
        score_diff (int): The score difference between the played move and the best move.
                           玩家棋布和最佳走法之间的分数差

    Returns:
        None

    """

    if score_diff >= MISSED_WIN_THRESHOLD:#2000
        log_file = MISSED_WIN_LOG
    elif score_diff >= BLUNDER_THRESHOLD:#150
        log_file = BLUNDER_LOG
    elif score_diff >= MISTAKE_THRESHOLD:#80
        log_file = MISTAKE_LOG
    elif score_diff >= INACCURACY_LOG_THRESHOLD:#30
        log_file = INACCURACY_LOG
    else:
        return

    with open(log_file, "a") as f:   #根据得分差异将走子记录到相应日志文件中
        f.write(f"Move: {board.fullmove_number}.{board.san(played_move)} -> Difference: {score_diff}\n")
        #将生成字符串写入  后面的是f-string  
        # board.fullmove_number 是chess.Board的属性，表示当前完整回合数 进行到哪个回合
        # board.san(played_move) 接受一个Move对象并返回该走法的标准代数记谱法（SAN）表示。
        #score_diff=score_best_move − score_played_move 正值玩家较差，负值玩家较好



# TODO 
# Ref. https://lichess.org/page/accuracy
# Calculate the accuracy of the player for "color" in the game "pgn_file"
# Accuracy% represents how much you deviated from the best moves, i.e. how much your winning chances decreased with each move you made. Indeed in chess, from a chess engine standpoint, good moves don't exist! You can't increase your winning chances by playing a move, only reduce them if you make a mistake. Because if you have a good move to play, then it means the position was already good for you before you played it.
#Accuracy是与最佳走法的偏差程度（每一步走法使胜率下降的程度）
#从引擎的角度看，好的走法并不存在，无法通过下一个走法去提高胜率；因为如果有一个好的走法可供选择，意味着您下这个走法前，局势已经对你有利
# First, calculate Win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * centipawns)) - 1)
#求Win
# Then, compute Accuracy%
# Now that we have a Win% number for each position, we can compute the accuracy of a move by comparing the Win% before and after the move. Here's the equation:
# Accuracy% = 103.1668 * exp(-0.04354 * (winPercentBefore - winPercentAfter)) - 3.1669
def calculate_accuracy(pgn_file, color):
    """
    Calculates the accuracy of the player for the specified color in the given game.
    算某玩家在指定颜色中的准确度
    Args:
        pgn_file (str): The path to the PGN file containing the chess game.
        color (str): The color to calculate the accuracy for ("black" or "white").

    Returns:
        The accuracy percentage of the player.

    Raises:
        FileNotFoundError: If the PGN file is not found.找不到pgn文件属于异常情况

    """

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    with open(pgn_file) as f:
        game = chess.pgn.read_game(f)

    board = game.board()
    accuracy = 0
    total_moves = 0

    for played_move in game.mainline_moves():
        # info_before = engine.analyse(board, chess.engine.Limit(time=1000))
        info_before = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))

        # Win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * centipawns)) - 1)

        win_percent_before = 50 + 50 * (2 / (1 + math.exp(-0.00368208 * info_before["score"].relative.score(mate_score=10000))) - 1)
        #info_before['score'].relative.score(mate_score=10000)获取当前局势的相对评分 mate_score=10000表示胜利评分非常高

        board.push(played_move)

        # info_after = engine.analyse(board, chess.engine.Limit(time=1000))
        info_after = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))

        win_percent_after = 50 + 50 * (2 / (1 + math.exp(-0.00368208 * info_after["score"].relative.score(mate_score=10000))) - 1)

        # Calculate the accuracy of the move. Ref. https://lichess.org/page/accuracy
        accuracy += 103.1668 * math.exp(-0.04354 * (win_percent_before - win_percent_after)) - 3.1669
        total_moves += 1

    engine.quit()
    #用stockfish分析每一步棋前后的胜率；根据胜率差异求每一步的准确性；返回平均准确率
    avg_accuracy = accuracy / total_moves if total_moves > 0 else 0

    # Write the accuracy to the log file ACCURACY_RESULT  把结果写入到准确率结果中
    with open(ACCURACY_RESULT, "a") as f:
        f.write(f"Accuracy: {color.capitalize()}: {avg_accuracy}\n")


    return avg_accuracy

def analyze_game(pgn_file, color):
    """
    Analyzes a chess game stored in a PGN file using Stockfish engine.
    分析pgn文件中的棋局
    Args:    输入参数: 1.pgn文件路径2.要求的ACPL的颜色(白/黑）
        pgn_file (str): The path to the PGN file containing the chess game.
        color (str): The color to calculate the average centipawn loss for ("black" or "white").

    Returns:   返回:指定颜色的平均分心兵损失
        The average centipawn loss for the specified color.

    Raises:  找不到PGN文件的时候返回异常
        FileNotFoundError: If the PGN file or Stockfish engine executable is not found.

    """

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    with open(pgn_file) as f:
        game = chess.pgn.read_game(f)

    #初始化期盼 累计分心兵损失和累计走棋步数都是0
    board = game.board()
    loss = 0
    moves = 0

    for played_move in game.mainline_moves():

        depth = 15
        info = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
        best_move = info["pv"][0]   #stockfish推荐的最佳走法
        score_diff = 0 #初始化分数差异

        if best_move != played_move:
            #分析最佳走法和实际走法的局面
            info1 = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH), root_moves=[best_move])
            info2 = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH), root_moves=[played_move])

            #提取最佳走法和实际走法的评分
            best1 = info1["score"].relative.score(mate_score=10000)
            best2 = info2["score"].relative.score(mate_score=10000)

            #求评分差值  abs是返回绝对值的
            score_diff = abs(best1 - best2)

            # TODO consider how to calculate centipawnloss if a move is a MISSED_WIN?
            score_diff = 0 if score_diff >= MISSED_WIN_THRESHOLD else score_diff

            #将棋布和分数差记录到日志中
            write_move_to_log(board, played_move, score_diff)
       
        with open(MOVES_LOG, "a") as f:  #（假设是黑方）
            # print the number of moves and the move itself
            # f.write(f"Move pair: {board.fullmove_number}\n")
            
            # print the move and the score difference   
            f.write(f"Move: {board.fullmove_number}.{board.san(played_move)} -> Difference: {score_diff} (Black)\n") 
            loss += score_diff #将当前步的分差加到总损失loss里面
            moves += 1 #步数计数器增加1
        with open(MOVES_LOG, "a") as f:  #（假设是白方）
            f.write(f"Move: {board.fullmove_number}.{board.san(played_move)} -> Difference: {score_diff} (White)\n") 
            loss += score_diff
            moves += 1

        #在棋盘上执行实际走法
        board.push(played_move)

    #如果有走棋（即moves数量大于0），求平均分心兵损失
    avg_loss = loss / moves if moves > 0 else 0

    #将ACPL结果写入ACPL_RESULT文件里
    with open(ACPL_RESULT, "a") as f:
        f.write(f"ACPL: {color.capitalize()}: {int(avg_loss)}\n")

    engine.quit()

    return avg_loss


def analyze_game_folder_black_white(pgn_folder):
    """
    Analyzes a folder containing PGN games using Stockfish engine and calculates the average centipawn loss for Black and White.
    用stockfish引擎分析包含pgn棋局的文件夹  并计算黑方和白方的Average Centipawn Loss
    Args:
        pgn_folder (str): The path to the folder containing the PGN games.

    Returns:  返回值 无
        None

    Raises:
        FileNotFoundError: If the PGN folder or Stockfish engine executable is not found.

    """

    #初始化变量 记录黑方和白方的总损失值和总分析的游戏数量
    total_black_loss = 0
    total_white_loss = 0
    total_games = 0

    # Skip the header  跳过文件头部内容，打开含元数据的文件
    with open(os.path.join(pgn_folder, GAME_META_DATA)) as f:
        lines = f.readlines()[1:]
    
    #遍历每一行 提取棋局颜色和文件名
    for line in lines:
        color, filename = line.strip().split(",")
        pgn_file = os.path.join(pgn_folder, filename)
        with open(ACPL_RESULT, "a") as f:
            f.write(f"Analyzing game: {filename}: {color}: ") 
        if color.lower() == "black":
            black_loss = analyze_game(pgn_file, "black")
            white_loss = 0
        elif color.lower() == "white":
            black_loss = 0
            white_loss = analyze_game(pgn_file, "white")
        else:
            black_loss = 0
            white_loss = 0

        if color.lower() == "black":
            total_black_loss += black_loss
        elif color.lower() == "white":
            total_white_loss += white_loss

        total_games += 1

    avg_black_loss = total_black_loss / total_games if total_games > 0 else 0
    avg_white_loss = total_white_loss / total_games if total_games > 0 else 0

    avg_centipawn_loss = (avg_black_loss + avg_white_loss) / 2

    #写入结果 返回总体平均损失值
    with open(ACPL_RESULT, "a") as f:
        f.write(f"Overall Average Centipawn Loss: {int(avg_centipawn_loss)}\n") 
        
    return avg_centipawn_loss
    
# Reset the log files
#清空/重置多个日志文件的内容 
def reset_logs():
    open(ACPL_RESULT, "w").close()  #打开文件后立即关闭相当于清空文件内容
    open(ACCURACY_RESULT, "w").close()

    open(MISSED_WIN_LOG, "w").close()
    open(MOVES_LOG, "w").close()
    open(BLUNDER_LOG, "w").close()
    open(INACCURACY_LOG, "w").close()
    open(MISTAKE_LOG, "w").close()

    return None

def test_calculate_accuracy():
    pgn_file = "games/transformer/10thsep-01.pgn"
    color = "white"
    print(calculate_accuracy(pgn_file, color))
    print("")
    pgn_file = "games/transformer/10thsep-02.pgn"
    color = "black"
    print(calculate_accuracy(pgn_file, color))
    print("")


def test_analyze_game_folder_black_white():
    pgn_folder = "games"
    analyze_game_folder_black_white(pgn_folder)

def test_analyze_game():
    pgn_file = "games/transformer/10thsep-01.pgn"
    color = "white"
    print(analyze_game(pgn_file, color))
    print("")
    pgn_file = "games/transformer/10thsep-02.pgn"
    color = "black"
    print(analyze_game(pgn_file, color))
    print("")

"""
TODO fix 
transformer outputs: 
741.2830261396034
541.2378874130336
"""

if __name__ == "__main__":

    # Reset/blank the log files
    reset_logs()

    # TODO validate the results
    # test_calculate_accuracy()

    # the first argument is the folder containing the games
    pgn_folder = sys.argv[1] if len(sys.argv) > 1 else "./games"
    #sys.argv是一个列表，包含了运行脚本时传入的命令行参数
    #若在命令行中运行python script.py my_games_folder，则
    #sys.argv=["script.py","my_games_folder"]
    analyze_game_folder_black_white(pgn_folder)
