"""
NAVY — Windows AI vs AI Match Runner.

Runs head-to-head matches between any two UCI-compliant chess engines on Windows
via standard process pipes (STDIN / STDOUT).

Examples:
  # 1. NAVY vs Stockfish (White: NAVY, Black: Stockfish)
  python play_match.py --engine1 "python bot.py" --engine2 "bin\\stockfish.exe" --movetime 500

  # 2. Stockfish vs NAVY (Colors reversed)
  python play_match.py --engine1 "bin\\stockfish.exe" --engine2 "python bot.py" --movetime 500

  # 3. NAVY Self-Play (AI vs AI)
  python play_match.py --engine1 "python bot.py" --engine2 "python bot.py" --movetime 400
"""

import sys
import os
import time
import subprocess
import argparse
from typing import Optional, List, Tuple
from engine import Board, Move, WHITE, BLACK, PIECE_NAMES


class UCIEngineProcess:
    def __init__(self, name: str, command: str):
        self.name = name
        self.command = command
        # Start engine process with piped STDIN/STDOUT
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            bufsize=1
        )
        self.init_uci()

    def send(self, cmd: str) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()

    def readline(self) -> str:
        if self.proc.poll() is None:
            return self.proc.stdout.readline().strip()
        return ""

    def init_uci(self) -> None:
        self.send("uci")
        while True:
            line = self.readline()
            if line == "uciok" or not line:
                break
        self.send("isready")
        while True:
            line = self.readline()
            if line == "readyok" or not line:
                break

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.send("isready")
        while True:
            line = self.readline()
            if line == "readyok" or not line:
                break

    def get_best_move(self, moves_history: List[str], movetime_ms: int) -> Tuple[str, float]:
        pos_cmd = "position startpos"
        if moves_history:
            pos_cmd += " moves " + " ".join(moves_history)
        self.send(pos_cmd)

        go_cmd = f"go movetime {movetime_ms}"
        t0 = time.time()
        self.send(go_cmd)

        best_move = ""
        while True:
            line = self.readline()
            if not line and self.proc.poll() is not None:
                break
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_move = parts[1]
                break

        elapsed = time.time() - t0
        return best_move, elapsed

    def terminate(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def render_board(board: Board) -> str:
    lines = ["  +-----------------+"]
    for r in range(7, -1, -1):
        row_str = f"{r + 1} | "
        for f in range(8):
            sq = (r << 4) | f
            p = board.board[sq]
            row_str += PIECE_NAMES[p] + " "
        row_str += "|"
        lines.append(row_str)
    lines.append("  +-----------------+")
    lines.append("    a b c d e f g h")
    return "\n".join(lines)


def run_match(engine1_cmd: str, engine1_name: str,
              engine2_cmd: str, engine2_name: str,
              movetime_ms: int = 500, max_moves: int = 150,
              pgn_output: str = "match.pgn") -> None:

    print("=" * 65)
    print("NAVY: WINDOWS AI VS AI MATCH RUNNER")
    print("=" * 65)
    print(f"White Engine : {engine1_name}  [{engine1_cmd}]")
    print(f"Black Engine : {engine2_name}  [{engine2_cmd}]")
    print(f"Time Control : {movetime_ms} ms per move")
    print(f"PGN Export   : {pgn_output}")
    print("=" * 65)

    print("\n[+] Spawning Engine 1 (White)...", end="", flush=True)
    p_white = UCIEngineProcess(engine1_name, engine1_cmd)
    print(" Ready!")

    print("[+] Spawning Engine 2 (Black)...", end="", flush=True)
    p_black = UCIEngineProcess(engine2_name, engine2_cmd)
    print(" Ready!\n")

    p_white.new_game()
    p_black.new_game()

    board = Board()
    moves_history: List[str] = []
    move_details: List[Tuple[int, str, str, float]] = []

    result = "*"
    termination = "Playing"

    try:
        ply = 0
        while ply < max_moves * 2:
            turn_color = board.turn
            curr_engine = p_white if turn_color == WHITE else p_black
            curr_name = engine1_name if turn_color == WHITE else engine2_name
            color_str = "White" if turn_color == WHITE else "Black"
            move_num = (ply // 2) + 1

            # Check game status before move
            legals = board.generate_legal_moves()
            if not legals:
                if board.in_check(turn_color):
                    result = "0-1" if turn_color == WHITE else "1-0"
                    winner = engine2_name if turn_color == WHITE else engine1_name
                    termination = f"Checkmate! {winner} wins!"
                else:
                    result = "1/2-1/2"
                    termination = "Draw by Stalemate!"
                break

            if board.is_repetition():
                result = "1/2-1/2"
                termination = "Draw by Threefold Repetition!"
                break

            if board.is_fifty_moves():
                result = "1/2-1/2"
                termination = "Draw by 50-Move Rule!"
                break

            if board.is_insufficient_material():
                result = "1/2-1/2"
                termination = "Draw by Insufficient Material!"
                break

            # Request move from current engine
            best_move_str, elapsed = curr_engine.get_best_move(moves_history, movetime_ms)

            # Validate move legality
            parsed_move = board.parse_uci(best_move_str)
            if parsed_move is None or parsed_move not in legals:
                result = "0-1" if turn_color == WHITE else "1-0"
                winner = engine2_name if turn_color == WHITE else engine1_name
                termination = f"Disqualification: {curr_name} made illegal move '{best_move_str}'! {winner} wins!"
                break

            # Apply move
            board.make_move(parsed_move)
            moves_history.append(best_move_str)
            move_details.append((move_num, color_str, best_move_str, elapsed))

            # Display move info
            dots = ". " if turn_color == WHITE else "... "
            print(f"Move {move_num}{dots}{best_move_str}  ({curr_name}, {elapsed:.3f}s)")
            if turn_color == BLACK or ply < 4:
                print(render_board(board))
                print("-" * 35)

            ply += 1

        if ply >= max_moves * 2 and result == "*":
            result = "1/2-1/2"
            termination = f"Draw by maximum moves limit ({max_moves}) reached!"

    finally:
        p_white.terminate()
        p_black.terminate()

    print("\n" + "=" * 65)
    print(f"MATCH TERMINATION: {termination}")
    print(f"FINAL RESULT     : {result}")
    print("=" * 65)

    # Save PGN
    pgn_lines = [
        '[Event "NAVY Exhibition Match"]',
        '[Site "Windows Localhost"]',
        f'[Date "{time.strftime("%Y.%m.%d")}"]',
        f'[White "{engine1_name}"]',
        f'[Black "{engine2_name}"]',
        f'[Result "{result}"]',
        f'[TimeControl "{movetime_ms/1000.0}+0"]',
        f'[Termination "{termination}"]',
        ""
    ]
    # Build PGN move text
    move_text = ""
    for i in range(0, len(moves_history), 2):
        move_num = (i // 2) + 1
        w_move = moves_history[i]
        b_move = moves_history[i + 1] if i + 1 < len(moves_history) else ""
        move_text += f"{move_num}. {w_move} "
        if b_move:
            move_text += f"{b_move} "
    move_text += result
    pgn_lines.append(move_text)

    with open(pgn_output, "w", encoding="utf-8") as f:
        f.write("\n".join(pgn_lines) + "\n")
    print(f"[+] Match PGN successfully saved to: {os.path.abspath(pgn_output)}\n")


def main():
    parser = argparse.ArgumentParser(description="Windows AI vs AI Chess Match Runner")
    parser.add_argument("--engine1", type=str, default="python bot.py", help="Command to run White engine")
    parser.add_argument("--name1", type=str, default="NAVY", help="Display name for White engine")
    parser.add_argument("--engine2", type=str, default="bin\\stockfish.exe", help="Command to run Black engine")
    parser.add_argument("--name2", type=str, default="Stockfish_19", help="Display name for Black engine")
    parser.add_argument("--movetime", type=int, default=500, help="Move budget in milliseconds (default: 500)")
    parser.add_argument("--maxmoves", type=int, default=80, help="Maximum moves before draw (default: 80)")
    parser.add_argument("--pgn", type=str, default="match.pgn", help="PGN export file path")

    args = parser.parse_args()

    run_match(
        engine1_cmd=args.engine1,
        engine1_name=args.name1,
        engine2_cmd=args.engine2,
        engine2_name=args.name2,
        movetime_ms=args.movetime,
        max_moves=args.maxmoves,
        pgn_output=args.pgn
    )


if __name__ == "__main__":
    main()
