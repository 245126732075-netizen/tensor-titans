"""
NAVY — Main Competition Entry Point (bot.py).

Acts as the tournament entry point invoking the pure-Python UCI engine.
Supports:
  - Default: Standard UCI protocol loop (Cutechess / Arena / Lichess / Judgement harness)
  - --test : Comprehensive self-tests (Perft, Tactics, Draw rules, Timers, UCI)
  - --play : Interactive CLI game against the bot in terminal
"""

import sys
from engine import UCIEngine, Board, Searcher, WHITE, BLACK


def play_interactive():
    print("=" * 60)
    print("NAVY: Interactive Terminal Match")
    print("=" * 60)
    print("Enter moves in UCI format (e.g., e2e4, g1f3, e7e8q).")
    print("Type 'quit' or 'exit' to stop.\n")

    side_choice = input("Play as White (w) or Black (b)? [w]: ").strip().lower()
    human_is_white = side_choice != "b"

    board = Board()
    searcher = Searcher(tt_size_mb=32)

    while True:
        # Display board
        print("\n" + "=" * 25)
        for r in range(7, -1, -1):
            row_chars = []
            for f in range(8):
                sq = (r << 4) | f
                p = board.board[sq]
                from engine import PIECE_NAMES
                row_chars.append(PIECE_NAMES[p])
            print(f"{r + 1}  {' '.join(row_chars)}")
        print("   a b c d e f g h")
        print("=" * 25)

        legals = board.generate_legal_moves()
        if not legals:
            if board.in_check(board.turn):
                winner = "Black" if board.turn == WHITE else "White"
                print(f"\nCheckmate! {winner} wins!")
            else:
                print("\nStalemate! Draw!")
            break

        if board.is_repetition() or board.is_fifty_moves() or board.is_insufficient_material():
            print("\nDraw declared by tournament rules!")
            break

        is_human = (board.turn == WHITE and human_is_white) or (board.turn == BLACK and not human_is_white)

        if is_human:
            while True:
                user_input = input(f"\nYour move ({'White' if board.turn == WHITE else 'Black'}): ").strip()
                if user_input.lower() in ("quit", "exit"):
                    print("Game terminated.")
                    return
                m = board.parse_uci(user_input)
                if m is not None:
                    board.make_move(m)
                    break
                print(f"Illegal move '{user_input}'! Legal moves: {', '.join([m.to_uci() for m in legals[:8]])}...")
        else:
            print("\nNAVY is thinking...", flush=True)
            best_move, score, depth = searcher.search(board, max_time_seconds=1.5)
            print(f"NAVY Move: {best_move.to_uci()} (eval: {score} cp, depth: {depth})")
            board.make_move(best_move)


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("--test", "-t"):
            import test_suite
            test_suite.run_all_tests()
            return
        elif arg in ("--play", "-p"):
            play_interactive()
            return

    # Default tournament execution: Standard UCI loop via STDIN/STDOUT
    engine = UCIEngine()
    engine.run()


if __name__ == "__main__":
    main()
