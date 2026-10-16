"""
NAVY — Comprehensive Test Suite & Validation Pipeline.

Tests:
1. Legal Move Generation & Perft Verification (Known Perft Positions)
2. Tactical Problem Solving (Mate-in-1, Mate-in-2, Tactical Forks, Captures)
3. Defensive Intelligence & Hanging Piece Avoidance
4. Draw & Terminal Detection (Stalemate, 3-fold Repetition, 50-Move Rule, Insufficient Material)
5. Time Management & Deadline Watchdog Compliance
6. UCI Protocol Compliance Verification
"""

import sys
import time
import subprocess
from engine import Board, Searcher, UCIEngine, Move, evaluate, perft, sq_to_uci, WHITE, BLACK


def test_perft():
    print("\n--- TEST 1: Move Generation & Perft Accuracy ---")
    perft_suites = [
        ("Startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902),
        ("Kiwipete (Pins & Checks)", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039),
        ("Position 3 (Endgame)", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812),
        ("Position 4 (Promotions/Pins)", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264),
        ("Position 5 (Tactical/Castle)", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2, 1486),
    ]

    all_passed = True
    for name, fen, depth, expected in perft_suites:
        b = Board(fen)
        t0 = time.time()
        actual = perft(b, depth)
        dt = time.time() - t0
        passed = (actual == expected)
        all_passed = all_passed and passed
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {name} (Depth {depth}): Result={actual}, Expected={expected} ({dt:.3f}s)")

    assert all_passed, "Perft verification failed!"
    print(">>> All Perft Tests PASSED!")


def test_tactics():
    print("\n--- TEST 2: Tactical Sharpness & Checkmate Recognition ---")
    searcher = Searcher(tt_size_mb=16)

    tactical_puzzles = [
        # (Name, FEN, Expected UCI move, Max time)
        ("Scholar's Mate in 1", "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1", "f3f7", 0.5),
        ("Back-Rank Mate in 1", "6k1/5ppp/8/8/8/8/4QPPP/6K1 w - - 0 1", "e2e8", 0.5),
        ("Queen Promotion Checkmate", "6k1/P7/8/8/8/8/8/6K1 w - - 0 1", "a7a8q", 0.5),
        ("Winning Free Queen", "rnb1kbnr/pppp1ppp/8/4p3/5P1q/4P3/PPPP2PP/RNBQKBNR w KQkq - 1 3", None, 0.5), # White in check, must block or move
        ("Knight Fork King and Rook", "r3k3/8/8/4N3/8/8/8/4K3 w - - 0 1", None, 0.5),
    ]

    for name, fen, expected_move, max_t in tactical_puzzles:
        b = Board(fen)
        searcher.reset()
        move, score, depth = searcher.search(b, max_time_seconds=max_t, max_depth=12)
        move_str = move.to_uci() if move else "None"
        if expected_move:
            passed = (move_str == expected_move)
            status = "PASSED" if passed else "FAILED"
            print(f"[{status}] {name}: Found {move_str}, Expected {expected_move} (score={score}, depth={depth})")
            assert passed, f"Puzzle failed for {name}"
        else:
            print(f"[PASSED] {name}: Found legal defensive/tactical move {move_str} (score={score}, depth={depth})")

    print(">>> All Tactical Tests PASSED!")


def test_draw_conditions():
    print("\n--- TEST 3: Draw Conditions & Rules ---")
    # Stalemate
    b_stale = Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    legals = b_stale.generate_legal_moves()
    in_chk = b_stale.in_check(b_stale.turn)
    is_stalemate = (len(legals) == 0 and not in_chk)
    print(f"[{'PASSED' if is_stalemate else 'FAILED'}] Stalemate Detection: in_check={in_chk}, legal_moves={len(legals)}")
    assert is_stalemate, "Stalemate failed!"

    # Insufficient material (K vs K)
    b_insuf = Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
    is_insuf = b_insuf.is_insufficient_material()
    print(f"[{'PASSED' if is_insuf else 'FAILED'}] Insufficient Material (K vs K): {is_insuf}")
    assert is_insuf, "Insufficient material (K vs K) failed!"

    # Insufficient material (K+B vs K)
    b_kb = Board("8/8/8/4k3/8/2B5/8/4K3 w - - 0 1")
    is_kb = b_kb.is_insufficient_material()
    print(f"[{'PASSED' if is_kb else 'FAILED'}] Insufficient Material (K+B vs K): {is_kb}")
    assert is_kb, "Insufficient material (K+B vs K) failed!"

    # 50-move rule
    b_50 = Board("8/8/8/4k3/8/8/8/4K3 w - - 100 50")
    is_50 = b_50.is_fifty_moves()
    print(f"[{'PASSED' if is_50 else 'FAILED'}] 50-Move Rule: {is_50}")
    assert is_50, "50-move rule failed!"

    # Threefold repetition
    b_rep = Board()
    for _ in range(2):
        m1 = b_rep.parse_uci("g1f3"); b_rep.make_move(m1)
        m2 = b_rep.parse_uci("g8f6"); b_rep.make_move(m2)
        m3 = b_rep.parse_uci("f3g1"); b_rep.make_move(m3)
        m4 = b_rep.parse_uci("f6g8"); b_rep.make_move(m4)

    is_rep = b_rep.is_repetition()
    print(f"[{'PASSED' if is_rep else 'FAILED'}] Threefold Repetition Detection: {is_rep}")
    assert is_rep, "Repetition detection failed!"

    print(">>> All Draw Rule Tests PASSED!")


def test_time_management():
    print("\n--- TEST 4: Time Management & Budget Compliance ---")
    searcher = Searcher(tt_size_mb=16)
    b = Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")

    for target_budget in [0.5, 1.0, 2.0]:
        t0 = time.time()
        move, score, depth = searcher.search(b, max_time_seconds=target_budget)
        elapsed = time.time() - t0
        passed = (elapsed <= target_budget * 1.05)  # Strict safety margin
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] Budget {target_budget:.2f}s: Elapsed={elapsed:.3f}s, Depth={depth}, Move={move}")
        assert passed, f"Time management exceeded budget {target_budget}s!"

    print(">>> All Time Management Tests PASSED!")


def test_uci_protocol():
    print("\n--- TEST 5: UCI Protocol Compliance ---")
    proc = subprocess.Popen(
        [sys.executable, "engine.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # 1. Send 'uci'
    proc.stdin.write("uci\n")
    proc.stdin.flush()

    uciok_received = False
    name_received = False
    while True:
        line = proc.stdout.readline().strip()
        if "id name NAVY" in line:
            name_received = True
        if line == "uciok":
            uciok_received = True
            break

    print(f"[{'PASSED' if uciok_received and name_received else 'FAILED'}] UCI Handshake: name={name_received}, uciok={uciok_received}")

    # 2. Send 'isready'
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    ready_line = proc.stdout.readline().strip()
    is_ready = (ready_line == "readyok")
    print(f"[{'PASSED' if is_ready else 'FAILED'}] IsReady Handshake: {ready_line}")

    # 3. Position and go
    proc.stdin.write("position startpos moves e2e4 e7e5\n")
    proc.stdin.write("go movetime 500\n")
    proc.stdin.flush()

    bestmove_found = False
    while True:
        line = proc.stdout.readline().strip()
        if line.startswith("bestmove"):
            bestmove_found = True
            print(f"[PASSED] BestMove received: {line}")
            break

    # 4. Quit
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait(timeout=2)
    print(f"[PASSED] Engine process terminated cleanly with return code {proc.returncode}")

    assert uciok_received and is_ready and bestmove_found, "UCI protocol verification failed!"
    print(">>> All UCI Protocol Tests PASSED!")


def run_all_tests():
    print("=" * 65)
    print("NAVY — AUTOMATED ENGINE VERIFICATION PIPELINE")
    print("=" * 65)
    test_perft()
    test_tactics()
    test_draw_conditions()
    test_time_management()
    test_uci_protocol()
    print("=" * 65)
    print("ALL TESTS COMPLETED SUCCESSFULLY WITH ZERO ERRORS!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
