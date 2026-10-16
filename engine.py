"""
NAVY — High-Performance Pure-Python Chess Engine.

Zero external dependencies (Standard Library Only).
Full 0x88 Board, Zobrist Hashing, Transposition Table, Quiescence Search,
Iterative Deepening, Alpha-Beta Negamax, PVS, Killer/History Heuristics,
Tapered Evaluation (MG/EG), and Standard UCI Protocol.
"""

import sys
import time
import random
from typing import List, Tuple, Optional, Dict

# =============================================================================
# CONSTANTS & DEFINITIONS
# =============================================================================

WHITE = 1
BLACK = -1

EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

PIECE_NAMES = {
    EMPTY: ".",
    PAWN: "P", KNIGHT: "N", BISHOP: "B", ROOK: "R", QUEEN: "Q", KING: "K",
    -PAWN: "p", -KNIGHT: "n", -BISHOP: "b", -ROOK: "r", -QUEEN: "q", -KING: "k"
}
CHAR_TO_PIECE = {v: k for k, v in PIECE_NAMES.items() if k != EMPTY}

# 0x88 Move Deltas
KNIGHT_DELTAS = (14, 18, 31, 33, -14, -18, -31, -33)
BISHOP_DELTAS = (15, 17, -15, -17)
ROOK_DELTAS = (1, 16, -1, -16)
QUEEN_DELTAS = (1, 16, -1, -16, 15, 17, -15, -17)
KING_DELTAS = (1, 16, -1, -16, 15, 17, -15, -17)

# Castling Rights Bits
# 1: White O-O (WK), 2: White O-O-O (WQ), 4: Black O-O (BK), 8: Black O-O-O (BQ)
CASTLE_MASK = [15] * 128
CASTLE_MASK[0x00] &= ~2   # a1
CASTLE_MASK[0x04] &= ~3   # e1
CASTLE_MASK[0x07] &= ~1   # h1
CASTLE_MASK[0x70] &= ~8   # a8
CASTLE_MASK[0x74] &= ~12  # e8
CASTLE_MASK[0x77] &= ~4   # h8

# Search Scores
INFINITY = 1000000
MATE_SCORE = 100000
MATE_THRESHOLD = 90000

# TT Entry Flags
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

# Evaluation Material Values (Centipawns)
PIECE_VALUES_MG = {EMPTY: 0, PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 20000}
PIECE_VALUES_EG = {EMPTY: 0, PAWN: 120, KNIGHT: 300, BISHOP: 335, ROOK: 540, QUEEN: 950, KING: 20000}
GAME_PHASE_WEIGHTS = {PAWN: 0, KNIGHT: 1, BISHOP: 1, ROOK: 2, QUEEN: 4, KING: 0}
MAX_PHASE = 24

# Piece Square Tables for 64 squares (Rank 1 to 8, File A to H)
# Converted to 0x88 internally
PAWN_PST_MG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     10,  15,  15, -10, -10,  15,  15,  10,
      5,   5,  10,  20,  20,   5,   5,   5,
      0,   0,  10,  25,  25,   0,   0,   0,
      5,   5,  10,  20,  20,   5,   5,   5,
     10,  10,  15,  20,  20,  10,  10,  10,
     50,  50,  50,  50,  50,  50,  50,  50,
      0,   0,   0,   0,   0,   0,   0,   0
]

PAWN_PST_EG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     -5,  -5,  -5,  -5,  -5,  -5,  -5,  -5,
      5,   5,   5,   5,   5,   5,   5,   5,
     15,  15,  15,  15,  15,  15,  15,  15,
     30,  30,  30,  30,  30,  30,  30,  30,
     50,  50,  50,  50,  50,  50,  50,  50,
     80,  80,  80,  80,  80,  80,  80,  80,
      0,   0,   0,   0,   0,   0,   0,   0
]

KNIGHT_PST_MG = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  20,  25,  25,  20,   0, -30,
    -30,   5,  20,  25,  25,  20,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]

KNIGHT_PST_EG = [
    -30, -20, -10, -10, -10, -10, -20, -30,
    -20,  -5,   0,   5,   5,   0,  -5, -20,
    -10,   5,  10,  15,  15,  10,   5, -10,
    -10,   5,  15,  20,  20,  15,   5, -10,
    -10,   5,  15,  20,  20,  15,   5, -10,
    -10,   5,  10,  15,  15,  10,   5, -10,
    -20,  -5,   0,   5,   5,   0,  -5, -20,
    -30, -20, -10, -10, -10, -10, -20, -30
]

BISHOP_PST_MG = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  15,  15,  10,  10, -10,
    -10,   0,  15,  15,  15,  15,   0, -10,
    -10,   5,  15,  15,  15,  15,   5, -10,
    -10,   0,  10,  15,  15,  10,   0, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]

BISHOP_PST_EG = [
    -15,  -5,  -5,  -5,  -5,  -5,  -5, -15,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   5,  10,  10,   5,   0,  -5,
     -5,   0,  10,  15,  15,  10,   0,  -5,
     -5,   0,  10,  15,  15,  10,   0,  -5,
     -5,   0,   5,  10,  10,   5,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
    -15,  -5,  -5,  -5,  -5,  -5,  -5, -15
]

ROOK_PST_MG = [
      0,   0,   5,  10,  10,   5,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0
]

ROOK_PST_EG = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,   5,   5,   5,   5,   5,   5,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
     10,  10,  10,  10,  10,  10,  10,  10,
      0,   0,   0,   0,   0,   0,   0,   0
]

QUEEN_PST_MG = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20
]

QUEEN_PST_EG = [
    -15,  -5,  -5,   0,   0,  -5,  -5, -15,
     -5,   0,   5,   5,   5,   5,   0,  -5,
     -5,   5,  10,  10,  10,  10,   5,  -5,
      0,   5,  10,  15,  15,  10,   5,   0,
      0,   5,  10,  15,  15,  10,   5,   0,
     -5,   5,  10,  10,  10,  10,   5,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -15,  -5,  -5,   0,   0,  -5,  -5, -15
]

KING_PST_MG = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
]

KING_PST_EG = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -10,   0,   0,   0,   0, -10, -30,
    -30,  -5,  20,  25,  25,  20,  -5, -30,
    -30,   0,  25,  35,  35,  25,   0, -30,
    -30,   0,  25,  35,  35,  25,   0, -30,
    -30,  -5,  20,  25,  25,  20,  -5, -30,
    -30, -10,   0,   0,   0,   0, -10, -30,
    -50, -30, -30, -30, -30, -30, -30, -50
]

# Convert 64-square tables to 0x88 arrays
def _to_0x88(table64: List[int]) -> List[int]:
    t = [0] * 128
    for r in range(8):
        for f in range(8):
            sq64 = r * 8 + f
            sq88 = (r << 4) | f
            t[sq88] = table64[sq64]
    return t

PST_MG = {
    PAWN: _to_0x88(PAWN_PST_MG),
    KNIGHT: _to_0x88(KNIGHT_PST_MG),
    BISHOP: _to_0x88(BISHOP_PST_MG),
    ROOK: _to_0x88(ROOK_PST_MG),
    QUEEN: _to_0x88(QUEEN_PST_MG),
    KING: _to_0x88(KING_PST_MG),
}

PST_EG = {
    PAWN: _to_0x88(PAWN_PST_EG),
    KNIGHT: _to_0x88(KNIGHT_PST_EG),
    BISHOP: _to_0x88(BISHOP_PST_EG),
    ROOK: _to_0x88(ROOK_PST_EG),
    QUEEN: _to_0x88(QUEEN_PST_EG),
    KING: _to_0x88(KING_PST_EG),
}

# =============================================================================
# ZOBRIST HASHING
# =============================================================================

# Deterministic PRNG seed for reproducible Zobrist tables
_rng = random.Random(20261016)

ZOBRIST_PIECES = {}
for _sq in range(128):
    if not (_sq & 0x88):
        for _p in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6):
            ZOBRIST_PIECES[(_sq, _p)] = _rng.getrandbits(64)

ZOBRIST_SIDE = _rng.getrandbits(64)
ZOBRIST_CASTLING = [_rng.getrandbits(64) for _ in range(16)]
ZOBRIST_EP_FILE = [_rng.getrandbits(64) for _ in range(8)]


def sq_to_uci(sq: int) -> str:
    return chr(ord('a') + (sq & 7)) + str((sq >> 4) + 1)


def uci_to_sq(s: str) -> int:
    return ((int(s[1]) - 1) << 4) | (ord(s[0].lower()) - ord('a'))


# =============================================================================
# MOVE REPRESENTATION
# =============================================================================

class Move:
    __slots__ = ('from_sq', 'to_sq', 'promo', 'captured', 'flags')

    def __init__(self, from_sq: int, to_sq: int, promo: int = 0, captured: int = 0, flags: int = 0):
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.promo = promo
        self.captured = captured
        self.flags = flags  # 1: ep, 2: castle

    def to_uci(self) -> str:
        s = sq_to_uci(self.from_sq) + sq_to_uci(self.to_sq)
        if self.promo:
            promo_char = {KNIGHT: 'n', BISHOP: 'b', ROOK: 'r', QUEEN: 'q'}.get(self.promo, '')
            s += promo_char
        return s

    def __repr__(self) -> str:
        return self.to_uci()

    def __eq__(self, other) -> bool:
        return (isinstance(other, Move) and
                self.from_sq == other.from_sq and
                self.to_sq == other.to_sq and
                self.promo == other.promo)

    def __hash__(self) -> int:
        return (self.from_sq << 10) | (self.to_sq << 3) | self.promo


# =============================================================================
# 0x88 BOARD ENGINE
# =============================================================================

class Board:
    def __init__(self, fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        self.board = [EMPTY] * 128
        self.turn = WHITE
        self.castling = 15  # 1: WK, 2: WQ, 4: BK, 8: BQ
        self.ep_sq: Optional[int] = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.king_sq = {WHITE: 0x04, BLACK: 0x74}
        self.zobrist = 0
        self.history: List[Tuple[Move, int, Optional[int], int, int, int]] = []
        self.position_history: List[int] = []  # List of hashes for repetition detection
        self.set_fen(fen)

    def set_fen(self, fen: str) -> None:
        self.board = [EMPTY] * 128
        self.history.clear()
        self.position_history.clear()

        parts = fen.strip().split()
        placement = parts[0]
        r = 7
        f = 0
        for ch in placement:
            if ch == '/':
                r -= 1
                f = 0
            elif ch.isdigit():
                f += int(ch)
            else:
                p = CHAR_TO_PIECE[ch]
                sq = (r << 4) | f
                self.board[sq] = p
                if p == KING:
                    self.king_sq[WHITE] = sq
                elif p == -KING:
                    self.king_sq[BLACK] = sq
                f += 1

        self.turn = WHITE if len(parts) < 2 or parts[1] == 'w' else BLACK

        self.castling = 0
        if len(parts) >= 3 and parts[2] != '-':
            for ch in parts[2]:
                if ch == 'K': self.castling |= 1
                elif ch == 'Q': self.castling |= 2
                elif ch == 'k': self.castling |= 4
                elif ch == 'q': self.castling |= 8

        self.ep_sq = None
        if len(parts) >= 4 and parts[3] != '-':
            self.ep_sq = uci_to_sq(parts[3])

        self.halfmove_clock = int(parts[4]) if len(parts) >= 5 else 0
        self.fullmove_number = int(parts[5]) if len(parts) >= 6 else 1

        self.zobrist = self.compute_zobrist()
        self.position_history.append(self.zobrist)

    def compute_zobrist(self) -> int:
        h = 0
        for sq in range(128):
            if not (sq & 0x88):
                p = self.board[sq]
                if p != EMPTY:
                    h ^= ZOBRIST_PIECES.get((sq, p), 0)

        if self.turn == BLACK:
            h ^= ZOBRIST_SIDE

        h ^= ZOBRIST_CASTLING[self.castling]

        if self.ep_sq is not None:
            ep_file = self.ep_sq & 7
            h ^= ZOBRIST_EP_FILE[ep_file]

        return h

    def is_attacked(self, sq: int, by_color: int) -> bool:
        # Pawn attacks
        pawn_dir = 16 if by_color == WHITE else -16
        for df in (-1, 1):
            p_sq = sq - pawn_dir + df
            if not (p_sq & 0x88):
                if self.board[p_sq] == by_color * PAWN:
                    return True

        # Knight attacks
        k_piece = by_color * KNIGHT
        for delta in KNIGHT_DELTAS:
            n_sq = sq + delta
            if not (n_sq & 0x88) and self.board[n_sq] == k_piece:
                return True

        # King attacks
        king_piece = by_color * KING
        for delta in KING_DELTAS:
            k_sq = sq + delta
            if not (k_sq & 0x88) and self.board[k_sq] == king_piece:
                return True

        # Ray attacks: Bishop / Queen (diagonals)
        b_piece = by_color * BISHOP
        q_piece = by_color * QUEEN
        for delta in BISHOP_DELTAS:
            ray_sq = sq + delta
            while not (ray_sq & 0x88):
                p = self.board[ray_sq]
                if p != EMPTY:
                    if p == b_piece or p == q_piece:
                        return True
                    break
                ray_sq += delta

        # Ray attacks: Rook / Queen (orthogonals)
        r_piece = by_color * ROOK
        for delta in ROOK_DELTAS:
            ray_sq = sq + delta
            while not (ray_sq & 0x88):
                p = self.board[ray_sq]
                if p != EMPTY:
                    if p == r_piece or p == q_piece:
                        return True
                    break
                ray_sq += delta

        return False

    def in_check(self, color: int) -> bool:
        return self.is_attacked(self.king_sq[color], -color)

    def generate_legal_moves(self, captures_only: bool = False) -> List[Move]:
        pseudo_moves = self.generate_pseudo_moves(captures_only=captures_only)
        legal = []
        for m in pseudo_moves:
            self.make_move(m)
            if not self.in_check(-self.turn):
                legal.append(m)
            self.unmake_move()
        return legal

    def generate_pseudo_moves(self, captures_only: bool = False) -> List[Move]:
        moves = []
        turn = self.turn
        opp = -turn

        for sq in range(128):
            if sq & 0x88:
                continue
            piece = self.board[sq]
            if piece == EMPTY or (piece > 0 and turn != WHITE) or (piece < 0 and turn != BLACK):
                continue

            ptype = abs(piece)

            if ptype == PAWN:
                forward = 16 if turn == WHITE else -16
                start_rank = 1 if turn == WHITE else 6
                promo_rank = 7 if turn == WHITE else 0
                to_sq = sq + forward

                if not captures_only:
                    if not (to_sq & 0x88) and self.board[to_sq] == EMPTY:
                        if (to_sq >> 4) == promo_rank:
                            for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                                moves.append(Move(sq, to_sq, promo=promo))
                        else:
                            moves.append(Move(sq, to_sq))
                            if (sq >> 4) == start_rank:
                                to_sq2 = to_sq + forward
                                if self.board[to_sq2] == EMPTY:
                                    moves.append(Move(sq, to_sq2))

                # Diagonal captures
                for df in (-1, 1):
                    cap_sq = sq + forward + df
                    if not (cap_sq & 0x88):
                        target = self.board[cap_sq]
                        if target != EMPTY and ((turn == WHITE and target < 0) or (turn == BLACK and target > 0)):
                            if (cap_sq >> 4) == promo_rank:
                                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                                    moves.append(Move(sq, cap_sq, promo=promo, captured=target))
                            else:
                                moves.append(Move(sq, cap_sq, captured=target))
                        elif cap_sq == self.ep_sq:
                            ep_captured = -turn * PAWN
                            moves.append(Move(sq, cap_sq, captured=ep_captured, flags=1))

            elif ptype == KNIGHT:
                for delta in KNIGHT_DELTAS:
                    to_sq = sq + delta
                    if not (to_sq & 0x88):
                        target = self.board[to_sq]
                        if target == EMPTY:
                            if not captures_only:
                                moves.append(Move(sq, to_sq))
                        elif (turn == WHITE and target < 0) or (turn == BLACK and target > 0):
                            moves.append(Move(sq, to_sq, captured=target))

            elif ptype == BISHOP:
                for delta in BISHOP_DELTAS:
                    to_sq = sq + delta
                    while not (to_sq & 0x88):
                        target = self.board[to_sq]
                        if target == EMPTY:
                            if not captures_only:
                                moves.append(Move(sq, to_sq))
                        else:
                            if (turn == WHITE and target < 0) or (turn == BLACK and target > 0):
                                moves.append(Move(sq, to_sq, captured=target))
                            break
                        to_sq += delta

            elif ptype == ROOK:
                for delta in ROOK_DELTAS:
                    to_sq = sq + delta
                    while not (to_sq & 0x88):
                        target = self.board[to_sq]
                        if target == EMPTY:
                            if not captures_only:
                                moves.append(Move(sq, to_sq))
                        else:
                            if (turn == WHITE and target < 0) or (turn == BLACK and target > 0):
                                moves.append(Move(sq, to_sq, captured=target))
                            break
                        to_sq += delta

            elif ptype == QUEEN:
                for delta in QUEEN_DELTAS:
                    to_sq = sq + delta
                    while not (to_sq & 0x88):
                        target = self.board[to_sq]
                        if target == EMPTY:
                            if not captures_only:
                                moves.append(Move(sq, to_sq))
                        else:
                            if (turn == WHITE and target < 0) or (turn == BLACK and target > 0):
                                moves.append(Move(sq, to_sq, captured=target))
                            break
                        to_sq += delta

            elif ptype == KING:
                for delta in KING_DELTAS:
                    to_sq = sq + delta
                    if not (to_sq & 0x88):
                        target = self.board[to_sq]
                        if target == EMPTY:
                            if not captures_only:
                                moves.append(Move(sq, to_sq))
                        elif (turn == WHITE and target < 0) or (turn == BLACK and target > 0):
                            moves.append(Move(sq, to_sq, captured=target))

                if not captures_only:
                    if turn == WHITE:
                        if sq == 0x04 and not self.in_check(WHITE):
                            if (self.castling & 1) and self.board[0x05] == EMPTY and self.board[0x06] == EMPTY:
                                if not self.is_attacked(0x05, BLACK) and not self.is_attacked(0x06, BLACK):
                                    moves.append(Move(0x04, 0x06, flags=2))
                            if (self.castling & 2) and self.board[0x03] == EMPTY and self.board[0x02] == EMPTY and self.board[0x01] == EMPTY:
                                if not self.is_attacked(0x03, BLACK) and not self.is_attacked(0x02, BLACK):
                                    moves.append(Move(0x04, 0x02, flags=2))
                    else:
                        if sq == 0x74 and not self.in_check(BLACK):
                            if (self.castling & 4) and self.board[0x75] == EMPTY and self.board[0x76] == EMPTY:
                                if not self.is_attacked(0x75, WHITE) and not self.is_attacked(0x76, WHITE):
                                    moves.append(Move(0x74, 0x76, flags=2))
                            if (self.castling & 8) and self.board[0x73] == EMPTY and self.board[0x72] == EMPTY and self.board[0x71] == EMPTY:
                                if not self.is_attacked(0x73, WHITE) and not self.is_attacked(0x72, WHITE):
                                    moves.append(Move(0x74, 0x72, flags=2))

        return moves

    def make_move(self, move: Move) -> None:
        from_sq = move.from_sq
        to_sq = move.to_sq
        piece = self.board[from_sq]
        ptype = abs(piece)

        # Save previous state
        prev_state = (
            move,
            self.castling,
            self.ep_sq,
            self.halfmove_clock,
            self.king_sq[self.turn],
            self.zobrist
        )
        self.history.append(prev_state)

        # Incremental hash update: remove moving piece, previous castling, previous ep
        h = self.zobrist ^ ZOBRIST_PIECES.get((from_sq, piece), 0)
        h ^= ZOBRIST_CASTLING[self.castling]
        if self.ep_sq is not None:
            h ^= ZOBRIST_EP_FILE[self.ep_sq & 7]

        # Handle captured piece hash update
        if move.captured != EMPTY and move.flags != 1:
            h ^= ZOBRIST_PIECES.get((to_sq, move.captured), 0)

        # Halfmove clock
        if ptype == PAWN or move.captured != EMPTY:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if self.turn == BLACK:
            self.fullmove_number += 1

        # En passant capture
        if move.flags == 1:
            ep_pawn_sq = to_sq - (16 if self.turn == WHITE else -16)
            ep_pawn = -self.turn * PAWN
            self.board[ep_pawn_sq] = EMPTY
            h ^= ZOBRIST_PIECES.get((ep_pawn_sq, ep_pawn), 0)

        # Move piece on board
        self.board[from_sq] = EMPTY
        new_piece = self.turn * move.promo if move.promo else piece
        self.board[to_sq] = new_piece
        h ^= ZOBRIST_PIECES.get((to_sq, new_piece), 0)

        # Castling rook movement & hash
        if move.flags == 2:
            if to_sq == 0x06:
                self.board[0x07] = EMPTY
                self.board[0x05] = WHITE * ROOK
                h ^= ZOBRIST_PIECES.get((0x07, WHITE * ROOK), 0) ^ ZOBRIST_PIECES.get((0x05, WHITE * ROOK), 0)
            elif to_sq == 0x02:
                self.board[0x00] = EMPTY
                self.board[0x03] = WHITE * ROOK
                h ^= ZOBRIST_PIECES.get((0x00, WHITE * ROOK), 0) ^ ZOBRIST_PIECES.get((0x03, WHITE * ROOK), 0)
            elif to_sq == 0x76:
                self.board[0x77] = EMPTY
                self.board[0x75] = BLACK * ROOK
                h ^= ZOBRIST_PIECES.get((0x77, BLACK * ROOK), 0) ^ ZOBRIST_PIECES.get((0x75, BLACK * ROOK), 0)
            elif to_sq == 0x72:
                self.board[0x70] = EMPTY
                self.board[0x73] = BLACK * ROOK
                h ^= ZOBRIST_PIECES.get((0x70, BLACK * ROOK), 0) ^ ZOBRIST_PIECES.get((0x73, BLACK * ROOK), 0)

        # King square update
        if ptype == KING:
            self.king_sq[self.turn] = to_sq

        # Update castling rights
        self.castling &= CASTLE_MASK[from_sq]
        self.castling &= CASTLE_MASK[to_sq]
        h ^= ZOBRIST_CASTLING[self.castling]

        # Update ep square
        if ptype == PAWN and abs(to_sq - from_sq) == 32:
            self.ep_sq = (from_sq + to_sq) // 2
            h ^= ZOBRIST_EP_FILE[self.ep_sq & 7]
        else:
            self.ep_sq = None

        # Side to move hash toggle
        h ^= ZOBRIST_SIDE
        self.turn = -self.turn
        self.zobrist = h
        self.position_history.append(self.zobrist)

    def unmake_move(self) -> None:
        self.position_history.pop()
        prev_state = self.history.pop()
        move, prev_castling, prev_ep, prev_halfmove, prev_king_sq, prev_zobrist = prev_state

        from_sq = move.from_sq
        to_sq = move.to_sq
        flags = move.flags

        self.turn = -self.turn
        if self.turn == BLACK:
            self.fullmove_number -= 1

        piece = self.board[to_sq]
        ptype = abs(piece) if not move.promo else PAWN

        self.board[from_sq] = self.turn * ptype
        self.board[to_sq] = move.captured if flags != 1 else EMPTY

        if flags == 1:
            ep_pawn_sq = to_sq - (16 if self.turn == WHITE else -16)
            self.board[ep_pawn_sq] = -self.turn * PAWN

        if flags == 2:
            if to_sq == 0x06:
                self.board[0x05] = EMPTY
                self.board[0x07] = WHITE * ROOK
            elif to_sq == 0x02:
                self.board[0x03] = EMPTY
                self.board[0x00] = WHITE * ROOK
            elif to_sq == 0x76:
                self.board[0x75] = EMPTY
                self.board[0x77] = BLACK * ROOK
            elif to_sq == 0x72:
                self.board[0x73] = EMPTY
                self.board[0x70] = BLACK * ROOK

        self.king_sq[self.turn] = prev_king_sq
        self.castling = prev_castling
        self.ep_sq = prev_ep
        self.halfmove_clock = prev_halfmove
        self.zobrist = prev_zobrist

    def is_repetition(self) -> bool:
        # Check if the current position hash has occurred >= 3 times in the irreversible window
        curr_hash = self.zobrist
        count = 0
        limit = min(len(self.position_history), self.halfmove_clock + 1)
        for i in range(1, limit + 1):
            if self.position_history[-i] == curr_hash:
                count += 1
                if count >= 3:
                    return True
        return False

    def is_insufficient_material(self) -> bool:
        # Fast check for K vs K, K+B vs K, K+N vs K, K+B vs K+B (same color)
        white_pieces = []
        black_pieces = []
        for sq in range(128):
            if not (sq & 0x88):
                p = self.board[sq]
                if p != EMPTY and abs(p) != KING:
                    if p > 0:
                        white_pieces.append((p, sq))
                    else:
                        black_pieces.append((p, sq))
                    if len(white_pieces) > 1 or len(black_pieces) > 1:
                        return False

        # Only kings
        if not white_pieces and not black_pieces:
            return True

        # K+N vs K or K+B vs K
        if len(white_pieces) == 1 and not black_pieces:
            return abs(white_pieces[0][0]) in (KNIGHT, BISHOP)
        if len(black_pieces) == 1 and not white_pieces:
            return abs(black_pieces[0][0]) in (KNIGHT, BISHOP)

        # K+B vs K+B same color bishop
        if len(white_pieces) == 1 and len(black_pieces) == 1:
            p1, sq1 = white_pieces[0]
            p2, sq2 = black_pieces[0]
            if abs(p1) == BISHOP and abs(p2) == BISHOP:
                color1 = ((sq1 >> 4) + (sq1 & 7)) % 2
                color2 = ((sq2 >> 4) + (sq2 & 7)) % 2
                return color1 == color2

        return False

    def is_fifty_moves(self) -> bool:
        return self.halfmove_clock >= 100

    def parse_uci(self, uci_str: str) -> Optional[Move]:
        clean_uci = uci_str.strip().lower()
        for m in self.generate_legal_moves():
            if m.to_uci().lower() == clean_uci:
                return m
        return None


def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    moves = board.generate_legal_moves()
    if depth == 1:
        return len(moves)
    nodes = 0
    for m in moves:
        board.make_move(m)
        nodes += perft(board, depth - 1)
        board.unmake_move()
    return nodes


# =============================================================================
# EVALUATION FUNCTION (TAPERED MG/EG WITH HEURISTICS)
# =============================================================================

def evaluate(board: Board) -> int:
    """
    Tapered evaluation returning centipawn score from the perspective of board.turn.
    Positive score means current player to move is ahead.
    """
    mg_score = 0
    eg_score = 0
    phase = 0

    white_bishops = 0
    black_bishops = 0
    white_pawn_files = [0] * 8
    black_pawn_files = [0] * 8

    # Piece count & placement
    for sq in range(128):
        if sq & 0x88:
            continue
        piece = board.board[sq]
        if piece == EMPTY:
            continue

        ptype = abs(piece)
        phase += GAME_PHASE_WEIGHTS[ptype]

        if piece > 0:  # White
            mg_score += PIECE_VALUES_MG[ptype] + PST_MG[ptype][sq]
            eg_score += PIECE_VALUES_EG[ptype] + PST_EG[ptype][sq]
            if ptype == BISHOP:
                white_bishops += 1
            elif ptype == PAWN:
                white_pawn_files[sq & 7] += 1
            elif ptype == ROOK:
                # Open/semi-open file bonus
                f = sq & 7
                if white_pawn_files[f] == 0:
                    mg_score += 15 if black_pawn_files[f] > 0 else 25
                    eg_score += 15 if black_pawn_files[f] > 0 else 25
        else:  # Black
            # Symmetrically mirror square for black PST lookup
            flipped_sq = ((7 - (sq >> 4)) << 4) | (sq & 7)
            mg_score -= (PIECE_VALUES_MG[ptype] + PST_MG[ptype][flipped_sq])
            eg_score -= (PIECE_VALUES_EG[ptype] + PST_EG[ptype][flipped_sq])
            if ptype == BISHOP:
                black_bishops += 1
            elif ptype == PAWN:
                black_pawn_files[sq & 7] += 1
            elif ptype == ROOK:
                f = sq & 7
                if black_pawn_files[f] == 0:
                    mg_score -= 15 if white_pawn_files[f] > 0 else 25
                    eg_score -= 15 if white_pawn_files[f] > 0 else 25

    # Bishop pair bonus
    if white_bishops >= 2:
        mg_score += 35
        eg_score += 45
    if black_bishops >= 2:
        mg_score -= 35
        eg_score -= 45

    # Pawn structure penalties
    for f in range(8):
        # Doubled pawns
        if white_pawn_files[f] > 1:
            mg_score -= 15 * (white_pawn_files[f] - 1)
            eg_score -= 20 * (white_pawn_files[f] - 1)
        if black_pawn_files[f] > 1:
            mg_score += 15 * (black_pawn_files[f] - 1)
            eg_score += 20 * (black_pawn_files[f] - 1)

        # Isolated pawns
        left_has_w = white_pawn_files[f - 1] if f > 0 else 0
        right_has_w = white_pawn_files[f + 1] if f < 7 else 0
        if white_pawn_files[f] > 0 and left_has_w == 0 and right_has_w == 0:
            mg_score -= 15
            eg_score -= 20

        left_has_b = black_pawn_files[f - 1] if f > 0 else 0
        right_has_b = black_pawn_files[f + 1] if f < 7 else 0
        if black_pawn_files[f] > 0 and left_has_b == 0 and right_has_b == 0:
            mg_score += 15
            eg_score += 20

    # Tapered phase interpolation
    if phase > MAX_PHASE:
        phase = MAX_PHASE
    eval_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE

    # Return perspective of side to move
    return eval_score if board.turn == WHITE else -eval_score


# =============================================================================
# TRANSPOSITION TABLE
# =============================================================================

class TTEntry:
    __slots__ = ('key', 'depth', 'score', 'flag', 'best_move')

    def __init__(self, key: int, depth: int, score: int, flag: int, best_move: Optional[Move]):
        self.key = key
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move


class TranspositionTable:
    def __init__(self, size_mb: int = 64):
        self.size = (size_mb * 1024 * 1024) // 64
        # Round down to nearest power of 2 for mask lookup
        power = 1
        while power * 2 <= self.size:
            power *= 2
        self.size = power
        self.mask = self.size - 1
        self.table: List[Optional[TTEntry]] = [None] * self.size

    def clear(self) -> None:
        self.table = [None] * self.size

    def store(self, key: int, depth: int, score: int, flag: int, best_move: Optional[Move], ply: int) -> None:
        # Mate score distance adjustment
        if score > MATE_THRESHOLD:
            score += ply
        elif score < -MATE_THRESHOLD:
            score -= ply

        idx = key & self.mask
        existing = self.table[idx]
        if existing is None or depth >= existing.depth:
            self.table[idx] = TTEntry(key, depth, score, flag, best_move)

    def probe(self, key: int, ply: int) -> Optional[Tuple[int, int, int, Optional[Move]]]:
        idx = key & self.mask
        entry = self.table[idx]
        if entry is not None and entry.key == key:
            score = entry.score
            # Mate score restore
            if score > MATE_THRESHOLD:
                score -= ply
            elif score < -MATE_THRESHOLD:
                score += ply
            return entry.depth, score, entry.flag, entry.best_move
        return None


# =============================================================================
# MOVE ORDERING (MVV-LVA, KILLERS, HISTORY, TT MOVE)
# =============================================================================

# MVV-LVA Matrix: [victim][attacker]
MVV_LVA = [
    [0] * 7,
    # Victim Pawn (1)
    [0, 105, 104, 103, 102, 101, 100],
    # Victim Knight (2)
    [0, 205, 204, 203, 202, 201, 200],
    # Victim Bishop (3)
    [0, 305, 304, 303, 302, 301, 300],
    # Victim Rook (4)
    [0, 405, 404, 403, 402, 401, 400],
    # Victim Queen (5)
    [0, 505, 504, 503, 502, 501, 500],
    # Victim King (6)
    [0, 605, 604, 603, 602, 601, 600],
]


class MoveOrderer:
    def __init__(self):
        self.killers: List[List[Optional[Move]]] = [[None, None] for _ in range(128)]
        # History heuristic: [side 0/1][from_sq][to_sq]
        self.history: List[List[List[int]]] = [
            [[0] * 128 for _ in range(128)],
            [[0] * 128 for _ in range(128)]
        ]

    def clear(self) -> None:
        self.killers = [[None, None] for _ in range(128)]
        self.history = [
            [[0] * 128 for _ in range(128)],
            [[0] * 128 for _ in range(128)]
        ]

    def record_killer(self, ply: int, move: Move) -> None:
        if ply < 128 and self.killers[ply][0] != move:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move

    def record_history(self, turn: int, move: Move, depth: int) -> None:
        side_idx = 0 if turn == WHITE else 1
        self.history[side_idx][move.from_sq][move.to_sq] += depth * depth
        # Prevent integer overflow in long games
        if self.history[side_idx][move.from_sq][move.to_sq] > 2000000:
            self.history[side_idx][move.from_sq][move.to_sq] //= 2

    def score_move(self, board: Board, move: Move, tt_move: Optional[Move], ply: int) -> int:
        # 1. Transposition table move gets top priority
        if tt_move is not None and move == tt_move:
            return 10000000

        # 2. Winning & equal captures scored by MVV-LVA
        if move.captured != EMPTY:
            vic = abs(move.captured)
            att = abs(board.board[move.from_sq])
            return 1000000 + MVV_LVA[vic][att]

        # 3. Promotions
        if move.promo:
            return 900000 + move.promo * 1000

        # 4. Killer moves
        if ply < 128:
            if move == self.killers[ply][0]:
                return 800000
            elif move == self.killers[ply][1]:
                return 700000

        # 5. History heuristic for quiet moves
        side_idx = 0 if board.turn == WHITE else 1
        return self.history[side_idx][move.from_sq][move.to_sq]


# =============================================================================
# SEARCH ENGINE (PVS, ASPIRATION, QUIESCENCE, TIMING)
# =============================================================================

class Searcher:
    def __init__(self, tt_size_mb: int = 32):
        self.tt = TranspositionTable(size_mb=tt_size_mb)
        self.orderer = MoveOrderer()
        self.nodes = 0
        self.stop_search = False
        self.time_limit = 0.0
        self.start_time = 0.0
        self.check_nodes_mask = 255  # Check time every 256 nodes

    def reset(self) -> None:
        self.tt.clear()
        self.orderer.clear()

    def check_time(self) -> None:
        if (self.nodes & self.check_nodes_mask) == 0:
            if time.time() >= self.time_limit:
                self.stop_search = True

    def quiescence(self, board: Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self.check_time()
        if self.stop_search:
            return 0

        # Stand-pat evaluation
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Delta pruning
        BIG_DELTA = 950  # Queen value
        if stand_pat < alpha - BIG_DELTA:
            return alpha

        # Generate captures only
        captures = board.generate_legal_moves(captures_only=True)
        # Order captures by MVV-LVA
        captures.sort(key=lambda m: MVV_LVA[abs(m.captured)][abs(board.board[m.from_sq])] if m.captured else 0, reverse=True)

        for move in captures:
            board.make_move(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.unmake_move()

            if self.stop_search:
                return 0

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def negamax(self, board: Board, depth: int, alpha: int, beta: int, ply: int, is_pv: bool = True) -> int:
        self.nodes += 1
        self.check_time()
        if self.stop_search:
            return 0

        # Draw detection (threefold repetition or 50-move rule)
        if ply > 0 and (board.is_repetition() or board.is_fifty_moves() or board.is_insufficient_material()):
            return 0

        orig_alpha = alpha
        in_check = board.in_check(board.turn)

        # Check extension: if in check, search deeper to resolve tactics
        if in_check:
            depth += 1

        # Base case: quiescence search
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        # Transposition table probe
        tt_probe = self.tt.probe(board.zobrist, ply)
        tt_move = None
        if tt_probe is not None:
            tt_depth, tt_score, tt_flag, tt_move = tt_probe
            if not is_pv and tt_depth >= depth:
                if tt_flag == TT_EXACT:
                    return tt_score
                elif tt_flag == TT_LOWERBOUND and tt_score >= beta:
                    return tt_score
                elif tt_flag == TT_UPPERBOUND and tt_score <= alpha:
                    return tt_score

        # Null Move Pruning (NMP)
        if depth >= 3 and not in_check and not is_pv and ply > 0:
            # Check non-pawn material exists to avoid zugzwang
            has_major = False
            for sq in range(128):
                if not (sq & 0x88):
                    p = board.board[sq]
                    if p != EMPTY and (p > 0 if board.turn == WHITE else p < 0) and abs(p) in (KNIGHT, BISHOP, ROOK, QUEEN):
                        has_major = True
                        break

            if has_major:
                R = 2 if depth < 6 else 3
                # Make null move
                prev_ep = board.ep_sq
                board.ep_sq = None
                board.turn = -board.turn
                board.zobrist ^= ZOBRIST_SIDE
                if prev_ep is not None:
                    board.zobrist ^= ZOBRIST_EP_FILE[prev_ep & 7]

                null_score = -self.negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, is_pv=False)

                # Unmake null move
                board.turn = -board.turn
                board.ep_sq = prev_ep
                board.zobrist ^= ZOBRIST_SIDE
                if prev_ep is not None:
                    board.zobrist ^= ZOBRIST_EP_FILE[prev_ep & 7]

                if self.stop_search:
                    return 0
                if null_score >= beta:
                    return beta

        # Generate legal moves
        moves = board.generate_legal_moves()
        if not moves:
            if in_check:
                # Checkmate: favor shorter mate paths
                return -MATE_SCORE + ply
            else:
                # Stalemate
                return 0

        # Move ordering
        scored_moves = []
        for m in moves:
            s = self.orderer.score_move(board, m, tt_move, ply)
            scored_moves.append((s, m))
        scored_moves.sort(key=lambda x: x[0], reverse=True)

        best_move = None
        best_score = -INFINITY
        moves_searched = 0

        for _, move in scored_moves:
            board.make_move(move)
            moves_searched += 1

            # Principal Variation Search (PVS) with Late Move Reductions (LMR)
            if moves_searched == 1:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, is_pv=True)
            else:
                # Late Move Reduction condition
                needs_full_search = True
                if (depth >= 3 and moves_searched >= 4 and not in_check and 
                    move.captured == EMPTY and not move.promo and not board.in_check(board.turn)):
                    reduction = 1
                    if moves_searched >= 8:
                        reduction = 2
                    score = -self.negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, is_pv=False)
                    needs_full_search = (score > alpha)

                if needs_full_search:
                    # Null-window search
                    score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, is_pv=False)
                    if alpha < score < beta:
                        # Re-search with full window
                        score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, is_pv=True)

            board.unmake_move()

            if self.stop_search:
                return 0

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
                if alpha >= beta:
                    # Beta cutoff
                    if move.captured == EMPTY:
                        self.orderer.record_killer(ply, move)
                        self.orderer.record_history(board.turn, move, depth)
                    break

        # Transposition Table store
        if not self.stop_search:
            tt_flag = TT_EXACT
            if best_score <= orig_alpha:
                tt_flag = TT_UPPERBOUND
            elif best_score >= beta:
                tt_flag = TT_LOWERBOUND
            self.tt.store(board.zobrist, depth, best_score, tt_flag, best_move, ply)

        return best_score

    def search(self, board: Board, max_time_seconds: float, max_depth: int = 64) -> Tuple[Optional[Move], int, int]:
        self.nodes = 0
        self.stop_search = False
        self.start_time = time.time()
        # Reserve a safe safety margin (e.g. 15%) so we never timeout
        effective_time = max(0.05, max_time_seconds * 0.85)
        self.time_limit = self.start_time + effective_time

        legal_moves = board.generate_legal_moves()
        if not legal_moves:
            return None, 0, 0
        if len(legal_moves) == 1:
            return legal_moves[0], 0, 1

        best_move = legal_moves[0]
        best_score = 0
        completed_depth = 0

        # Iterative Deepening
        for depth in range(1, max_depth + 1):
            if depth >= 4:
                # Aspiration window
                alpha = best_score - 40
                beta = best_score + 40
                score = self.negamax(board, depth, alpha, beta, ply=0, is_pv=True)

                if score <= alpha or score >= beta:
                    # Fail low / fail high re-search with full window
                    alpha = -INFINITY
                    beta = INFINITY
                    score = self.negamax(board, depth, alpha, beta, ply=0, is_pv=True)
            else:
                score = self.negamax(board, depth, -INFINITY, INFINITY, ply=0, is_pv=True)

            if self.stop_search:
                break

            # Probe TT for principal variation best move
            tt_entry = self.tt.probe(board.zobrist, ply=0)
            if tt_entry is not None and tt_entry[3] is not None:
                best_move = tt_entry[3]

            best_score = score
            completed_depth = depth

            # Print UCI search progress (info depth ... score cp ... nodes ...)
            elapsed_ms = max(1, int((time.time() - self.start_time) * 1000))
            nps = int((self.nodes * 1000) / elapsed_ms)
            sys.stdout.write(f"info depth {depth} score cp {best_score} nodes {self.nodes} nps {nps} time {elapsed_ms} pv {best_move.to_uci()}\n")
            sys.stdout.flush()

            # Early exit if checkmate found
            if abs(best_score) > MATE_THRESHOLD:
                break

            # Stop if time has reached 50% of budget on completed depth (next iteration will take ~3-4x)
            if time.time() - self.start_time > effective_time * 0.5:
                break

        return best_move, best_score, completed_depth


# =============================================================================
# UCI PROTOCOL HANDLER
# =============================================================================

class UCIEngine:
    def __init__(self):
        self.board = Board()
        self.searcher = Searcher(tt_size_mb=32)

    def run(self) -> None:
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                cmd = parts[0].lower()

                if cmd == "uci":
                    sys.stdout.write("id name NAVY\n")
                    sys.stdout.write("id author DeepMind_Antigravity\n")
                    sys.stdout.write("uciok\n")
                    sys.stdout.flush()

                elif cmd == "isready":
                    sys.stdout.write("readyok\n")
                    sys.stdout.flush()

                elif cmd == "ucinewgame":
                    self.board = Board()
                    self.searcher.reset()

                elif cmd == "position":
                    self.handle_position(parts[1:])

                elif cmd == "go":
                    self.handle_go(parts[1:])

                elif cmd == "stop":
                    self.searcher.stop_search = True

                elif cmd == "quit":
                    break

            except (KeyboardInterrupt, SystemExit):
                break
            except Exception as e:
                # Send error details strictly to stderr; keep stdout clean for UCI
                sys.stderr.write(f"[ERROR] {e}\n")
                sys.stderr.flush()

    def handle_position(self, tokens: List[str]) -> None:
        if not tokens:
            return

        moves_idx = -1
        if "moves" in tokens:
            moves_idx = tokens.index("moves")

        # Parse board placement
        if tokens[0].lower() == "startpos":
            self.board = Board()
            fen_moves_start = 1
        elif tokens[0].lower() == "fen":
            fen_end = moves_idx if moves_idx != -1 else len(tokens)
            fen_str = " ".join(tokens[1:fen_end])
            self.board = Board(fen_str)

        # Apply subsequent moves
        if moves_idx != -1:
            for move_str in tokens[moves_idx + 1:]:
                m = self.board.parse_uci(move_str)
                if m is not None:
                    self.board.make_move(m)
                else:
                    sys.stderr.write(f"[WARN] Illegal or unparsed move in position: {move_str}\n")
                    sys.stderr.flush()

    def handle_go(self, tokens: List[str]) -> None:
        # Default fixed move time: 1.0s (fits 1.0s to 3.0s competition budget)
        allocated_time = 1.0
        max_depth = 64
        wtime_ms = None
        btime_ms = None
        winc_ms = 0.0
        binc_ms = 0.0
        movestogo = 35

        i = 0
        while i < len(tokens):
            t = tokens[i].lower()
            if t == "movetime" and i + 1 < len(tokens):
                allocated_time = max(0.05, float(tokens[i + 1]) / 1000.0)
                i += 1
            elif t == "wtime" and i + 1 < len(tokens):
                wtime_ms = float(tokens[i + 1])
                i += 1
            elif t == "btime" and i + 1 < len(tokens):
                btime_ms = float(tokens[i + 1])
                i += 1
            elif t == "winc" and i + 1 < len(tokens):
                winc_ms = float(tokens[i + 1])
                i += 1
            elif t == "binc" and i + 1 < len(tokens):
                binc_ms = float(tokens[i + 1])
                i += 1
            elif t == "movestogo" and i + 1 < len(tokens):
                movestogo = max(1, int(tokens[i + 1]))
                i += 1
            elif t == "depth" and i + 1 < len(tokens):
                max_depth = int(tokens[i + 1])
                i += 1
            i += 1

        tokens_lower = [x.lower() for x in tokens]
        if "movetime" not in tokens_lower:
            if self.board.turn == WHITE and wtime_ms is not None:
                allocated_time = max(0.05, (wtime_ms / movestogo + winc_ms * 0.75) / 1000.0)
            elif self.board.turn == BLACK and btime_ms is not None:
                allocated_time = max(0.05, (btime_ms / movestogo + binc_ms * 0.75) / 1000.0)

        best_move, score, depth = self.searcher.search(self.board, max_time_seconds=allocated_time, max_depth=max_depth)

        # Fail-safe legality guard
        if best_move is None or best_move not in self.board.generate_legal_moves():
            legals = self.board.generate_legal_moves()
            if legals:
                best_move = legals[0]

        if best_move is not None:
            sys.stdout.write(f"bestmove {best_move.to_uci()}\n")
        else:
            sys.stdout.write("bestmove (none)\n")
        sys.stdout.flush()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running quick test suite...")
        b = Board()
        print("Board initialized. Legal moves from startpos:", len(b.generate_legal_moves()))
        searcher = Searcher(tt_size_mb=16)
        m, s, d = searcher.search(b, max_time_seconds=0.5)
        print(f"Startpos Search Result: {m} (score: {s}, depth: {d}, nodes: {searcher.nodes})")
        return

    # Default to standard UCI protocol loop
    engine = UCIEngine()
    engine.run()


if __name__ == "__main__":
    main()
