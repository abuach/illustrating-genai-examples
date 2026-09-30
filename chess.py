"""Chess helpers for the Introduction chapter.

Pull a move out of a model's free-text reply, judge it against the rules with
python-chess (rules only, no engine, so the verdicts are reproducible anywhere the
library is installed), and draw the board annotating the moves different players
pick. ``engine_move`` adds the move a real UCI engine (Stockfish) actually plays,
so the demo can contrast a big LLM, a small LLM, and genuine search on one
position. The engine row needs a ``stockfish`` binary on PATH (``brew install
stockfish``); the move-parsing and rule verdicts do not.
"""

import os
import re
import shutil

import chess
import chess.engine
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import matplotlib.patheffects as pe

from genai.viz import GREEN, ORANGE, RED, DGRAY, MGRAY, _save

_SAN = re.compile(r"\b(O-O(?:-O)?|[KQRBN]?[a-h]?x?[a-h][1-8](?:=[QRBN])?[+#]?)")
_PTYPE = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK,
          "B": chess.BISHOP, "N": chess.KNIGHT}
_GLYPH = {"P": "♙", "N": "♘", "B": "♗", "R": "♖",
          "Q": "♕", "K": "♔", "p": "♟", "n": "♞",
          "b": "♝", "r": "♜", "q": "♛", "k": "♚"}
_WHITE_GLYPHS = "♔♕♖♗♘♙"
_LIGHT, _DARK = "#F1E7D2", "#9FB1BD"
_COLOR = {"illegal": RED, "legal": ORANGE, "legal, gives check": GREEN}


def first_move(text):
    """Return the first algebraic move found in a model's free-text reply."""
    m = _SAN.search(text or "")
    return m.group(1) if m else None


def verdict(fen, san):
    """Judge a move by the rules: 'illegal', 'legal', or 'legal, gives check'."""
    board = chess.Board(fen)
    try:
        move = board.parse_san(san)
    except ValueError:
        return "illegal"
    return "legal, gives check" if board.gives_check(move) else "legal"


def _engine_path():
    """Locate a UCI engine binary, or raise with an install hint."""
    found = shutil.which("stockfish")
    if found:
        return found
    for p in ("/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish",
              "/usr/bin/stockfish"):
        if os.path.exists(p):
            return p
    raise RuntimeError(
        "No UCI chess engine found. Install one, e.g. `brew install stockfish`.")


def engine_move(fen, depth=18):
    """Return the move a real engine actually plays here, in algebraic notation.

    Unlike the models, the engine isn't pattern-matching chess-flavored text: it
    runs a genuine UCI search (Stockfish) to a fixed depth, so its move is the same
    on every machine that has the binary. Needs ``stockfish`` on PATH.
    """
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(_engine_path()) as eng:
        result = eng.play(board, chess.engine.Limit(depth=depth))
    return board.san(result.move)


def _dest(san):
    """Destination square of a move, parsed from the text even when it's illegal."""
    squares = re.findall(r"[a-h][1-8]", san or "")
    return chess.parse_square(squares[-1]) if squares else None


def _src(board, san, dest):
    """Source square: the real one if legal, else the nearest matching piece."""
    try:
        return board.parse_san(san).from_square
    except ValueError:
        cands = list(board.pieces(_PTYPE.get((san or " ")[0], chess.PAWN), board.turn))
        return min(cands, key=lambda sq: chess.square_distance(sq, dest)) if cands else None


def _piece(ax, sq, glyph):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    white = glyph in _WHITE_GLYPHS
    ax.text(f + 0.5, r + 0.5, glyph, fontsize=26, ha="center", va="center", zorder=3,
            color="white" if white else DGRAY,
            path_effects=[pe.withStroke(linewidth=2.2, foreground=DGRAY)] if white else [])


def show_chess(fen, plays, path=None):
    """Print each player's move + rule verdict and draw the annotated board.

    plays: list of (label, san) tuples. Arrows are colored by verdict
    (red = illegal, amber = legal, green = legal and gives check).
    """
    board = chess.Board(fen)
    for label, san in plays:
        print(f"{label:14}{san or '?':6}->  {verdict(fen, san)}")

    fig, ax = plt.subplots(figsize=(5.2, 5.7))
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ax.add_patch(mp.Rectangle((f, r), 1, 1, zorder=0,
                                  facecolor=_LIGHT if (f + r) % 2 else _DARK))
        p = board.piece_at(sq)
        if p:
            _piece(ax, sq, _GLYPH[p.symbol()])

    for label, san in plays:
        dest, col = _dest(san), _COLOR[verdict(fen, san)]
        src = _src(board, san, dest)
        if src is None or dest is None:
            continue
        x0, y0 = chess.square_file(src) + 0.5, chess.square_rank(src) + 0.5
        x1, y1 = chess.square_file(dest) + 0.5, chess.square_rank(dest) + 0.5
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=3.2,
                                    mutation_scale=22, shrinkA=10, shrinkB=14))
        ax.scatter([x1], [y1], s=95, color=col, zorder=5, alpha=0.9,
                   edgecolors="white", linewidths=1.4)

    for i in range(8):
        ax.text(i + 0.5, -0.34, "abcdefgh"[i], ha="center", color=MGRAY, fontsize=10)
        ax.text(-0.34, i + 0.5, str(i + 1), va="center", color=MGRAY, fontsize=10)
    handles = [mp.Patch(color=RED, label="illegal"),
               mp.Patch(color=ORANGE, label="legal"),
               mp.Patch(color=GREEN, label="legal + gives check")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False, fontsize=9, handlelength=1.0, columnspacing=1.3)
    ax.set_xlim(-0.7, 8); ax.set_ylim(-0.8, 8)
    ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    _save(fig, path)
    plt.show()
