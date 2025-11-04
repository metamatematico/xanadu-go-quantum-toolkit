"""SGF parsing and Go board reconstruction utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_COLOR_MAP = {"B": -1, "W": 1}
_COORD_RE = re.compile(r";([BW])\[([^\]]*)\]")
_METADATA_RE = re.compile(r"([A-Z]{1,2})\[([^\]]*)\]")


@dataclass
class SGFGame:
    size: int
    moves: List[Tuple[str, Optional[Tuple[int, int]]]]
    metadata: Dict[str, str]

    def board_sequence(self) -> List[List[List[int]]]:
        return list(_board_sequence(self.size, self.moves))


def load_sgf(path: str | Path) -> SGFGame:
    """Parse an SGF file and return moves plus metadata."""

    text = Path(path).read_text(encoding="utf-8").strip()
    metadata = _parse_metadata(text)
    size = int(metadata.get("SZ", 19))
    moves = _parse_moves(text, size=size)
    return SGFGame(size=size, moves=moves, metadata=metadata)


def _parse_metadata(text: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}

    first_semicolon = text.find(";")
    if first_semicolon == -1:
        return metadata

    second_semicolon = text.find(";", first_semicolon + 1)
    if second_semicolon == -1:
        second_semicolon = len(text)

    header = text[first_semicolon + 1 : second_semicolon]
    for key, value in _METADATA_RE.findall(header):
        metadata[key] = value
    return metadata


def _parse_moves(text: str, size: int) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
    moves: List[Tuple[str, Optional[Tuple[int, int]]]] = []
    for color, coord in _COORD_RE.findall(text):
        move = (color, _coord_to_indices(coord.strip(), size))
        moves.append(move)
    return moves


def _coord_to_indices(coord: str, size: int) -> Optional[Tuple[int, int]]:
    if coord == "":
        return None  # pass move
    if len(coord) != 2:
        raise ValueError(f"Unexpected coordinate token: '{coord}'")
    col = ord(coord[0]) - ord("a")
    row = ord(coord[1]) - ord("a")
    if not (0 <= row < size and 0 <= col < size):
        raise ValueError(f"Coordinate out of bounds: '{coord}' for size {size}")
    return row, col


def _board_sequence(
    size: int, moves: Iterable[Tuple[str, Optional[Tuple[int, int]]]]
) -> Iterable[List[List[int]]]:
    board = [[0 for _ in range(size)] for _ in range(size)]
    yield _copy_board(board)
    for color, coord in moves:
        if coord is None:
            yield _copy_board(board)
            continue
        place_stone(board, _COLOR_MAP[color], coord)
        yield _copy_board(board)


def place_stone(board: List[List[int]], color: int, coord: Tuple[int, int]) -> None:
    r, c = coord
    if board[r][c] != 0:
        raise ValueError(f"Illegal move on occupied point {coord}")
    board[r][c] = color
    opponent = -color
    size = len(board)

    for nr, nc in _neighbors(r, c, size):
        if board[nr][nc] == opponent:
            group = _collect_group(board, nr, nc)
            if not _has_liberty(board, group):
                _remove_group(board, group)

    group = _collect_group(board, r, c)
    if not _has_liberty(board, group):
        _remove_group(board, group)


def _neighbors(r: int, c: int, size: int) -> Iterable[Tuple[int, int]]:
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc


def _collect_group(board: List[List[int]], r: int, c: int) -> List[Tuple[int, int]]:
    color = board[r][c]
    stack = [(r, c)]
    visited = {(r, c)}
    group = []
    while stack:
        cr, cc = stack.pop()
        group.append((cr, cc))
        for nr, nc in _neighbors(cr, cc, len(board)):
            if board[nr][nc] == color and (nr, nc) not in visited:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return group


def _has_liberty(board: List[List[int]], group: List[Tuple[int, int]]) -> bool:
    for r, c in group:
        for nr, nc in _neighbors(r, c, len(board)):
            if board[nr][nc] == 0:
                return True
    return False


def _remove_group(board: List[List[int]], group: List[Tuple[int, int]]) -> None:
    for r, c in group:
        board[r][c] = 0


def _copy_board(board: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in board]
