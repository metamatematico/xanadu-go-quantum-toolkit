"""Analyze a Go SGF game move by move using xanadu_go modules."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from xanadu_go.quantum_graph import QuantumGoGraph
from xanadu_go.qaoa import GoGraphAdvancedXanadu
from xanadu_go.pipelines import pipeline_cfg_to_gbs_features
from xanadu_go.sgf import SGFGame, load_sgf
from xanadu_go.visualization import plot_board_with_analysis

GO_COLUMNS = "ABCDEFGHJKLMNOPQRST"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Ruta al archivo SGF a analizar")
    parser.add_argument("--start", type=int, default=0, help="Índice inicial de tablero (0 = posición inicial)")
    parser.add_argument("--end", type=int, default=None, help="Índice final de tablero (inclusive)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Umbral para colorear expectativas cuánticas")
    parser.add_argument("--gbs-samples", type=int, default=0, help="Número de muestras GBS por posición (0 = desactivar)")
    parser.add_argument("--gbs-node-count", type=int, default=None, help="Tamaño máximo de subgrafo para GBS")
    parser.add_argument("--qaoa", action="store_true", help="Ejecutar QAOA Max-Cut sobre cada posición")
    parser.add_argument("--qaoa-graph", choices=["fgr", "cfg", "rsf"], default="fgr", help="Grafo objetivo para QAOA")
    parser.add_argument("--qaoa-max-nodes", type=int, default=12, help="Número máximo de nodos en QAOA")
    parser.add_argument("--qaoa-depth", type=int, default=2, help="Profundidad de QAOA")
    parser.add_argument("--qaoa-steps", type=int, default=60, help="Iteraciones del optimizador QAOA")
    parser.add_argument("--qaoa-stepsize", type=float, default=0.2, help="Tamaño de paso para QAOA")
    parser.add_argument("--vertex-penalty", type=float, default=2.0, help="Penalización para Vertex Cover (cuando se use)")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Directorio destino para guardar gráficos por jugada")
    parser.add_argument("--plot-prefix", type=str, default="move_", help="Prefijo de archivo para las capturas")
    parser.add_argument(
        "--plot-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Formato de imagen para las capturas",
    )
    parser.add_argument("--plot-dpi", type=int, default=200, help="Resolución (DPI) de las capturas")
    return parser.parse_args()


def coord_to_label(coord: Optional[Tuple[int, int]], size: int) -> str:
    if coord is None:
        return "pass"
    row, col = coord
    col_label = GO_COLUMNS[col]
    row_label = str(size - row)
    return f"{col_label}{row_label}"


def stone_counts(board: Sequence[Sequence[int]]) -> Dict[int, int]:
    counts = {-1: 0, 0: 0, 1: 0}
    for row in board:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return counts


def summarize_cfg(cfg_data: Dict[int, Dict[str, object]]) -> Dict[str, object]:
    total = len(cfg_data)
    whites = 0
    blacks = 0
    max_size = 0
    for data in cfg_data.values():
        color = data.get("color", 0)
        members = data.get("members", [])
        size = len(members)
        max_size = max(max_size, size)
        if color == 1:
            whites += 1
        elif color == -1:
            blacks += 1
    return {
        "clusters": total,
        "white_clusters": whites,
        "black_clusters": blacks,
        "largest_cluster": max_size,
    }


def run_qaoa(board: List[List[int]], args: argparse.Namespace) -> Dict[str, object]:
    solver = GoGraphAdvancedXanadu(
        board,
        qaoa_max_nodes=args.qaoa_max_nodes,
        qaoa_depth=args.qaoa_depth,
        qaoa_steps=args.qaoa_steps,
        qaoa_stepsize=args.qaoa_stepsize,
        vertex_penalty=args.vertex_penalty,
    )
    solver.create_cfg()
    solver.create_rsf()
    target_graph = {
        "fgr": solver.fgr,
        "cfg": solver.cfg,
        "rsf": solver.rsf,
    }[args.qaoa_graph]
    solution, info = solver.solve_problem_xanadu(target_graph, problem="max_cut")
    return {
        "solution_nodes": sorted(solution),
        "cut_edges": info.get("cut_edges", []),
        "num_cut_edges": info["num_cut_edges"],
        "total_edges": len(info["subgraph"].edges()) if info.get("subgraph") is not None else 0,
        "bitstring": info["bitstring"],
        "note": info["note"],
    }


def analyze_game(game: SGFGame, args: argparse.Namespace) -> None:
    boards = game.board_sequence()
    total_boards = len(boards)
    end = total_boards - 1 if args.end is None else min(args.end, total_boards - 1)
    start = max(args.start, 0)

    plot_dir = args.plot_dir
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Juego SGF: {game.metadata.get('PB', 'Negro')} vs {game.metadata.get('PW', 'Blanco')}")
    print(f"Tablero: {game.size}x{game.size} | Jugadas: {len(game.moves)}")
    print(f"Analizando posiciones {start}..{end}\n")

    for idx in range(start, end + 1):
        board = boards[idx]
        last_move = None if idx == 0 else game.moves[idx - 1]
        move_label = "Inicio" if last_move is None else f"{last_move[0]} {coord_to_label(last_move[1], game.size)}"
        counts = stone_counts(board)

        qgo = QuantumGoGraph(board, threshold=args.threshold)
        qgo.create_cfg()
        cfg_stats = summarize_cfg(qgo.cfg["nodes"] if qgo.cfg else {})

        gbs_stats = None
        if args.gbs_samples > 0:
            gbs_stats = pipeline_cfg_to_gbs_features(board, samples=args.gbs_samples)

        qaoa_stats = None
        if args.qaoa:
            qaoa_stats = run_qaoa(board, args)

        print(f"== Posición {idx} | {move_label} ==")
        print(
            f"  Piedras: negras={counts.get(-1, 0)}, blancas={counts.get(1, 0)}, vacías={counts.get(0, 0)}"
        )
        print(
            f"  CFG: clusters={cfg_stats['clusters']}, blancos={cfg_stats['white_clusters']}, negros={cfg_stats['black_clusters']}, max_cluster={cfg_stats['largest_cluster']}"
        )
        if gbs_stats:
            print(
                f"  GBS: muestras={gbs_stats['samples']}, tamaño_medio={gbs_stats['avg_size']:.2f}"
            )
        if qaoa_stats:
            print(
                f"  QAOA Max-Cut: nodos_solución={qaoa_stats['solution_nodes']}, aristas_cortadas={qaoa_stats['num_cut_edges']}/{qaoa_stats['total_edges']}, bitstring={qaoa_stats['bitstring']}"
            )
            print(f"  Nota QAOA: {qaoa_stats['note']}")
        print()

        if plot_dir is not None:
            annotation_lines = [
                move_label,
                f"Negras={counts.get(-1, 0)} Blancas={counts.get(1, 0)} Vacías={counts.get(0, 0)}",
                f"Clusters B/W={cfg_stats['black_clusters']}/{cfg_stats['white_clusters']} Max={cfg_stats['largest_cluster']}",
            ]
            if gbs_stats:
                annotation_lines.append(
                    f"GBS muestras={gbs_stats['samples']:.0f} avg={gbs_stats['avg_size']:.2f}"
                )
            if qaoa_stats:
                annotation_lines.append(
                    f"Max-Cut: {qaoa_stats['num_cut_edges']}/{qaoa_stats['total_edges']}"  # noqa: E501
                )
            annotation = "\n".join(annotation_lines)

            fig, ax = plt.subplots(figsize=(5, 5))
            plot_board_with_analysis(
                board,
                last_move=None if last_move is None else last_move[1],
                title=f"Posición {idx}",
                max_cut_nodes=qaoa_stats["solution_nodes"] if qaoa_stats else None,
                cut_edges=qaoa_stats["cut_edges"] if qaoa_stats else None,
                annotation=annotation,
                ax=ax,
            )
            output_path = plot_dir / f"{args.plot_prefix}{idx:03d}.{args.plot_format}"
            fig.savefig(output_path, dpi=args.plot_dpi, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.path.exists():
        raise FileNotFoundError(args.path)
    game = load_sgf(args.path)
    analyze_game(game, args)


if __name__ == "__main__":
    main()
