"""Entry point for running QAOA analyses on Go board graphs."""

from __future__ import annotations

from xanadu_go import GoGraphAdvancedXanadu


BOARD_MATRIX = [
        [0, -1, -1, -1, 1, -1, 0, -1, -1],
        [-1, 0, -1, -1, 1, 1, -1, -1, 0],
        [-1, -1, 1, -1, 1, 1, 0, -1, 0],
        [1, 1, 1, 0, 1, 1, 1, -1, 0],
        [1, 0, 1, 1, 0, 1, 1, -1, -1],
        [1, 1, -1, 1, 1, 1, -1, 1, 1],
        [1, 1, -1, -1, -1, -1, -1, 1, 0],
        [1, -1, -1, 0, -1, 0, -1, 1, 1],
        [0, -1, 0, 0, -1, 0, -1, 1, 0],
    ]


def main() -> None:
    go_graph = GoGraphAdvancedXanadu(BOARD_MATRIX)
    go_graph.create_cfg()
    go_graph.create_rsf()

    for graph_name, graph in (
        ("FGR", go_graph.fgr),
        ("CFG", go_graph.cfg),
        ("RSF", go_graph.rsf),
    ):
        if graph is None:
            continue

        print(f"\n--- Resolviendo problemas en {graph_name} ---")

        max_cut_solution, max_cut_info = go_graph.solve_problem_xanadu(graph, problem="max_cut")
        max_cut_edges = max_cut_info["cut_edges"]
        num_cut_edges, total_edges = go_graph.verify_max_cut(max_cut_info["subgraph"], max_cut_solution)
        print("Solución Max-Cut:")
        print(f" - Nodos en el corte: {sorted(max_cut_solution)}")
        print(f" - Aristas cortadas: {num_cut_edges}/{total_edges}")
        print(f" - Bitstring: {max_cut_info['bitstring']}")
        print(f" - Nota: {max_cut_info['note']}")

        vertex_cover_solution, vertex_cover_info = go_graph.solve_problem_xanadu(
            graph, problem="vertex_cover"
        )
        verified_vc = go_graph.verify_vertex_cover(
            vertex_cover_info["subgraph"], vertex_cover_solution
        )
        print("Solución Vertex Cover:")
        print(f" - Nodos en la cobertura: {sorted(vertex_cover_solution)}")
        print(f" - Vertex Cover verificado: {'Sí' if verified_vc else 'No'}")
        print(f" - Bitstring: {vertex_cover_info['bitstring']}")
        print(f" - Nota: {vertex_cover_info['note']}")

        features = go_graph.extract_features_from_solutions(
            max_cut_solution, vertex_cover_solution
        )
        print("Características extraídas:")
        print(f" - Tamaño Max-Cut: {features['max_cut_size']}")
        print(f" - Tamaño Vertex Cover: {features['vertex_cover_size']}")


if __name__ == "__main__":
    main()
