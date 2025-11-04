from xanadu_go import QuantumGoGraph


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
    go_graph = QuantumGoGraph(BOARD_MATRIX)
    print("Board state (ASCII representation):")
    go_graph.print_board()

    print("\nSingle-qubit expectations (FGR proxy):")
    print(go_graph._fgr_expectations.reshape(go_graph.size, go_graph.size))

    go_graph.create_cfg()
    print("\nCommon Fate Graph summary:")
    go_graph.summarize_cfg()


if __name__ == "__main__":
    main()
