import bz2
import lzma

import networkx as nx
import scipy.sparse as sparse

from . import utils


def create_sparse_matrix_from_file(file):
    """Create a NetworkX graph from DIMACS edge-format text.

    The parser materializes vertices declared by ``p edge n m`` so isolated
    vertices are preserved.  It also accepts the legacy local edge form
    ``p u v`` used by existing local complement files.

    Args:
        file: A file-like object (e.g., an opened file) containing the matrix data.

    Returns:
        A NetworkX Graph.

    Raises:
        ValueError: If the input matrix is not the correct DIMACS format.
    """
    graph = nx.Graph()
    for line_number, raw_line in enumerate(file, start=1):
        parts = raw_line.split()
        if not parts or parts[0] == "c":
            continue

        if parts[0] == "p" and len(parts) >= 4 and parts[1] == "edge":
            try:
                num_vertices = int(parts[2])
            except ValueError as exc:
                raise ValueError(f"Invalid DIMACS vertex count at line {line_number}") from exc
            if num_vertices < 0:
                raise ValueError(f"Invalid negative vertex count at line {line_number}")
            graph.add_nodes_from(range(num_vertices))
            continue

        is_edge_line = parts[0] == "e" or (parts[0] == "p" and len(parts) == 3)
        if not is_edge_line:
            continue

        try:
            u, v = int(parts[-2]), int(parts[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid DIMACS edge at line {line_number}") from exc
        if u <= 0 or v <= 0:
            raise ValueError(f"Vertex identifiers must be positive at line {line_number}")
        graph.add_edge(u - 1, v - 1)

    return graph

def save_sparse_matrix_to_file(matrix, filename):
    """
    Writes a SciPy sparse matrix to a DIMACS format.

    Args:
        matrix: The SciPy sparse matrix.
        filename: The name of the output text file.
    """
    matrix = matrix.copy()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    rows, cols = matrix.nonzero()

    with open(filename, 'w') as f:
        f.write(f"p edge {matrix.shape[0]} {matrix.nnz // 2}" + "\n")
        for i, j in zip(rows, cols):
            if i < j:
                f.write(f"e {i + 1} {j + 1}" + "\n")
    

def read(filepath):
    """Reads a file and returns its lines in an array format.

    Args:
        filepath: The path to the file.

    Returns:
        A NetworkX Graph.

    Raises:
        FileNotFoundError: If the file is not found.
    """

    try:
        extension = utils.get_extension_without_dot(filepath)
        if extension == 'xz' or extension == 'lzma':
            with lzma.open(filepath, 'rt') as file:
                matrix = create_sparse_matrix_from_file(file)
        elif extension == 'bz2' or extension == 'bzip2':
            with bz2.open(filepath, 'rt') as file:
                matrix = create_sparse_matrix_from_file(file)
        else:
            with open(filepath, 'r') as file:
                matrix = create_sparse_matrix_from_file(file)
        
        return matrix
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
