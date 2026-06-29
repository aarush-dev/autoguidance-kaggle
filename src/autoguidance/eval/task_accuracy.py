"""Sudoku task accuracy for Phase 1.

Self-contained: generates Sudoku puzzles deterministically, checks if the
model's generated text contains a valid solution.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import re
import random


def _is_valid_sudoku(grid: List[List[int]]) -> bool:
    """Check if a 9x9 Sudoku grid is valid and complete."""
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        return False
    digits = set(range(1, 10))
    # Rows
    for row in grid:
        if set(row) != digits:
            return False
    # Columns
    for col in range(9):
        if {grid[row][col] for row in range(9)} != digits:
            return False
    # 3x3 boxes
    for br in range(3):
        for bc in range(3):
            box = {grid[br * 3 + r][bc * 3 + c] for r in range(3) for c in range(3)}
            if box != digits:
                return False
    return True


def _make_puzzle(seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Generate a (puzzle, solution) pair deterministically."""
    rng = random.Random(seed)

    def _shuffle_solve(base: List[List[int]]) -> List[List[int]]:
        """Fill a Sudoku by shuffling rows within bands and columns within stacks."""
        grid = [row[:] for row in base]
        nums = list(range(1, 10))
        rng.shuffle(nums)
        mapping = {i + 1: nums[i] for i in range(9)}
        return [[mapping[v] for v in row] for row in grid]

    # Base valid grid
    base = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9, 1, 2, 3],
        [7, 8, 9, 1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7, 8, 9, 1],
        [5, 6, 7, 8, 9, 1, 2, 3, 4],
        [8, 9, 1, 2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8, 9, 1, 2],
        [6, 7, 8, 9, 1, 2, 3, 4, 5],
        [9, 1, 2, 3, 4, 5, 6, 7, 8],
    ]
    solution = _shuffle_solve(base)

    # Remove ~40 cells to make a puzzle
    puzzle = [row[:] for row in solution]
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    for r, c in cells[:40]:
        puzzle[r][c] = 0

    return puzzle, solution


def format_puzzle(puzzle: List[List[int]]) -> str:
    """Format puzzle as a text prompt."""
    lines = []
    for row in puzzle:
        lines.append(" ".join(str(v) if v != 0 else "." for v in row))
    return "Solve this Sudoku puzzle:\n" + "\n".join(lines) + "\nSolution:"


def extract_grid(text: str) -> Optional[List[List[int]]]:
    """Try to parse a 9x9 grid from model output."""
    numbers = re.findall(r"\d", text)
    if len(numbers) < 81:
        return None
    # Take first 81 digits
    flat = [int(n) for n in numbers[:81]]
    return [flat[r * 9 : r * 9 + 9] for r in range(9)]


def sudoku_accuracy(
    model_outputs: List[str],
    solutions: List[List[List[int]]],
) -> float:
    """Fraction of model outputs that contain a valid Sudoku solution."""
    correct = 0
    for output, solution in zip(model_outputs, solutions):
        grid = extract_grid(output)
        if grid is None:
            continue
        if grid == solution or _is_valid_sudoku(grid):
            correct += 1
    return correct / len(solutions) if solutions else 0.0


def make_sudoku_batch(n: int, seed: int = 42) -> Tuple[List[str], List[List[List[int]]]]:
    """Generate n (prompt, solution) pairs."""
    prompts, solutions = [], []
    for i in range(n):
        puzzle, solution = _make_puzzle(seed + i)
        prompts.append(format_puzzle(puzzle))
        solutions.append(solution)
    return prompts, solutions
