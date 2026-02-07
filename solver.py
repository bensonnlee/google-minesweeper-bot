"""Minesweeper solver logic."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, FrozenSet, Dict

# Computational safeguards
MAX_SOLUTIONS = 100_000    # Stop enumeration after this many solutions
MAX_REGION_SIZE = 70       # Fall back to region-based enumeration if border too large


@dataclass
class Constraint:
    """Represents a constraint from a numbered cell.

    A numbered cell tells us exactly how many mines are among its unknown neighbors.
    """
    cell: Tuple[int, int]                    # The numbered cell position
    unknowns: FrozenSet[Tuple[int, int]]     # Unknown neighbors
    mines_needed: int                         # number - flagged_count


class Solver:
    """Determines the next move based on current grid state."""

    def __init__(self, cells: List[List], mine_count: int = None):
        """Initialize solver with current grid state.

        Args:
            cells: 2D array of cell states ('unknown' | 'flag' | 0-8).
            mine_count: Total mines in the game (enables endgame logic).
        """
        self.cells = cells
        self.rows = len(cells)
        self.cols = len(cells[0]) if cells else 0
        self.mine_count = mine_count
        self.is_guess = False
        self.guess_probability = None  # P(mine) for the guessed cell, if available

    def get_move(self) -> Tuple[Optional[str], List[Tuple[int, int]]]:
        """Determine the next move(s) to make.

        Uses a multi-pass approach:
        1. Endgame check (if mine_count known)
        2-4. Iterative chaining: basic flagging, basic safe cells, and
             subset analysis loop until no new moves are found
        5. Full enumeration per region (expensive, only if passes 2-4 found nothing)
        6. Probability-based guess
        7. Random fallback

        Returns:
            Tuple of (action, cells) where:
            - action: 'click' | 'flag' | None (None means game over or stuck)
            - cells: List of (row, col) tuples to perform action on
        """
        # Pass 1: Endgame check (takes priority)
        action, cells = self._check_endgame()
        if action:
            return (action, cells)

        # Iterate Passes 2-4 until convergence (allows inferences to chain)
        all_mines = set()
        all_safe = set()
        all_probabilities = {}
        modified_cells = {}  # (r, c) -> original value, for restoring after

        for _ in range(10):  # Safety limit
            prev_total = len(all_mines) + len(all_safe)

            # Pass 2: Basic flagging
            new_mines = set(self._find_cells_to_flag()) - all_mines
            all_mines.update(new_mines)
            self._temporarily_flag(new_mines, modified_cells)

            # Pass 3: Basic safe cells
            all_safe.update(self._find_safe_cells())

            # Pass 4: Subset analysis
            constraints = self._build_constraints()
            if constraints:
                definite_mines, definite_safe = self._subset_analysis(constraints)
                new_subset_mines = definite_mines - all_mines
                all_mines.update(definite_mines)
                all_safe.update(definite_safe)
                self._temporarily_flag(new_subset_mines, modified_cells)

            if len(all_mines) + len(all_safe) == prev_total:
                break

        # Restore original cell states
        for (r, c), original_value in modified_cells.items():
            self.cells[r][c] = original_value

        # Pass 5: Full enumeration (only if passes 2-4 found nothing)
        border_cells = self._get_border_cells()
        if not all_mines and not all_safe and border_cells and constraints:
            # Try global enumeration first if border is small enough
            if len(border_cells) <= MAX_REGION_SIZE:
                regions = [border_cells]
            else:
                # Fall back to region-based for very large borders
                regions = self._find_connected_regions(border_cells, constraints)

            for region in regions:
                if len(region) > MAX_REGION_SIZE:
                    continue

                solutions, _complete = self._enumerate_solutions(region, constraints)
                if solutions:
                    mines, safe, probs = self._process_solutions(region, solutions)
                    all_mines.update(mines)
                    all_safe.update(safe)
                    all_probabilities.update(probs)

        # Return ALL accumulated moves
        if all_mines:
            return ('flag', list(all_mines))
        if all_safe:
            return ('click', list(all_safe))

        # Pass 6: Probability-based guess
        if all_probabilities:
            unknown_cells = self._find_unknown_cells()
            interior_cells = set(unknown_cells) - border_cells

            best_guess, best_prob = self._select_best_guess(all_probabilities, interior_cells)
            if best_guess:
                self.is_guess = True
                self.guess_probability = best_prob
                return ('click', [best_guess])

        # Pass 7: Random fallback
        unknown_cells = self._find_unknown_cells()
        if unknown_cells:
            self.is_guess = True
            self.guess_probability = None
            # Prefer corners and edges (slightly lower mine probability)
            prioritized = self._prioritize_unknowns(unknown_cells)
            return ('click', [prioritized[0]])

        # No moves available (game over or won)
        return (None, [])

    def _temporarily_flag(self, cells: Set[Tuple[int, int]],
                          modified_cells: Dict[Tuple[int, int], str]):
        """Mark cells as flags in self.cells so subsequent passes see them.

        Records which cells were modified so they can be restored later.
        Only modifies cells that are still 'unknown'.
        """
        for r, c in cells:
            if self.cells[r][c] == 'unknown':
                modified_cells[(r, c)] = 'unknown'
                self.cells[r][c] = 'flag'

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all valid neighbor coordinates for a cell.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            List of (row, col) tuples for all 8 neighbors that exist.
        """
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    def _numbered_cell_info(self):
        """Yield (cell_value, unknowns, flag_count) for each numbered cell with neighbors."""
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.cells[row][col]
                if not isinstance(cell, int) or cell == 0:
                    continue

                unknowns = []
                flag_count = 0
                for nr, nc in self.get_neighbors(row, col):
                    neighbor_state = self.cells[nr][nc]
                    if neighbor_state == 'unknown':
                        unknowns.append((nr, nc))
                    elif neighbor_state == 'flag':
                        flag_count += 1

                yield (row, col), cell, unknowns, flag_count

    def _find_cells_to_flag(self) -> List[Tuple[int, int]]:
        """Find unknown cells that must be mines.

        For each numbered cell, if unknowns + flags == number, all unknowns are mines.
        """
        cells_to_flag = set()
        for _pos, cell, unknowns, flag_count in self._numbered_cell_info():
            remaining_mines = cell - flag_count
            if remaining_mines > 0 and len(unknowns) == remaining_mines:
                cells_to_flag.update(unknowns)
        return list(cells_to_flag)

    def _find_safe_cells(self) -> List[Tuple[int, int]]:
        """Find unknown cells that are guaranteed safe.

        For each numbered cell, if flags == number, all unknowns are safe.
        """
        safe_cells = set()
        for _pos, cell, unknowns, flag_count in self._numbered_cell_info():
            if flag_count == cell and unknowns:
                safe_cells.update(unknowns)
        return list(safe_cells)

    def _find_unknown_cells(self) -> List[Tuple[int, int]]:
        """Find all unknown (unclicked, unflagged) cells."""
        unknowns = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.cells[row][col] == 'unknown':
                    unknowns.append((row, col))
        return unknowns

    def _prioritize_unknowns(self, unknowns: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Prioritize unknown cells for random selection.

        Prefers cells that are:
        1. Not adjacent to any numbered cells (less likely to be mines)
        2. Corners and edges
        """
        def priority_score(cell):
            row, col = cell
            score = 0

            # Prefer cells not adjacent to numbers
            neighbors = self.get_neighbors(row, col)
            adjacent_numbers = sum(
                1 for nr, nc in neighbors
                if isinstance(self.cells[nr][nc], int) and self.cells[nr][nc] > 0
            )
            score -= adjacent_numbers * 10

            # Slight preference for corners/edges
            if row == 0 or row == self.rows - 1:
                score += 1
            if col == 0 or col == self.cols - 1:
                score += 1

            return score

        return sorted(unknowns, key=priority_score, reverse=True)

    def _count_flags_and_unknowns(self) -> Tuple[int, int]:
        """Count total flags and unknown cells on the board.

        Returns:
            Tuple of (flag_count, unknown_count).
        """
        flags = 0
        unknowns = 0
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.cells[row][col]
                if cell == 'flag':
                    flags += 1
                elif cell == 'unknown':
                    unknowns += 1
        return flags, unknowns

    def _check_endgame(self) -> Tuple[Optional[str], List[Tuple[int, int]]]:
        """Check for endgame conditions when mine_count is known.

        Returns:
            (action, cells) if endgame move found, (None, []) otherwise.
        """
        if self.mine_count is None:
            return (None, [])

        flags, unknowns = self._count_flags_and_unknowns()
        remaining_mines = self.mine_count - flags

        # If all remaining unknowns are mines, flag them all
        if remaining_mines == unknowns and unknowns > 0:
            unknown_cells = self._find_unknown_cells()
            return ('flag', unknown_cells)

        # If no mines remain, all unknowns are safe
        if remaining_mines == 0 and unknowns > 0:
            unknown_cells = self._find_unknown_cells()
            return ('click', unknown_cells)

        return (None, [])

    def _build_constraints(self) -> List[Constraint]:
        """Build constraint list from all numbered cells.

        Returns:
            List of Constraint objects for cells with unknown neighbors.
        """
        constraints = []
        for pos, cell, unknowns, flag_count in self._numbered_cell_info():
            mines_needed = cell - flag_count
            if unknowns and mines_needed >= 0:
                constraints.append(Constraint(
                    cell=pos,
                    unknowns=frozenset(unknowns),
                    mines_needed=mines_needed
                ))
        return constraints

    def _subset_analysis(self, constraints: List[Constraint]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """Analyze constraint pairs for subset relationships.

        When one constraint's unknowns are a subset of another's, we can deduce
        information about cells outside the subset.

        Example:
            - Constraint A: needs 2 mines in {X, Y, Z}
            - Constraint B: needs 2 mines in {Y, Z}
            - Deduction: X must be safe (B accounts for all mines in shared region)

        Args:
            constraints: List of Constraint objects.

        Returns:
            Tuple of (definite_mines, definite_safe) sets.
        """
        definite_mines = set()
        definite_safe = set()

        for i, c1 in enumerate(constraints):
            for j, c2 in enumerate(constraints):
                if i == j:
                    continue

                # Check if c2's unknowns are a subset of c1's
                if c2.unknowns <= c1.unknowns:
                    outside = c1.unknowns - c2.unknowns
                    if not outside:
                        continue

                    # mines_outside = mines in c1's region that are NOT in c2's region
                    mines_outside = c1.mines_needed - c2.mines_needed

                    if mines_outside < 0:
                        # Contradiction - shouldn't happen in valid game state
                        continue
                    elif mines_outside == 0:
                        # All mines are in the subset (c2), so outside cells are safe
                        definite_safe.update(outside)
                    elif mines_outside == len(outside):
                        # All outside cells must be mines
                        definite_mines.update(outside)

        return definite_mines, definite_safe

    def _get_border_cells(self) -> Set[Tuple[int, int]]:
        """Find unknown cells adjacent to at least one numbered cell.

        Border cells are the only cells that provide information for constraint
        satisfaction. Interior cells (surrounded by unknowns) don't affect constraints.

        Returns:
            Set of (row, col) tuples for border cells.
        """
        border = set()

        for row in range(self.rows):
            for col in range(self.cols):
                if self.cells[row][col] != 'unknown':
                    continue

                # Check if adjacent to any numbered cell
                for nr, nc in self.get_neighbors(row, col):
                    neighbor = self.cells[nr][nc]
                    if isinstance(neighbor, int) and neighbor > 0:
                        border.add((row, col))
                        break

        return border

    def _find_connected_regions(self, border_cells: Set[Tuple[int, int]],
                                 constraints: List[Constraint]) -> List[Set[Tuple[int, int]]]:
        """Group border cells into connected regions that share constraints.

        Cells are in the same region if they appear in the same constraint.
        This allows us to enumerate solutions independently for each region.

        Args:
            border_cells: Set of border cell positions.
            constraints: List of Constraint objects.

        Returns:
            List of sets, each containing cells in a connected region.
        """
        if not border_cells:
            return []

        # Build adjacency: cells are connected if they share a constraint
        cell_to_constraints = {cell: set() for cell in border_cells}
        for idx, constraint in enumerate(constraints):
            for cell in constraint.unknowns:
                if cell in cell_to_constraints:
                    cell_to_constraints[cell].add(idx)

        # Union-Find for grouping
        parent = {cell: cell for cell in border_cells}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union cells that share constraints
        for constraint in constraints:
            cells_in_constraint = [c for c in constraint.unknowns if c in border_cells]
            for i in range(1, len(cells_in_constraint)):
                union(cells_in_constraint[0], cells_in_constraint[i])

        # Group by root
        groups = {}
        for cell in border_cells:
            root = find(cell)
            if root not in groups:
                groups[root] = set()
            groups[root].add(cell)

        return list(groups.values())

    def _enumerate_solutions(self, region_cells: Set[Tuple[int, int]],
                              constraints: List[Constraint],
                              max_solutions: int = MAX_SOLUTIONS) -> Tuple[List[Dict[Tuple[int, int], bool]], bool]:
        """Enumerate all valid mine placements for a region using backtracking.

        Args:
            region_cells: Cells in this region to assign.
            constraints: Constraints involving these cells.
            max_solutions: Stop after finding this many solutions.

        Returns:
            Tuple of (solutions, complete) where:
                - solutions: List of dicts mapping cell -> is_mine
                - complete: True if enumeration finished, False if truncated
        """
        # Order cells by constraint count (most constrained first = better pruning)
        def constraint_count(cell):
            return sum(1 for c in constraints if cell in c.unknowns)
        cells = sorted(region_cells, key=constraint_count, reverse=True)

        # Use all constraints that overlap with this region (not just fully contained).
        # constraint_possible() handles out-of-region cells conservatively.
        relevant_constraints = [
            c for c in constraints
            if c.unknowns & region_cells
        ]

        solutions = []
        complete = True

        def constraint_possible(assignment: Dict[Tuple[int, int], bool], constraint: Constraint) -> bool:
            """Check if constraint can still be satisfied."""
            unknowns_in_region = constraint.unknowns & region_cells
            assigned = {c for c in unknowns_in_region if c in assignment}
            unassigned = unknowns_in_region - assigned

            mines_assigned = sum(1 for c in assigned if assignment[c])

            # Too many mines already
            if mines_assigned > constraint.mines_needed:
                return False

            # Not enough cells left to place required mines
            mines_still_needed = constraint.mines_needed - mines_assigned
            if mines_still_needed > len(unassigned):
                return False

            return True

        def constraint_satisfied(assignment: Dict[Tuple[int, int], bool], constraint: Constraint) -> bool:
            """Check if constraint is fully satisfied (all cells assigned)."""
            unknowns_in_region = constraint.unknowns & region_cells
            if not all(c in assignment for c in unknowns_in_region):
                return False  # Not fully assigned yet, cannot verify

            mines = sum(1 for c in unknowns_in_region if assignment[c])
            return mines == constraint.mines_needed

        def backtrack(index: int, assignment: Dict[Tuple[int, int], bool]):
            nonlocal complete

            if len(solutions) >= max_solutions:
                complete = False
                return

            # All cells assigned
            if index >= len(cells):
                # Verify all constraints satisfied
                if all(constraint_satisfied(assignment, c) for c in relevant_constraints):
                    solutions.append(dict(assignment))
                return

            cell = cells[index]

            # Try SAFE (False = not a mine)
            assignment[cell] = False
            if all(constraint_possible(assignment, c) for c in relevant_constraints):
                backtrack(index + 1, assignment)

            if not complete:
                return  # Don't corrupt assignment, just stop exploring

            # Try MINE (True = is a mine)
            assignment[cell] = True
            if all(constraint_possible(assignment, c) for c in relevant_constraints):
                backtrack(index + 1, assignment)

            del assignment[cell]

        backtrack(0, {})
        return solutions, complete

    def _process_solutions(self, cells: Set[Tuple[int, int]],
                           solutions: List[Dict[Tuple[int, int], bool]]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Dict[Tuple[int, int], float]]:
        """Classify cells as definite mines/safe and compute probabilities from solutions.

        Returns:
            Tuple of (definite_mines, definite_safe, probabilities).
        """
        definite_mines = set()
        definite_safe = set()

        for cell in cells:
            mine_in_all = all(sol.get(cell, False) for sol in solutions)
            safe_in_all = all(not sol.get(cell, False) for sol in solutions)

            if mine_in_all:
                definite_mines.add(cell)
            elif safe_in_all:
                definite_safe.add(cell)

        probabilities = self._calculate_probabilities(solutions)
        return definite_mines, definite_safe, probabilities

    def _calculate_probabilities(self, solutions: List[Dict[Tuple[int, int], bool]]) -> Dict[Tuple[int, int], float]:
        """Calculate mine probability for each cell from enumerated solutions.

        Args:
            solutions: List of valid solution dicts.

        Returns:
            Dict mapping cell -> P(cell is mine).
        """
        if not solutions:
            return {}

        probabilities = {}
        total = len(solutions)

        # Get all cells from solutions
        all_cells = set()
        for sol in solutions:
            all_cells.update(sol.keys())

        for cell in all_cells:
            mine_count = sum(1 for sol in solutions if sol.get(cell, False))
            probabilities[cell] = mine_count / total

        return probabilities

    def _select_best_guess(self, probabilities: Dict[Tuple[int, int], float],
                           interior_cells: Set[Tuple[int, int]]) -> Tuple[Tuple[int, int], float]:
        """Select the cell with lowest mine probability for guessing.

        Args:
            probabilities: Dict mapping border cell -> P(mine).
            interior_cells: Set of cells not adjacent to any numbers.

        Returns:
            Tuple of (best_cell, mine_probability).
        """
        # Estimate interior probability from global mine count
        interior_prob = 0.5  # Default if no mine count known
        if self.mine_count is not None and interior_cells:
            flags, unknowns = self._count_flags_and_unknowns()
            remaining_mines = self.mine_count - flags

            # Mines in border cells from probabilities
            expected_border_mines = sum(probabilities.values())

            # Remaining mines must be in interior
            interior_mines = remaining_mines - expected_border_mines
            if len(interior_cells) > 0:
                interior_prob = max(0.0, min(1.0, interior_mines / len(interior_cells)))

        # Find minimum probability cell
        best_cell = None
        best_prob = 1.1

        # Check border cells
        for cell, prob in probabilities.items():
            if prob < best_prob:
                best_prob = prob
                best_cell = cell

        # Check interior cells
        if interior_prob < best_prob and interior_cells:
            # Pick an interior cell (prefer corners/edges)
            interior_list = list(interior_cells)
            interior_prioritized = self._prioritize_unknowns(interior_list)
            if interior_prioritized:
                best_cell = interior_prioritized[0]
                best_prob = interior_prob

        # Tiebreaker: corners/edges among cells with similar probability
        if best_cell and probabilities:
            threshold = best_prob + 0.05  # 5% tolerance
            candidates = [c for c, p in probabilities.items() if p <= threshold]
            if candidates:
                prioritized = self._prioritize_unknowns(candidates)
                if prioritized:
                    best_cell = prioritized[0]

        return best_cell, best_prob
