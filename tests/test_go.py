import itertools
from collections import defaultdict, namedtuple
import numpy as np

import pytest

from thick_goban import go
import tests.test_fixtures as fixt

fixture_params = [n for n in range(9, 26, 2)]


@pytest.fixture(params=fixture_params)
def position(request):
    return fixt.open_position()(size=request.param)


@pytest.fixture(params=fixture_params)
def position_moves(request):
    return fixt.first_position()(s=request.param)


def test_make_neighbors(position):
    """Corners have two neighbors; Edges points have 3, and Centre points have 4.
    """

    def result_row(i, size):
        return [i] + [i + 1] * (size - 2) + [i]

    size = position.size
    neigh_counts = [0] * (size ** 2)
    first_row = result_row(2, size)
    last_row = result_row(2, size)
    middle_row = result_row(3, size)
    desired_result = first_row + (middle_row) * (size - 2) + last_row

    for c, neighs in go.make_neighbors(size=size):
        for pt in list(neighs):
            neigh_counts[pt] += 1

    assert desired_result == neigh_counts


def test_Position_initial(position):
    assert position.kolock is None
    assert position.next_player is go.BLACK
    assert len(position.board._pointers) == position.size ** 2
    assert position.komi == -7.5


def test_Position_groups(position_moves):
    """ The fixture makes a number of moves,
    and these assertions test the results in the fixture.
    Upper left
    .X.Xo.
    X.Xoo.
    XXX...
    ......
    Lower right
    ......
    ..oooo
    .oooXX
    .oXXX.
    """
    position, moves = position_moves
    s = position.size
    QuasiGroup = namedtuple('QuasiGroup', 'colour, stones')
    groups = [QuasiGroup(colour=1, stones={3},),
              QuasiGroup(colour=1, stones={1},),
              QuasiGroup(colour=1, stones={s, 2*s, 2*s+1, 2*s+2, s+2}, ),
              QuasiGroup(colour=1, stones={s**2-s-1, s**2-3, s**2-4, s**2-s-2, s**2-2}),
              QuasiGroup(colour=-1, stones={4, s+3, s+4}, ),
              QuasiGroup(colour=-1, stones={s*(s-2)-4, s*(s-2)-3, s*(s-2)-2, s*(s-2)-1,
                                          s*(s-1)-5, s*(s-1)-4, s*(s-1)-3,
                                          s**2-5}),
              ]
    position.board.discover_all_libs()

    for pt in position.board:
        group, _ = position.board._find(pt=pt)
        assert group is None or QuasiGroup(colour=group.colour, stones=group._stones) in groups

    position.move(move_pt=s-1, colour=go.BLACK)
    assert position.board._find(s-1)[0] == go.Group(stones={s-1}, colour=go.BLACK,)
    position.move(move_pt=2, colour=go.BLACK)
    position.board.discover_liberties(group_pt=2, limit=np.infty)
    assert position.board._find(1)[0] == go.Group(colour=go.BLACK,
                                               stones={1, 2, 3,
                                                       s, s+2,
                                                       2*s, 2*s+1, 2*s+2,})


def test_move_capture(position_moves):
    position, moves = position_moves
    s = position.size

    position.move(s ** 2 - 1, go.WHITE)  # capture corner
    assert position.board._find(s ** 2 - 1)[0] == go.Group(stones={s**2-1}, colour=go.WHITE,)

    position.board.discover_liberties(s ** 2 - 5, limit=np.infty)
    position.board.discover_liberties(s ** 2 - 1, limit=np.infty)
    group1, _ = position.board._find(pt=s**2-1)
    group2, _ = position.board._find(pt=s**2-5)
    for lib in (group1._liberties | group2._liberties):
        assert position.board._find(lib)[0] is None
        assert lib in position.actions

    def kolock_point():
        position.move(2, go.WHITE)
        position.move(3, go.BLACK) # the play on a ko

    with pytest.raises(go.MoveError) as excinfo:
        kolock_point()
    assert 'ko locked point' in str(excinfo.value)

    assert position.board._find(2)[0] == go.Group(colour=go.WHITE, stones={2}, )


def test_position_playout(position):
    """Added to test a bug where during playout _stones were not being removed
    when they had no _liberties.
    """
    passes = 0
    moves = []
    board = position.board
    while passes < 2:
        try:
            moves.append(position.random_move())
        except go.MoveError:
            position.pass_move()
            passes +=1
        else:
            passes = 0
            for pt in board:
                if board.colour(pt) is not go.OPEN:
                    for neigh_pt in board.neighbors[pt]:
                        group, _ = board._find(pt=neigh_pt)
                        try:
                            # pt is no longer in the liberty set
                            assert pt not in group._liberties
                        except AttributeError:
                            assert group is None
                else:
                    assert pt in position.actions


def test_move_exceptions(position_moves):
    position, moves = position_moves

    def suicide_move():
        position.move((position.size ** 2) - 1, go.BLACK)

    def suicide_moveII():
        position.move(0, go.WHITE)

    def play_on_all_moves():
        for pt in moves:
            def existing_stone():
                position.move(pt, go.WHITE)
            yield existing_stone

    def bad_colour():
        position.move(4 * position.size, 't')

    excep_functionsI = [(suicide_move,'friendly eye'),
                        (suicide_moveII,'self capture'),
                        (bad_colour,'Unrecognized move colour')
                       ]

    excep_functions2 = [(func,'Playing on another stone')
                            for func in play_on_all_moves()]
    excep_functions = dict(excep_functionsI+excep_functions2)
    for excep_func, message in excep_functions.items():
        with pytest.raises(go.MoveError) as excinfo:
            excep_func()
        assert message in str(excinfo.value)


def test_position_actions(position_moves):
    position, moves = position_moves
    s = position.size
    assert (set(range(s ** 2)) - position.actions) - set(moves.keys()) == set()

    assert 2 in position.actions
    assert 3 not in position.actions
    position.move(move_pt=2, colour=go.WHITE)
    assert 2 not in position.actions
    assert 3 in position.actions    # added after capture

    position.move(move_pt=s**2-1, colour=go.WHITE)
    assert {s**2-s-1, s**2-3, s**2-4, s**2-s-2, s**2-2}.issubset(position.actions)

    term_position, moves = position.random_playout()
    for action in term_position.actions:
        with pytest.raises(go.MoveError):
            term_position.move(action, colour=go.BLACK)
        with pytest.raises(go.MoveError):
            term_position.move(action, colour=go.WHITE)


def test_score(position_moves):
    position, moves = position_moves
    black_stones, black_liberties = 12, 8
    white_stones, white_liberties = 11, 11
    komi = -7.5
    assert position.score() == black_stones + black_liberties - white_stones - white_liberties + komi

    term_position, moves = position.random_playout()
    assert term_position is not position

    score = term_position.komi
    for pt in term_position.board:
        score += term_position.board.colour(pt)
        if term_position.board.colour(pt) == go.OPEN:
            # will find the 1 neighbor colour unless it is a seki liberty, then its 0
            neigh_colours = set([term_position.board.colour(neigh_pt)
                                    for neigh_pt in term_position.board.neighbors[pt]])
        else:
            neigh_colours = []
        score += sum(neigh_colours)

    assert score == term_position.score()


def test_Group_init():
    for col, pt in itertools.product([go.BLACK, go.WHITE], range(361)):
        group = go.Group(colour=col, stones={pt} )
        assert group.colour == col
        assert group.size == 1
        assert group.liberties == 0


def test_failing_sgfs():
    """Test strange MoveError behaviour
    
    While parsing all available sgfs, thick_goban seemed to be raising strange exceptions.
    Looking at the sgfs in questions didn't reveal what the problem is in any immediate way.
    As such, this test is intended to replicate the error.

    These moves are from sgf 'chap005a.sgf' of the test sets.
    Move 185 raised an Illegal MoveError for no apparent reason at the time this test was implemented.
    
    Reason found: I forgot about the handicap stones, so various captures didn't occur as expected in the play out,
    which caused the MoveErrors.
    I've added in the appropriate handicap stones for these sgfs to test the handicap functionality and the playouts.
    """
    chap005a_handi = [(pt, 1) for pt in [72, 288, 300]]
    chap005a_moves = [61, 59, 40, 39, 41, 77, 47, 117, 111, 92, 110, 51, 317, 244, 206, 319, 279, 242, 312, 310, 337, 
                      192, 321, 302, 339, 204, 168, 65, 66, 45, 99, 103, 118, 85, 69, 70, 88, 136, 83, 102, 63, 84, 137,
                      155, 50, 89, 105, 159, 108, 86, 106, 156, 44, 240, 54, 53, 93, 74, 73, 55, 91, 35, 222, 241, 224, 
                      223, 243, 262, 261, 282, 280, 188, 205, 207, 226, 225, 245, 263, 169, 264, 187, 246, 181, 258, 296, 
                      236, 274, 276, 311, 295, 314, 313, 315, 291, 238, 200, 179, 182, 162, 273, 293, 255, 199, 218, 121, 
                      160, 139, 178, 122, 123, 157, 143, 163, 161, 306, 104, 287, 249, 180, 268, 307, 328, 330, 308, 289, 
                      269, 327, 309, 219, 201, 217, 237, 256, 275, 254, 257, 235, 256, 197, 176, 196, 177, 184, 203, 213, 
                      326, 325, 346, 193, 174, 211, 248, 191, 173, 79, 78, 271, 253, 234, 252, 272, 292, 251, 332, 270, 
                      286, 305, 323, 285, 267, 195, 175, 344, 304, 348, 343, 347, 266, 290, 324, 294, 185, 46, 64, 26, 
                      92, 20, 19, 2, 338, 357, 320, 340, 164, 144, 172, 147, 210, 333, 82]

    chap070d_handi = [(pt,1) for pt in [60, 72, 288]]
    chap070d_moves = [301, 318, 316, 320, 321, 300, 282, 339, 243, 279, 111, 149, 109, 206, 70, 281, 245, 74, 43, 98, 309, 230, 294, 307,
                     93, 51, 50, 52, 31, 69, 71, 54, 91, 108, 48, 88, 90, 147, 73, 92, 127, 126, 73, 53, 55, 92, 146, 128, 73, 263, 75,
                     244, 165, 163, 144, 67, 66, 86, 65, 145, 226, 264, 164, 182, 203, 47, 29, 143, 125, 107, 124, 123, 142, 104, 201,
                     181, 225, 205, 262, 302, 265, 283, 224, 223, 204, 222, 202, 261, 207, 168, 187, 186, 188, 167, 132, 200, 239, 257,
                     219, 218, 237, 238, 220, 198, 217, 199, 259, 296, 242, 236, 154, 311, 292, 310, 291, 328, 276, 277, 256, 258, 255,
                     216, 234, 214, 312, 314, 331, 290, 271, 308, 333, 116, 211, 231, 229, 191, 210, 153, 268, 286, 233, 213, 287, 305,
                     269, 267, 172, 173, 174, 171, 192, 172, 215]
    failures = []
    
    try:
        go.Position(moves=chap005a_moves, setup=chap005a_handi)
    except go.MoveError as err:
        failures.append(err)
    try:
        go.Position(moves=chap070d_moves, setup=chap070d_handi)
    except go.MoveError as err:
        failures.append(err)
        
    assert not failures
    

def test_ko_lock_problem():
    """Ko lock error is triggering incorrectly

    It is being triggered even when the capture is of two or more stones.
    """
    moves = [0, 19, 20, 2, 21,  3, 22, 4, 23, 38, 5, 1, 0]
    try:
        go.Position(moves=moves)
    except go.MoveError as err:
        assert not str(err)
