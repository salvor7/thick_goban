import cProfile

from thick_goban import go

def n_playouts(n=100):

    for idx in range(n):
        position = go.Position()
        position.random_playout()


if __name__ == '__main__':
    cProfile.run('n_playouts(10)')
    cProfile.run('n_playouts(100)')
    cProfile.run('n_playouts(1000)')