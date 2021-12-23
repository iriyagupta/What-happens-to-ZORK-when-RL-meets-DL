import numpy as np
import matplotlib.pyplot as plt

with open('timestamp.txt', 'r') as f:
    lines = f.readlines()
    lines = [l for l in lines if len(l) > 1]
    lines = [l.strip().split() for l in lines]
    lines = [(l[0], float(l[1])) for l in lines]

    game_info = [lines[i:i+6] for i in range(0, len(lines), 6)]

    for g in game_info:
        g[0] = (g[0][1] + g[1][1] + g[2][1] + g[3][1])/4

    avg_loss = []
    avg_time = []
    eps = []

    for g in game_info:
        avg_loss.append(g[0])
        eps.append(g[-2][1])
        avg_time.append(g[-1][1])

    t = list(range(32))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Game')

    markevery=[14, 25]

    ax1.set_ylabel('Q-net training time (avg)')
    ax1.plot(t, avg_time, '-gD', markevery=markevery)
    ax1.tick_params(axis='y')
    ax1.set_label('Effect of new discoveries on learning')

    plt.show()