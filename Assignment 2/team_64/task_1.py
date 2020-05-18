import numpy
import os

directory = "outputs"
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
    os.makedirs(final_directory)

for file in range(1,5):
    
    utility = numpy.zeros((5,4,3))
    utility_next = numpy.zeros((5,4,3))

    dodge = recharge = shoot = 0

    delta = 0

    if file == 1:
    
        f = open('outputs/task_1_trace.txt','w')

        gamma = 0.99
        given_delta = 1e-3

        dodge_reward = -10
        recharge_reward = -10
        shoot_reward = -10

    elif file == 2:

        f = open('outputs/task_2_part_1_trace.txt','w')

        gamma = 0.99
        given_delta = 1e-3

        dodge_reward = -2.5
        recharge_reward = -2.5
        shoot_reward = -0.25

    elif file == 3:

        f = open('outputs/task_2_part_2_trace.txt','w')

        gamma = 0.1
        given_delta = 1e-3

        dodge_reward = -2.5
        recharge_reward = -2.5
        shoot_reward = -2.5

    elif file == 4:

        f = open('outputs/task_2_part_3_trace.txt','w')

        gamma = 0.1
        given_delta = 1e-10

        dodge_reward = -2.5
        recharge_reward = -2.5
        shoot_reward = -2.5

    iteration = 0

    while iteration == 0 or delta >= given_delta:
    
        f.write("iteration={}\n".format(iteration))
        iteration += 1
        
        utility = numpy.copy(utility_next)

        delta = 0

        for i in range(len(utility)): #md_healthoutputs/
            for j in range(len(utility[0])): #no of arrows
                for k in range(len(utility[0][0])): #stamina

                    action = "-1"

                    dodge = recharge = shoot = -1e15

                    if i > 0:

                        # dodge
                        if k == 2:
                            dodge = dodge_reward + gamma * (0.8 * (0.8 * utility[i][min(j+1,3)][k-1] + 0.2 * utility[i][j][k-1]) + 0.2 * (0.8 * utility[i][min(j+1,3)][k-2] + 0.2 * utility[i][j][k-2]))
                        elif k == 1:
                            dodge = dodge_reward + gamma * (0.8 * utility[i][min(j+1,3)][k-1] + 0.2 * utility[i][j][k-1])

                        # recharge
                        recharge = recharge_reward + gamma * (0.8 * utility[i][j][min(k+1,2)] + 0.2 * utility[i][j][min(k,2)])
        
                        # shoot
                        if j > 0 and k > 0:
                            shoot = shoot_reward + gamma * (0.5 * (utility[i-1][j-1][k-1] + utility[i][j-1][k-1]))
                            if i == 1:
                                shoot += 5

                        utility_next[i][j][k] = max(dodge, recharge, shoot)

                        if max(dodge, recharge, shoot) == dodge:
                            action = "DODGE"
                        elif max(dodge, recharge, shoot) == recharge:
                            action = "RECHARGE"
                        elif max(dodge, recharge, shoot) == shoot:
                            action = "SHOOT"

                    delta = max(delta, utility[i][j][k] - utility_next[i][j][k])

                    f.write("({0},{1},{2}):{3}=[{4:.3f}]\n".format(i,j,k,action,round(utility_next[i][j][k],3)))
        f.write("\n\n")