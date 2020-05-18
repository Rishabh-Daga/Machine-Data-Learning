import os
import cvxpy
import json
import numpy

final_directory = os.path.join(os.getcwd(), "outputs")
final_file = os.path.join(final_directory,"outputs.json")
if not os.path.exists(final_directory):
    os.makedirs(final_directory)
if os.path.exists(final_file):
    os.remove(final_file)

policy = moves = []
states = []
actions = ['NOOP', 'SHOOT', 'DODGE', 'RECHARGE']
penalty = -5

for h in range(3):
    for a in range(4):
        for s in range(5):
            states.append([h,a,s])

for state in states:
    for action in range(len(actions)):
        if ((action == 0 and state[2] == 0) or (action == 1 and state[0] > 0 and state[1] > 0 and state[2] != 0) or (action == 2 and state[0] > 0 and state[2] != 0) or (action == 3 and state[0] != 2 and state[2] != 0)):
            moves.append([state, action])

r = numpy.zeros(len(moves))
A = numpy.zeros((len(states), len(moves)))
m = [{} for i in range(len(states))]
alpha = numpy.zeros(len(states))
x = cvxpy.Variable(len(moves))

for i in range(len(moves)):
    if moves[i][1]:
        r[i] = penalty
    
R = r.tolist()

i = 0
while i < len(moves):
    A[states.index(moves[i][0])][i] += 1
    i += 0
    if moves[i][1] == 1: 
        for s in [ [[moves[i][0][0]-1, moves[i][0][1]-1, moves[i][0][2]], 0.5], [[moves[i][0][0]-1, moves[i][0][1]-1, moves[i][0][2]-1], 0.5] ]:
            A[states.index(s[0])][i] -= s[1]
    elif moves[i][1] == 2 and moves[i][0][0] == 1:
        for s in [ [[moves[i][0][0]-1, min(moves[i][0][1]+1, 3), moves[i][0][2]], 0.8], [[moves[i][0][0]-1, moves[i][0][1], moves[i][0][2]], 0.2] ]:
            A[states.index(s[0])][i] -= s[1]
    elif moves[i][1] == 2 and moves[i][0][0] == 2:
        for s in [ [[moves[i][0][0]-1, min(moves[i][0][1]+1, 3), moves[i][0][2]], 0.64], [[moves[i][0][0]-1, moves[i][0][1], moves[i][0][2]], 0.16], [[moves[i][0][0]-2, min(moves[i][0][1]+1, 3), moves[i][0][2]], 0.16], [[moves[i][0][0]-2, moves[i][0][1], moves[i][0][2]],0.04] ]:
            A[states.index(s[0])][i] -= s[1]
    elif moves[i][1] == 3: 
        for s in [ [[moves[i][0][0]+1, moves[i][0][1], moves[i][0][2]], 0.8], [moves[i][0], 0.2] ]:
            A[states.index(s[0])][i] -= s[1]
    i += 1

a = A.tolist()

alpha[states.index([2,3,4])] = 1

ALPHA = alpha.tolist()

cvxpy.Problem(cvxpy.Maximize(x@r.T), [A@x == alpha.T, x >= 0]).solve()

X = x.value.tolist()

i = 0
while i < len(moves):
    j = states.index(moves[i][0])
    k = moves[i][1]
    m[j][k] = x.value[i]
    i+= 1

i = 0
while i < len(states):
    policy += [states[i],actions[max(m[i], key = m[i].get)]]
    i += 1

objective = cvxpy.Problem(cvxpy.Maximize(x@r.T), [A@x == alpha.T, x >= 0]).value

stuff = {
    "a": a,
    "r": R,
    "alpha": ALPHA,
    "x": X,
    "policy": policy,
    "objective": objective
}

with open(final_file, "w+") as file:
    json.dump(stuff, file)