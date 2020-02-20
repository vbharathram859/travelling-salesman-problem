import numpy as np
import matplotlib.pyplot as plt
N=150
M=1
plt.axis('off')
initial_dist = 1

def gen_cities():
    mat = np.zeros((N, 2))  # stores cities x and y coordinates, where (mat[i][0], mat[i][1]) is the ordered pair for the city
    dist = {}  # dist[(a, b)] is the distance between city a and city b
    for i in range(N):  # randomly generate cities
        mat[i][0] = np.random.random()*M
        mat[i][1] = np.random.random()*M
    for i in range(N):
        for j in range(N):
            dist[i, j] = np.sqrt(pow(mat[i][0]-mat[j][0], 2)+pow(mat[i][1]-mat[j][1], 2))  # use distance formula to find the distance between city i and city j
    return mat, dist

def cost(sol, dist):
    count = 0
    last = sol[0]  # last city we went to
    for i in range(1, len(sol)):  # loop through path and add the distance of every path we take
        cur = sol[i]
        count += dist[(last, cur)]
        last = cur
    count+=dist[(sol[0], last)]  # complete the loop, since we end at the place we started

    return count/initial_dist  # heuristic is percentage of initial distance

def neighbor(sol):  # 2-opt swap, returns random neighbor
    new_sol = sol.copy()
    rand1 = np.random.randint(N)
    rand2 = np.random.randint(N)

    if rand2 < rand1:  # make sure rand1 is smaller
        rand1, rand2 = rand2, rand1

    new_sol[rand1:rand2 + 1] = new_sol[rand1:rand2 + 1][::-1]  # reverse this section of the list

    return new_sol

def probability(old_cost, new_cost, T):
    return np.exp(-(new_cost-old_cost)/T)

def display(sol, mat):
    x = []  # all the x coordinates of the cities
    y = []  # all the y coordinates of the cities
    for i in range(N):
        x.append(mat[i][0])
        y.append(mat[i][1])

    plt.scatter(x, y)  # plot all of the cities as points
    tot = []  # stores all of the paths using the ordered pairs
    last = sol[0]
    for item in sol[1:]:
        tot.append((mat[last], mat[item]))
        last = item
    tot.append((mat[last], mat[sol[0]]))

    for i in range(N):
        plt.plot((tot[i][0][0], tot[i][1][0]), (tot[i][0][1], tot[i][1][1]))  # plot the lines connecting the paths
    plt.show()


def anneal(sol, dist, mat):
    old_cost = cost(sol, dist)
    T = 1
    T_min = 0.00001
    alpha = 0.999
    costs = []  # use this to graph change of heuristic over time
    while T > T_min:
        for i in range(N**2):
            new_sol = neighbor(sol)
            new_cost = cost(new_sol, dist)
            if new_cost < old_cost:  # then definitely make this move
                sol = new_sol  # update solution
                old_cost = new_cost  # update cost
                costs.append(new_cost)
                break
            ap = probability(old_cost, new_cost, T)  # ap is the acceptance probability
            if ap > np.random.random():  # accept this move with probability ap
                sol = new_sol
                old_cost = new_cost
                costs.append(new_cost)
                break
        T = T * alpha
    display(sol, mat)  # display the final solution
#     plt.plot(costs)
#     plt.show()
    return sol, old_cost

def local_search(sol, dist, c=0):
    old_cost = cost(sol, dist)
    for i in range(N**3):
        new_sol = neighbor(sol)
        new_cost = cost(new_sol, dist)
        if new_cost < old_cost:
            return local_search(new_sol, dist, c)
    return sol, old_cost

def random_sol(dist):
    global initial_dist  # initial_dist stores cost of initial random solution
    order = np.arange(N)
    np.random.shuffle(order)  # create random solution
    initial_dist = cost(order, dist)
    return order

def main():
    mat, dist = gen_cities()
    sol = random_sol(dist)

    a, b = anneal(sol, dist, mat)
    print(b*initial_dist)  # b * initial_dist gives us the actual cost of the end result

main()
