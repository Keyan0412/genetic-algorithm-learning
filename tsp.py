# traveling salesman problem

import random
import math


def crossover(travel1, travel2):
    while True:
        start, end = random.randint(1, len(travel1)-3), random.randint(1, len(travel1)-2)
        if start == end: continue
        if start > end: 
            start, end = end, start
        break

    cross = travel1[start: end+1]
    child1 = travel2.copy()
    for i in cross:
        child1.remove(i)
    child1 += cross

    cross = travel2[start: end+1]
    child2 = travel1.copy()
    for i in cross:
        child2.remove(i)
    child2 += cross

    return child1, child2 


def mutation(travel):
    if random.random() > 0.2: return travel
    while True:
        start, end = random.randint(1, len(travel1)-3), random.randint(1, len(travel1)-2)
        if start == end: continue
        if start > end: 
            start, end = end, start
        break

    mut = travel[start: end+1]
    random.shuffle(mut)
    return travel[0: start] + mut + travel[end+1: ]


def cal_distance(travel, locations):
    current_loc = locations[travel[0]]
    distance = 0
    for i in travel[1: ]:
        next_loc = locations[i]
        distance += math.sqrt((current_loc[0] - next_loc[0])**2 + (current_loc[1] - next_loc[1])**2)
        current_loc = next_loc
    return distance


def min_selection(travels):
    result = []
    first = cal_distance(travels[0])
    first_i = 0
    second = cal_distance(travels[1])
    second_i = 1
    if second < first:
        first, second = second, first
        first_i, second_i = second_i, first_i
    for i, travel in enumerate(travels[2: ]):
        dis = cal_distance(travel)
        if dis < first:
            second = first
            first = dis
            second_i = first_i
            first_i = i
        elif dis < second:
            second = dis
            second_i = i
    return first_i, second_i


num_loc = 10
locations = []
for i in range(num_loc):
    loc = (random.random(), random.random())
    locations.append(loc)

num = 10
travels = []
for i in range(num):
    travels.append(random.shuffle([i for i in range(10)]))

distances = []
for travel in travels:
    distances.append(cal_distance(travel))
print("original minimum distance: ", min(distances))

for i in range(10):
    parent1_i, parent2_i = min_selection(travels)
    parent1 = travel[parent1_i]
    parent2 = travel[parent2_i]
    child1, child2 = crossover(parent1, parent2)
    child1 = mutation(child1)
    child2 = mutation(child2)
    travels.append(child1)
    travels.append(child2)
    

