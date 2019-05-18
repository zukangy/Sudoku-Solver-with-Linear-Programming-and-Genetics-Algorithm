import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as scs # sparse matrix construction
import scipy.linalg as scl # linear algebra algorithms
import scipy.optimize as sco # for minimization use
import matplotlib.pylab as plt # for visualization
import random as rd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


# read the data in small1.csv, it only has 24 problems.

# we can turn it into a numpy array by the following

#GA

def evaluate1(Lst):
    score = 0
    NumberSet1 = list(range(1, 10))
    Row = []
    Colunm = []
    for i in range(9):
        newR = []
        newC = []
        for j in range(9):
            newR.append(Lst[i+j*9])
            newC.append(Lst[i*9 + j])
        Row.append(newR)
        Colunm.append(newC)

    for m in range(9):
        for n in range(9):
            if Row[m].count(str(NumberSet1[n])) > 1:
                score += Row[m].count(str(NumberSet1[n])) - 1
            if Colunm[m].count(str(NumberSet1[n])) > 1:
                score += Colunm[m].count(str(NumberSet1[n])) - 1
    return score

# select the best population from ordered population

def select(Pop,Pop_score):
    P = Pop
    S = Pop_score
    for g in range(len(Pop)-21):
        Max = max(S)
        Max_Index = S.index(Max)
        del P[Max_Index]
        del S[Max_Index]
    return [P,S]

#order the population

def order(Pop,Pop_socre):
    P = Pop
    S = Pop_socre
    New_pop = []
    New_score = []
    for i in range(len(P)):
        index1 = S.index(min(S))
        New_pop.append(P[index1])
        New_score.append(S[index1])
        del P[index1]
        del S[index1]
    return [New_pop,New_score]

# create gene of a single population
def gene(singlePop):
    Gene = []

    for m in range(3):
        for n in range(3):

            New_Gene = []
            for i in range(3):
                for j in range(3):
                    New_Gene.append(singlePop[m*27 + i*9+ n*3 + j])
            Gene.append(New_Gene)

    return Gene


# transfer the gene to a population
def genTonorm(gen):
    norm = []

    for j in range(3):
        for n in range(3):
            for m in range(3):

                    for i in range(3):
                            norm.append(gen[m+j*3][n*3+i])

    return norm

# random crossover combination.

def combin(best,single1,single2):
    Gene0 = gene(best)
    Gene1 = gene(single1)
    Gene2 = gene(single2)
    New_gene = []
    NumberSet = list(range(1,10))
    times1 = rd.randint(1,7)
    times2 = rd.randint(1,8-times1)
    g1 = rd.sample(NumberSet,times1)
    g2 = rd.sample(list(set(NumberSet).difference(set(g1))),times2)
    g1.sort()
    for i in range(9):
        if NumberSet[i] in g1:
            New_gene.append(Gene1[i])
        elif NumberSet[i] in g2:
            New_gene.append(Gene2[i])
        else:
            New_gene.append(Gene0[i])

    New_gene = genTonorm(New_gene)

    return New_gene

# adaptive random mutation

def mutation(single,fit1,ident,AQ,BL):

    List = single
    Associated_quiz = AQ
    blocklist = BL

    if fit1 > 10:
        swap = 1
    elif fit1 > 3:
        swap = rd.randint(1,5)
    else:

        #if evaluate(single) > 4:
            #swap = 10
        if ident <5:
            swap = rd.randint(1,8)
        else:
            swap = rd.randint(1,3)

    k = 0
    while k<swap:

        posi = rd.randint(0,80)

        while Associated_quiz[posi] == '1':
            posi = rd.randint(0, 80)

        for i in blocklist:
            if posi in i:
                posi2 = rd.sample(i,1)[0]
                while posi == posi2:
                    posi2 = rd.sample(i, 1)[0]

        s1 = List[posi]
        s2 = List[posi2]

        List[posi2] = s1
        List[posi] = s2

        k += 1

    return List

'''
def modify(single):
    List = single
    Ge = gene(single)
    repind = repeat_index(single)
    for i in range(len(blocklist)):
        if len(set(repind).intersection(set(blocklist[i]))) > 1:
            print(set(repind).intersection(set(blocklist[i])))
            New = rd.shuffle(Ge[i])
            for j in range(81):
                if (int(Associated_quiz[j]) == 1) and j in blocklist[i]:
                    id1 = List[j]
                    id2 = New.index()
                    v1 = blockvalue[i][j]
                    v2 = New[j]

                    New[id1] = v2
                    New[id2] = v1
            Ge[i] = New

    List = genTonorm(Ge)

    return List
'''
# Create Next generation, and include order, select, mutation and crossover combination
# Also include 3 children method to increase the ability to find local optimal point.


def generate(Pop,Pop_score,AQ,BL):
    New_gen = [Pop[0]]
    New_score = [Pop_score[0]+0.25]

    for i in range(20):
        identifier = len(New_gen)
        if identifier < 5:
            Bound = 5
        elif identifier < 10:
            Bound = 11
        elif identifier < 15:
            Bound = 15
        else:
            Bound = 20

        X1 = rd.randint(1,Bound)
        X2 = rd.randint(1,Bound)
        X3 = rd.randint(1,Bound)
        X4 = rd.randint(1,Bound)

        while X1 == X2:
            X2 = rd.randint(1, Bound)
        while X1 == X3 or X2 == X3:
            X3 = rd.randint(1, Bound)
        while X1 == X4 or X2 == X4 or X3 == X4:
            X4 = rd.randint(1, Bound)

        ng = combin(Pop[0],Pop[X1],Pop[X2])
        ng = mutation(ng,New_score[0],identifier,AQ,BL)
        New_gen.append(ng)
        New_score.append(evaluate1(ng))


        #if Population_Score[0] < 3 and X1 <= 5:
            #ng = combin(Pop[0],Pop[X1],Pop[X3])
            #ng = mutation(ng,New_score[0],identifier)
            #New_gen.append(ng)
            #New_score.append(evaluate(ng))

            #ng = combin(Pop[0],Pop[X1],Pop[X4])
            #ng = mutation(ng,New_score[0],identifier)
            #New_gen.append(ng)
            #New_score.append(evaluate(ng))


    New_gen.extend(Pop[1:11])
    New_score.extend(Pop_score[1:11])



    n1 = order(New_gen,New_score)
    New_gen = n1[0]
    New_score = n1[1]

    n2 = select(New_gen,New_score)
    New_gen = n2[0]
    New_score = n2[1]

    return [New_gen,New_score]

count = 0
Q = 0
F = 0

# create Associated form to localize the unchangable point

def solve(Quiz,LP1):

    LP = LP1

    Restart = False

    quiz = Quiz


    Associated_quiz = ''

    for j in quiz:
        if j != '0':
            Associated_quiz = Associated_quiz + '1'
        else:
            Associated_quiz = Associated_quiz + '0'

    blocklist = []
    blockvalue = []

    for m in range(3):
        for n in range(3):

            New_block = []
            New_value = []
            for k in range(3):
                for j in range(3):
                    if Associated_quiz[m * 27 + k * 9 + n * 3 + j] != '1':
                        New_block.append(m * 27 + k * 9 + n * 3 + j)
                        New_value.append(quiz[m * 27 + k * 9 + n * 3 + j])
            blocklist.append(New_block)
            blockvalue.append(New_value)

    # Initialize

    Population = []
    Population_Score = []

    for t in range(41):

        NumberSet = list(range(1,10))

        NewChrom = list(quiz)

        for m in range(3):
            for n in range (3):

                NumberSet = list(range(1, 10))

                for k in range(m*3,m*3+3):
                    for j in range(n*3,n*3+3):
                        if int(Associated_quiz[k*9+j]) != 0:
                            NumberSet.remove(int(quiz[k*9+j]))

                for k in range(m*3,m*3+3):
                    for j in range(n*3,n*3+3):
                        if int(Associated_quiz[k*9+j]) == 0:
                            ran = rd.sample(NumberSet,1)[0]
                            NewChrom[k*9+j] = str(ran)
                            NumberSet.remove(int(ran))

        Population.append(NewChrom)
        Population_Score.append(evaluate1(NewChrom))

#

    New1 = order(Population,Population_Score)
    Population = New1[0]
    Population_Score = New1[1]

    New1 = select(Population,Population_Score)
    Population = New1[0]
    Population_Score = New1[1]

    Population[0] = LP
    Population_Score[0] = evaluate1(LP)

    G = 0

    last = Population_Score[0]

# iteration

    while Population_Score[0] != 0:
        New1 = generate(Population,Population_Score,Associated_quiz,blocklist)
        Population = New1[0]
        Population_Score = New1[1]

        G+= 1

        if G % 100 == 0:

            for k in range(len(Population_Score)):
                Population_Score[k] = evaluate1(Population[k])
            New1 = order(Population,Population_Score)
            Population = New1[0]
            Population_Score = New1[1]

# GA restart method
        if G % 1000 ==0:
            if Population_Score[0] > 4:
                Restart = True
            elif last > Population_Score[0]:
                last = Population_Score[0]
            else:
                Restart = True




        if Restart:

            Restart = False

            Population = []
            Population_Score = []

            for t in range(41):

                NumberSet = list(range(1, 10))

                NewChrom = list(quiz)

                for m in range(3):
                    for n in range(3):

                        NumberSet = list(range(1, 10))

                        for k in range(m * 3, m * 3 + 3):
                            for j in range(n * 3, n * 3 + 3):
                                if int(Associated_quiz[k * 9 + j]) != 0:
                                    NumberSet.remove(int(quiz[k * 9 + j]))

                        for k in range(m * 3, m * 3 + 3):
                            for j in range(n * 3, n * 3 + 3):
                                if int(Associated_quiz[k * 9 + j]) == 0:
                                    ran = rd.sample(NumberSet, 1)[0]
                                    NewChrom[k * 9 + j] = str(ran)
                                    NumberSet.remove(int(ran))

                Population.append(NewChrom)
                Population_Score.append(evaluate1(NewChrom))


            New1 = order(Population, Population_Score)
            Population = New1[0]
            Population_Score = New1[1]

            New1 = select(Population, Population_Score)
            Population = New1[0]
            Population_Score = New1[1]

            Population[0] = LP
            Population_Score[0] = evaluate1(LP)
            last = Population_Score[0]



# if the problem is too hard, just give up.

        if G == 10000:
            print('cannot solve this problem')
            break



    return Population[0]





#








# In the following, the fixed_constraints are constructed from the board directly.
# This part only needs to be constructed once. The output has been returned as a sparse matrix for efficiency.

def fixed_constraints(N=9):
    rowC = np.zeros(N)
    rowC[0] = 1
    rowR = np.zeros(N)
    rowR[0] = 1
    row = scl.toeplitz(rowC, rowR)
    ROW = np.kron(row, np.kron(np.ones((1, N)), np.eye(N)))

    colR = np.kron(np.ones((1, N)), rowC)
    col = scl.toeplitz(rowC, colR)
    COL = np.kron(col, np.eye(N))

    M = int(np.sqrt(N))
    boxC = np.zeros(M)
    boxC[0] = 1
    boxR = np.kron(np.ones((1, M)), boxC)
    box = scl.toeplitz(boxC, boxR)
    box = np.kron(np.eye(M), box)
    BOX = np.kron(box, np.block([np.eye(N), np.eye(N), np.eye(N)]))

    cell = np.eye(N ** 2)
    CELL = np.kron(cell, np.ones((1, N)))

    return scs.csr_matrix(np.block([[ROW], [COL], [BOX], [CELL]]))

A0 = fixed_constraints()

print(A0[0][0])

plt.spy(A0, markersize=0.2)



# For the constraint from clues, we extract the nonzeros from the quiz string.
def clue_constraint(input_quiz, N=9):
    m = np.reshape([int(c) for c in input_quiz], (N, N))
    r, c = np.where(m.T)
    v = np.array([m[c[d], r[d]] for d in range(len(r))])

    table = N * c + r
    table = np.block([[table], [v - 1]])

    # it is faster to use lil_matrix when changing the sparse structure.
    CLUE = scs.lil_matrix((len(table.T), N ** 3))
    for i in range(len(table.T)):
        CLUE[i, table[0, i] * N + table[1, i]] = 1
    # change back to csr_matrix.
    CLUE = CLUE.tocsr()

    return CLUE

# get the constraint matrix from clue.


# Formulate the matrix A and vector B (B is all ones).
A = scs.vstack((A0,A1))
B = np.ones((np.size(A, 0)))


def evaluate(Lst):
    score = 0
    NumberSet = list(range(1, 10))
    Newlist = np.reshape(Lst,(9,9))

    NewCol = []
    for i in range(9):
        Col = []
        for j in range(9):
            Col.append(Newlist[i][j])
        NewCol.append(Col)

    for i in NumberSet:
        for j in NewCol:
            if i not in j:
                score += 1

    NewRow = []
    for i in range(9):
        Row = []
        for j in range(9):
            Row.append(Newlist[j][i])
        NewCol.append(Row)

    for i in NumberSet:
        for j in NewRow:
            if i not in j:
                score += 1




    return score

def gene(singlePop):
    Gene = []

    for m in range(3):
        for n in range(3):

            New_Gene = []
            for i in range(3):
                for j in range(3):
                    New_Gene.append(singlePop[m*27 + i*9+ n*3 + j])
            Gene.append(New_Gene)

    return Gene

def genTonorm(gen):
    norm = []

    for j in range(3):
        for n in range(3):
            for m in range(3):

                    for i in range(3):
                            norm.append(gen[m+j*3][n*3+i])

    return norm

def modify(single):
    G0 = gene(single)
    Number = list(range(1,10))

    for i in range(9):
        for j in range(9):
            if G0[i].count(G0[i][j]) > 1:
                G0[i][j] = list(set(Number).difference(set(G0[i])))[0]

    Modi = genTonorm(G0)

    return Modi


import time

# We test the following algoritm on small data set.


corr_cnt = 0
start = time.time()

random_seed = 42
np.random.seed(random_seed)


#for i in range(len(samples)):
def solver(quiz):
    corr_cnt = 0
    i = 0
    quiz1 = [str(x) for x in quiz]
    A0 = fixed_constraints()
    A1 = clue_constraint(quiz1)

    # Formulate the matrix A and vector B (B is all ones).
    A = scs.vstack((A0, A1))
    A = A.toarray()
    B = np.ones(A.shape[0])

    # Because rank defficiency. We need to extract effective rank.
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    K = np.sum(s > 1e-12)
    S = np.block([np.diag(s[:K]), np.zeros((K, A.shape[0] - K))])
    A = S @ vh
    B = u.T @ B
    B = B[:K]

    c = np.block([np.ones(A.shape[1]), np.ones(A.shape[1])])
    G = np.block([[-np.eye(A.shape[1]), np.zeros((A.shape[1], A.shape[1]))], \
                  [np.zeros((A.shape[1], A.shape[1])), -np.eye(A.shape[1])]])
    h = np.zeros(A.shape[1] * 2)
    H = np.block([A, -A])
    b = B

    ret = sco.linprog(c, G, h, H, b, method='interior-point', options={'tol': 1e-6})
    x = ret.x[:A.shape[1]] - ret.x[A.shape[1]:]

    z = np.reshape(x, (81, 9))

    LPansw = np.array([np.argmax(d) + 1 for d in z])

    if evaluate(modify(list(LPansw))) != 0:
    #if np.linalg.norm(np.reshape(np.array([np.argmax(d) + 1 for d in z]), (9, 9)) \
                      #- np.reshape([int(c) for c in solu], (9, 9)), np.inf) > 0:
        fair = np.array([np.argmax(d) + 1 for d in z])
        fair = modify(list(fair))
        for i in range(81):
            fair[i] = str(fair[i])
        Solution = solve(quiz1,fair)
        print('GA')
        return "".join(Solution)
    else:
        # print("CORRECT")
        print('LP')
        Lptransferans = [str(x) for x in LPansw]

        return "".join(Lptransferans)
