#BFS

def bfs(graph,start,goal,heuristic,path=[]):
    openlist=[(0,start)]
    closedlist=set()
    closedlist.add(start)

    while(openlist):
        openlist.sort(key=lambda x:heuristic[x[1]],reverse=True)
        cost,node=openlist.pop()
        path.append(node)

        if node==goal:
            return cost,path

        closedlist.add(node)
        for neighbour,neighbour_cost in graph[node]:
            if neighbour not in closedlist:
                openlist.append((cost+neighbour_cost,neighbour))
                closedlist.add(neighbour)
    return None

graph={
    'a':[('b',11),('c',14),('d',7)],
    'b':[('a',11),('e',15)],
    'c':[('a',14),('d',18),('f',10)],
    'd':[('a',7),('c',18),('f',25)],
    'e':[('b',15),('h',9)],
    'f':[('c',10),('d',25),('g',20)],
    'g':[],
    'h':[('e',9),('g',10)]
}

start='a'
goal='g'

heuristic={
    'a':20,'b':8,'c':10,'d':12,'e':10,'f':6,'g':0,'h':4
}

optval,optpath=bfs(graph,start,goal,heuristic)

print("Optimum value: ",optval)
print("Optimum path: ",optpath)
