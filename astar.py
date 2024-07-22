def h(n):
    H={'a':3,'b':4,'c':2,'d':6,'g':0,'s':5}
    return H[n]

def astar(graph,start,goal):
    openlist=[start]
    closedlist=set()
    g={start:0}
    parent={start:start}

    while(openlist):
        openlist.sort(key=lambda x:g[x]+h(x),reverse=True)
        n=openlist.pop()

        if n==goal:
            path=[]
            while(parent[n]!=n):
                path.append(n)
                n=parent[n]
            path.append(start)
            path.reverse()
            return path

        for(m,weight) in graph[n]:
            if m not in closedlist or m not in openlist:
                openlist.append(m)
                parent[m]=n
                g[m]=g[n]+weight

            else:
                if g[m]>g[n]+weight:
                    g[m]=g[n]+weight
                    parent[m]=n

                    if m in closedlist:
                        closedlist.remove(m)
                        openlist.append(m)

        closedlist.add(n)
    return None

graph={
    's':[('a',1),('g',10)],
    'a':[('b',2),('c',1)],
    'b':[('d',5)],
    'c':[('d',3),('g',4)],
    'd':[('g',2)]
}

optpath=astar(graph,'s','g')
print("Optimum path: ",optpath)
