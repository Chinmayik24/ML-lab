class tn:
    def __init__(self,val,children=[]):
        self.val=val
        self.children=children

def minimax(node,depth,maximizing_p):
    if depth==0 or not node.children:
        return node.val,[node.val]

    if maximizing_p:
        maxval=float('-inf')
        maxpath=[]
        for childnode in node.children:
            childval,childpath=minimax(childnode,depth-1,False)
            if childval>maxval:
                maxval=childval
                maxpath=[node.val]+childpath
        return maxval,maxpath

    else:
        minval=float('inf')
        minpath=[]
        for childnode in node.children:
            childval,childpath=minimax(childnode,depth-1,True)
            if childval<minval:
                minval=childval
                minpath=[node.val]+childpath
        return minval,minpath

Gametree= tn(0,[
    tn(1,[tn(3,[tn(3),tn(12)]),tn(4,[tn(6),tn(9)])]),
    tn(2,[tn(5,[tn(1),tn(10)]),tn(6,[tn(2),tn(7)])])
])

optval,optpath=minimax(Gametree,3,True)

print(optval)
print(optpath)
