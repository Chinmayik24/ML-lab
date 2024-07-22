class TreeNode:
    def __init__(self,value,children=[]):
        self.value=value
        self.children=children
        self.alpha=float('-inf')
        self.beta=float('inf')

def minimax(node,depth,alpha,beta,max_p):
    global prune_count
    if depth==0 or not node.children:
        return node.value,[node.value]

    if max_p:
        maxval=float('-inf')
        maxpath=[]
        for childnode in node.children:
            childval,childpath=minimax(childnode,depth-1,alpha,beta,False)
            if childval>maxval:
                maxval=childval
                maxpath=[node.value]+childpath

            alpha=max(alpha,maxval)
            if alpha>=beta:
                prune_count+=len(childnode.children)+1
                break
        return maxval,maxpath

    else:
        minval=float('inf')
        minpath=[]
        for childnode in node.children:
            childval,childpath=minimax(childnode,depth-1,alpha,beta,True)
            if childval<minval:
                minval=childval
                minpath=[node.value]+childpath

            beta=min(beta,minval)
            if alpha>=beta:
                prune_count+=len(childnode.children)+1
                break
        return minval,minpath

gametree=tn(0,[tn(0,[tn(0,[tn(10),tn(9)]),tn(0,[tn(14),tn(18)])]),tn(0,[tn(0,[tn(5),tn(4)]),tn(0,[tn(50),tn(3)])])])

prune_count=0
optval,optpath=minimax(gametree,3,float('-inf'),float('inf'),True)

print(optval)
print(prune_count)
