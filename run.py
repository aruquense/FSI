# Search methods

import search

ab = search.GPSProblem('A', 'B', search.romania)
on = search.GPSProblem('O', 'N', search.romania)
lr = search.GPSProblem('L', 'R', search.romania)
hz = search.GPSProblem('H', 'Z', search.romania)
do = search.GPSProblem('D', 'O', search.romania)
lista=[ab,on,lr,hz,do]
while len(lista)>0:
    x=lista.pop()
    print "\n-----------------------------------------------------------------------"
    print "\nnivel-anchura, FIFO"
    print search.breadth_first_graph_search(x).path()
    print "\nprofundidad-altura, LIFO-pila-stack"
    print search.depth_first_graph_search(x).path()

    #print search.best_first_graph_search(ab,2).path()
    print "\nramificacion y acotacion"
    print search.ramificacionacotacion_first_graph_search(x).path()
    print "\nramificacion y acotacion con subestimacion"
    print search.ramificacionacotacionconsubestimacion_first_graph_search(x).path()

    #print search.iterative_deepening_search(ab).path()
    #print search.depth_limited_search(ab).path()

    #print search.astar_search(ab).path()

    # Result:
    # [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
    # [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
