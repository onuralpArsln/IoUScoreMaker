""" for a given matrix seearch for center of mass"""


def centerMassSearch(area,startx=0,starty=0)-> list:
    verLen=0
    horLen=0
    currentx=startx
    currenty=starty
    # horiziontal search
    while area[currenty][currentx]==1:
        horLen+=1
        currentx=currentx+1
        print(horLen , currentx)
    
   
    currentx=startx    # reset initial value
    horLen=horLen//2   # gets traversed length  finds mid point
    currentx+= horLen  # adds mid point to start val
    currenty=currenty+1 # moves to nest line (gets handy for recurision reasons)
    print( currentx)

    #vertical search


    while area[currenty][currentx]==1:
        verLen+=1
        currenty=currenty+1


    
    
    

    currenty=starty    # reset initial value
    verLen=verLen//2   # gets traversed length  finds mid point
    currenty+= verLen  # adds mid point to start val
    currentx=currentx+1 # moves to nest line (gets handy for recurision reasons)
    print( currenty)

    print(currentx,currenty)
    
    pass
        

def traverseMatris(matris):
    for i in matris[0]:
        if i ==1:
            pass 
            # call center search algo
            # then jump start your point


if __name__ == "__main__":
    example = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    
]
  
    centerMassSearch(area=example,startx=2,starty=2)