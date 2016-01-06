def list_del_indices(mylist,indices):
    for index in sorted(indices, reverse=True):
        del mylist[index]
    return mylist
