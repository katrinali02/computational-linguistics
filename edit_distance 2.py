def del_cost():
    return 1

def ins_cost():
    return 1

def sub_cost(c1, c2):
    if c1 == c2: 
        return 0
    else:
        return 2

def min_edit_distance(source, target, do_print_chart=False):
    """Compare `source` and `target` strings and return their edit distance with
    Levenshtein costs, according to the algorithm given in SLP Ch. 2, Figure 2.17.

    Parameters
    ----------
    source : str
        The source string.
    target : str
        The target string.

    Returns
    -------
    int
        The edit distance between the two strings.
    """

    # >>> YOUR ANSWER HERE
    pass
    # >>> END YOUR ANSWER
        
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 3:
        w1 = sys.argv[1]
        w2 = sys.argv[2]
    else:
        w1 = 'intention'
        w2 = 'execution'
    print('edit distance between', repr(w1), 'and', repr(w2), 'is', min_edit_distance(w1, w2))


