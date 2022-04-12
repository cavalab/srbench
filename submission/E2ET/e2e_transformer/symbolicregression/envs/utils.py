def zip_dic(lst):
    dico = {}
    for d in lst:
        for k in d:
            if k not in dico:
                dico[k] = []
            dico[k].append(d[k])
    for k in dico:
        if isinstance(dico[k][0], dict):
            dico[k] = zip_dic(dico[k])
    return dico


def unsqueeze_dic(dico):
    dico_copy = {}
    for d in dico:
        if isinstance(dico[d], dict):
            dico_copy[d] = unsqueeze_dic(dico[d])
        else:
            dico_copy[d] = [dico[d]]
    return dico_copy


def squeeze_dic(dico):
    dico_copy = {}
    for d in dico:
        if isinstance(dico[d], dict):
            dico_copy[d] = squeeze_dic(dico[d])
        else:
            dico_copy[d] = dico[d][0]
    return dico_copy


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def getSizeOfNestedList(listOfElem):
    """Get number of elements in a nested list"""
    count = 0
    # Iterate over the list
    for elem in listOfElem:
        # Check if type of element is list
        if type(elem) == list:
            # Again call this function to get the size of this element
            count += getSizeOfNestedList(elem)
        else:
            count += 1
    return count


class ZMQNotReady(Exception):
    pass


class ZMQNotReadySample:
    pass
