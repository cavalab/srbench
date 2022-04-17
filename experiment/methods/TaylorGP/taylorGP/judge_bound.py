import copy
import math
def add( bound1, bound2):
    return [bound1[0] + bound2[0], bound1[1] + bound2[1]]
def sub( bound1, bound2):
    return [bound1[0] - bound2[1], bound1[1] - bound2[0]]
def mul( bound1, bound2):
    max_ = max(bound1[0] * bound2[0],bound1[0] * bound2[1],bound1[1] * bound2[0], bound1[1] * bound2[1])
    min_ = min(bound1[0] * bound2[0], bound1[0] * bound2[1], bound1[1] * bound2[0], bound1[1] * bound2[1])
    return [min_,max_]
def div( bound1, bound2):
    if bound1[0] > 0 and bound2[0]>0:
        return [1e-10,1e+10]
    else: return [-1e+10,1e+10]
def sin( bound1, bound2):
    return [-1, 1]
def cos( bound1, bound2):
    return [-1, 1]
def log( bound1, bound2):
    low_bound = max(1e-10, bound1[0])
    up_bound = max(1e-10, bound1[1])
    return [math.log(low_bound), math.log(up_bound)]
def exp( bound1, bound2):
    low = min(bound1[0],100)
    up = min(bound1[1],100)
    return [math.exp(low), math.exp(up)]
def sqrt( bound1, bound2):
    down = max(1e-10,bound1[0])
    return [math.sqrt(down), math.sqrt(bound1[1])]
def cal_spacebound(function_set=[ 'add','sub','mul','div','sin','cos', 'log' ,'exp','sqrt'],n_features_=1,var_bound=[-5,5],const_flag=True):
    space = {}
    log_bound, exp_bound, sin_bound, const_bound= [-1e+10, 1e+10], [1e-10, 1e+10], [-1, 1], [-1, 1]
    _bound_map = {'add': log_bound,
                  'sub': log_bound,
                  'mul': log_bound,
                  'div': log_bound,
                  'sin': sin_bound,
                  'cos': sin_bound,
                  'log': log_bound,
                  'exp': exp_bound,
                  'sqrt': exp_bound,
                  'const': const_bound}
    for i in range(len(var_bound)//2):
        _bound_map.update({i:var_bound[2*i:2*i+2]})

    set = copy.deepcopy(function_set)
    for i in range(n_features_):
        set.append(i)
    if const_flag:
        opNum = len(function_set) + n_features_ + 1
        set.append('const')
    else:
        opNum = len(function_set) + n_features_

    for op1 in range(opNum):
        for op2 in range(opNum):
            for op3 in range(opNum):
                if op1 < len(function_set):
                    temp_bound = eval(set[op1] + '(_bound_map[set[op2]],_bound_map[set[op3]])')
                    space[set[op1] +' '+ str(set[op2])+' ' + str(set[op3])] = temp_bound
                elif op1 < len(function_set) + n_features_:
                    space[str(set[op1]) + ' '] = _bound_map[set[op1]]
    return space
def select_space(space_bound,low_bound=0,high_bound=1,var_bound=[-1,1]):
    selected_space = []
    for key,value in space_bound.items():#key = space,value = bound
        if value[0] <= (low_bound+0.1) and value[1] >= (high_bound-0.1) :
            selected_space.append(key)
    print(low_bound,high_bound,'total selected space:',len(selected_space))
    return selected_space

if __name__ == '__main__':
    select_space(cal_spacebound(var_bound=[-5,5]))

