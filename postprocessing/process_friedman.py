import pdb

filename = 'friedman_r1.txt'
#nice = {"":"GTB",
#        "RandomForestClassifier":"RF", 
#        "SVC":"SVC",
#        "ExtraTreesClassifier":"ERF",
#        "SGDClassifier":"SGD",
#        "DecisionTreeClassifier":"DT",
#        "LogisticRegression":"LR",
#        "KNeighborsClassifier":"KNN",
#        "AdaBoostClassifier":"AB",
#        "PassiveAggressiveClassifier":"PAC",
#        "BernoulliNB":"BNB",
#        "GaussianNB":"GNB",
#        "MultinomialNB":"MNB"}
m1 = []

models = ["eplex-1m","xgboost","gradboost","mlp","rf","eplex","mrgp","kernel-ridge","adaboost","afp","lasso-lars","linear-svr","linear-regression","sgd-regression","gsgp"]

m1 = []
m2 = []
pval = {}
with open(filename,'r') as f:
    for line in f:
        ms = ' '.join(line.split(' ')[:3])
        # print('ms:', ms,end='\t')
        #pdb.set_trace()
        #m1.append(ms.split(' - ')[0])
        #m2.append(ms.split(' - ')[1])
        p = float(line.split(' ')[-1].split('\n')[0])
        # print('pval:',p)
        pval[ms] = p

# for k,v in sorted(pval.items()):
#     print(k,':',v)
# print results to table
print('\n\n')
print('&',' & '.join([m for m in models[:-1]]),'\\\\')
for r in models[1:]:
    print(r,end='\t')
    for c in models[:-1]:
        if c!=r:
            pre=''
            key = r +' - '+c 
            if key not in pval:  
                print('& -',end='')
            #if key not in pval:
            #    print(key,'not in pval')
            #    raise ValueError
                #print('& -',end='')
            else:
                if pval[key] < 0.05:
                    pre = '\\textbf'
                else:
                    pre = ''
                print('&',pre,'{','{:1.1g}'.format(pval[key]),'}',end='')
        else:
            print('& -',end='')
           
    print('\\\\')
