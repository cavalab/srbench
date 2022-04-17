def readTaylorFeatures(path,fileNum):
    inputName =path+'\gplearn_7_' + str(fileNum) +  '.out'
    outputName = 'GECCO\\1TaylorFeature.out'
    file = open(inputName,"r", encoding='utf-8', errors='ignore')
    fitness_write = open(outputName, "a+", encoding='utf-8', errors='ignore')
    fitness_write.write(str(fileNum) + ' ')
    lines = file.readlines()
    low_order_flag = True
    for line in lines:
        if '具有乘法可分性' in line:
            fitness_write.write('*')
            break
        if '具有加法可分性' in line:
            fitness_write.write('+')
            break

    print(fileNum,'over')
    fitness_write.close()

if __name__ == '__main__':
    list = "[[[-1.6094379124341003, 0.6931471805599453], [0.2, 2.0]], \
     -272.71562274955952 * x0 ** 6 + 250.29936538667023 * x0 ** 5 - 176.23121295760512 * x0 ** 4 + 94.798489254606202 * x0 ** 3 - 38.946532911339539 * x0 ** 2 + 12.998571293082988 * x0 - 3.1798347707773156,\
     14125817.008432716, -3.179834770777316, -1, 1]"
    for j in range(0,45):
        readTaylorFeatures(path = 'GECCO',fileNum=j)
    for j in range(0, 100):
        readTaylorFeatures(path='AIFeynman', fileNum=j)
    for j in range(0, 17):
        readTaylorFeatures(path='ML',fileNum=j)