from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def readData():
    excel = pd.read_excel (r'COMP3217CW2Input.xlsx', sheet_name = 'User & Task ID')
    name = excel['User & Task ID'].tolist()
    readytime = excel['Ready Time'].tolist()
    deadline = excel['Deadline'].tolist()
    maxkwh = excel['Maximum scheduled energy per hour'].tolist()
    demand = excel['Energy Demand'].tolist()
    alltasks = []
    task_names = []
    for i in range (len(readytime)):
        task = []
        task.append(readytime[i])
        task.append(deadline[i])
        task.append(maxkwh[i])
        task.append(demand[i])
        alltasks.append(task)
        task_names.append(name[i])
        
    #Reading Testing Data Output
    testDF = pd.read_csv('TestingDataOutput.txt', header=None)
    y_labels = testDF[24].tolist()
    testDF = testDF.drop(24, axis=1)
    x_data = testDF.values.tolist()
    
    return alltasks, task_names, x_data, y_labels

def createLPModel(tasks, task_names):

    task_vars = []
    n_list = []
    c = []
    eq = []
    
    model = LpProblem(name="scheduling-problem", sense=LpMinimize)
    

    for task in tasks:
        n_list.append(task[1] - task[0] + 1)

    for ind, task in enumerate(tasks):
        n = task[1] - task[0] + 1
        temp_list = []
        for i in range(task[0], task[1] + 1):
            x = LpVariable(name=task_names[ind]+'_'+str(i), lowBound=0, upBound=task[2])
            temp_list.append(x)
        task_vars.append(temp_list)
        
    #print (type(task_vars))
        
    for ind, task in enumerate(tasks):
        for var in task_vars[ind]:
            #print (int(var.name.split('_')[1]))
            price = price_list[int(var.name.split('_')[2])]
            #print (c)
            c.append(price * var)

    #print (c)
    model += lpSum(c)
            
    for ind, task in enumerate(tasks):
        temp_list = []
        for var in task_vars[ind]:
            temp_list.append(var)
        eq.append(temp_list)
        #temp_list.append(task[3])
        
        ##adding the constraints
        model += lpSum(temp_list) == task[3]
        
    return model

def plot(model, count):
    hours = [str(x) for x in range(0, 24)]
    pos = np.arange(len(hours))
    users = ['user1', 'user2', 'user3', 'user4', 'user5']
    color_list = ['midnightblue','mediumvioletred','mediumturquoise','gold','linen']
    plot_list = []
    to_plot = []
    for user in users:
        temp_list = []
        for hour in hours:
            hour_list_temp = []
            task_count = 0
            for var in model.variables():
                if user == var.name.split('_')[0] and str(hour) == var.name.split('_')[2]:
                    task_count += 1
                    #print('{} {} {} {}'.format(user, hour, var, var.value()))
                    hour_list_temp.append(var.value())
            temp_list.append(sum(hour_list_temp))
        plot_list.append(temp_list)

    plt.bar(pos,plot_list[0],color=color_list[0],edgecolor='black',bottom=0)
    plt.bar(pos,plot_list[1],color=color_list[1],edgecolor='black',bottom=np.array(plot_list[0]))
    plt.bar(pos,plot_list[2],color=color_list[2],edgecolor='black',bottom=np.array(plot_list[0])+np.array(plot_list[1]))
    plt.bar(pos,plot_list[3],color=color_list[3],edgecolor='black',bottom=np.array(plot_list[0])+np.array(plot_list[1])+np.array(plot_list[2]))
    plt.bar(pos,plot_list[4],color=color_list[4],edgecolor='black',bottom=np.array(plot_list[0])+np.array(plot_list[1])+np.array(plot_list[2])+np.array(plot_list[3]))
    
    plt.xticks(pos, hours)
    plt.xlabel('Hour')
    plt.ylabel('Energy Usage (kW)')
    plt.title('Energy Usage Per Hour For All Users')
    plt.legend(users,loc=0)
    #plt.show()
    plt.savefig('plots\\'+str(count)+'.png')
    plt.clf()

    return plot_list

tasks, task_names, x_data, y_labels = readData()

for ind, price_list in enumerate(x_data):
    if y_labels[ind] == 1:
        model = createLPModel(tasks, task_names)
        answer = model.solve()
        print(answer)
        plot(model,ind+1)