#-*- coding:utf-8 -*-
'''
凸包问题是指在n个点中，寻找一个凸多边形，使所有的点在凸多边形的边界或者内部。这是一个很有意思的计算机图形学问题，一种常用的解法是Graham扫描法，运行时间为O(nlgn)。
维基百科地址：https://en.wikipedia.org/wiki/Graham_scan
笔者拿processing语言对整个算法过程进行了模拟，上动态图：
![processing语言模拟Graham扫描法]("模拟Graham扫描法")
注意processing拿左上为(0,0)原点，与一般数学的原点位置不同。
从动态图上可以看出整个算法分三个部分：
**1、寻找y轴最小的点，如果y轴位置是相同的，那个找x轴位置最小的，称之为基准点。**
**2、计算1中找到基准点与其他点的极角（即过此2点的直线与x轴正方向的夹角，代码中以弧度表示），将这些点按极角的大小正序排列。**
**3、进行基准点与2中点的连线迭代，对新连线的点计算其是否符合凸多边形的定义，如果不满足舍弃此点。判断的方法是计算三点组成线段的叉乘，值为正表示满足条件。**
叉乘维基百科地址：https://zh.wikipedia.org/zh-cn/%E5%90%91%E9%87%8F%E7%A7%AF
'''

import sys
##sys.path.append(".")
##sys.path.append("..")
##sys.path.append("../..")
import tensorflow as tf  
import math
import time
import random
import matplotlib.pyplot as plt 
import numpy as np
import datetime

def add_layer(inputs,in_size,out_size,activation_function=None): #activation_function=None线性函数  
    Weights = tf.Variable(tf.random_normal([in_size,out_size])) #Weight中都是随机变量  
    biases = tf.Variable(tf.zeros([1,out_size])+0.1) #biases推荐初始值不为0  
    Wx_plus_b = tf.matmul(inputs,Weights)+biases #inputs*Weight+biases  
    if activation_function is None:  
        outputs = Wx_plus_b  
    else:  
        outputs = activation_function(Wx_plus_b)  
    return outputs  


#获取基准点的下标
def get_leftbottompoint(p):
    k = 0
    for i in range(1, len(p)):
        if p[i][1] < p[k][1] or (p[i][1] == p[k][1] and p[i][0] < p[k][0]):
            k = i
    return k

#叉乘计算方法
def multiply(p1, p2, p0):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])

#获取极角，通过求反正切得出，考虑pi / 2的情况
def get_arc(p1, p0):
    # 兼容sort_points_tan的考虑
    if (p1[0] - p0[0]) == 0:
        if ((p1[1] - p0[1])) == 0:
            return -1;
        else:
            return math.pi / 2
    tan = float((p1[1] - p0[1])) / float((p1[0] - p0[0]))
    arc = math.atan(tan)
    if arc >= 0:
        return arc
    else:
        return math.pi + arc
    return -1;

#对极角进行排序
def sort_points_tan(p, k):
    p2 = []
    for i in range(0, len(p)):
        p2.append({"index": i, "arc": get_arc(p[i], p[k])})
    p2.sort(key=lambda k: (k.get('arc', 0)))
    p_out = []
    for i in range(0, len(p2)):
        p_out.append(p[p2[i]["index"]])
    return p_out

def convex_hull(p):

    p=list(set(p))

    k = get_leftbottompoint(p)
    p_sort = sort_points_tan(p, k)
    p_result = [None] * len(p_sort)
    p_result[0] = p_sort[0]
    p_result[1] = p_sort[1]
    p_result[2] = p_sort[2]
    top = 2
    for i in range(3, len(p_sort)):
        #叉乘为正则符合条件
        while (top >= 1 and multiply(p_sort[i], p_result[top], p_result[top - 1]) > 0):
            top -= 1
        top += 1
        p_result[top] = p_sort[i]

    for i in range(len(p_result) - 1, -1, -1):
        if p_result[i] == None:
            p_result.pop()

    return p_result#测试

def millis(t1, t2):
    micros = (t2 - t1).microseconds
    print("micros: ",micros)
    delta = micros/1000
    return delta


if __name__ == '__main__':
    pass
    t1=datetime.datetime.now()
    print(t1.strftime('%H:%M:%S.%f'))
    test_data1 = []
    test_data2 = []
    test_data3 = []
    test_data4 = []
    test_data5 = []
    test_data6 = []
    train_data = []  #总的训练数据
    test_data = []  #总的测试数据
    train_data1 = []
    train_data2 = []
    train_data3 = []
    inpt       = []
    oupt       = []
    X=np.loadtxt("Iris.txt",delimiter=",",skiprows=0)
    X1=X[:100]
    np.random.shuffle(X1)
    X2=X[100:150]
    np.random.shuffle(X2)
    X3=X[150:200]
    np.random.shuffle(X3)
    #for i in range(50):
    #    test_data.append((X3[i,0],X3[i,1]))
    for r in X1[:60]:
         test_data1.append((r[0],r[1]))
    for r in X2[0:30]:
         test_data2.append((r[0],r[1]))
    for r in X3[0:30]:
         test_data3.append((r[0],r[1]))
    
    for r in X1[60:]:
         test_data4.append((r[0],r[1]))
    for r in X2[30:]:
         test_data5.append((r[0],r[1]))
    for r in X3[30:]:
         test_data6.append((r[0],r[1]))
    
    b=np.array(test_data3)
    
    print("test_data1长度："+str(len(test_data1)))
    print("test_data2长度："+str(len(test_data2)))
    print("test_data3长度："+str(len(test_data3)))
    #plt.scatter(b[:,0], b[:,1], marker = 'x',color = 'green', s = 40 ,label = 'First')
    
    #a=np.zeros((X.shape[0],2))

    #b=np.zeros((X.shape[0],2))
    
    #依次获得三个分类的凸包络
    result = convex_hull(test_data1)
     
    for r in result:
         train_data.append((r[0],r[1]))  
         train_data1.append((r[0],r[1]))
         inpt.append((r[0],r[1])) 
         oupt.append((1,1)) 
    result = convex_hull(test_data2)
    for r in result:
         train_data.append((r[0],r[1])) 
         train_data2.append((r[0],r[1]))
         inpt.append((r[0],r[1])) 
         oupt.append((0,1)) 
    result = convex_hull(test_data3)
    for r in result:
         train_data.append((r[0],r[1])) 
         train_data3.append((r[0],r[1]))
         inpt.append((r[0],r[1])) 
         oupt.append((1,0)) 

    np.random.shuffle(X)  #随机化
    #X1=X[:120]            #训练集
    #X2=X[120:]            #测试集
    x_data =inpt
    y_data =oupt
    xs = tf.placeholder(tf.float32,[None,2])
    ys = tf.placeholder(tf.float32,[None,2])	
	#三层神经，输入层（2个神经元），隐藏层（10神经元），输出层（2个神经元）  
    l1 = add_layer(xs,2,19,activation_function=tf.nn.relu) #输入层
    l2 = add_layer(l1,19,22,activation_function=tf.nn.sigmoid) #隐藏层
     
    prediction = add_layer(l2,22,2,activation_function=None) #输出层  
  
	#predition值与y_data差别  
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1])) #square()平方,sum()求和,mean()平均值  
  
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #0.1学习效率,minimize(loss)减小loss误差  
  
    init = tf.global_variables_initializer()  
    sess = tf.Session()  
    sess.run(init) #先执行init  

    
	#训练N次  
    for i in range(10):  
       sess.run(train_step,feed_dict={xs:x_data,ys:y_data}) 

    su=0
    correct=0
	#测试集验证
    for r in X1[60:]:
        s=np.zeros((1,2))
        s[0][0],s[0][1]=r[0],r[1]
     
        prediction_value = sess.run(prediction,feed_dict={xs:s,})  
    
    
		 
        su=su+1
        if((r[2]>0.5 and prediction_value[0][0]>0.5) and (r[3]>0.5 and prediction_value[0][1]>0.5)):
            correct=correct+1
    
        if((r[2]<0.5 and prediction_value[0][0]<0.5) and (r[3]>0.5 and prediction_value[0][1]>0.5)):
            correct=correct+1
        if((r[2]>0.5 and prediction_value[0][0]>0.5) and (r[3]<0.5 and prediction_value[0][1]<0.5)):
            correct=correct+1
    for r in X2[30:]:
        s=np.zeros((1,2))
        s[0][0],s[0][1]=r[0],r[1]
     
        prediction_value = sess.run(prediction,feed_dict={xs:s,})  
    
    
		 
        su=su+1
        if((r[2]>0.5 and prediction_value[0][0]>0.5) and (r[3]>0.5 and prediction_value[0][1]>0.5)):
            correct=correct+1
    
        if((r[2]<0.5 and prediction_value[0][0]<0.5) and (r[3]>0.5 and prediction_value[0][1]>0.5)):
            correct=correct+1
        if((r[2]>0.5 and prediction_value[0][0]>0.5) and (r[3]<0.5 and prediction_value[0][1]<0.5)):
            correct=correct+1
    for r in X3[30:]:
        s=np.zeros((1,2))
        s[0][0],s[0][1]=r[0],r[1]
     
        prediction_value = sess.run(prediction,feed_dict={xs:s,})  
    
  
        su=su+1
        if((r[2]>0.5 and prediction_value[0][0]>0.5) and (r[3]>0.5 and prediction_value[0][1]>0.5)):
            correct=correct+1
    
        if((r[2]<0.5 and prediction_value[0][0]<0.5) and (r[3]>0.5 and prediction_value[0][1]>0.5)):
            correct=correct+1
        if((r[2]>0.5 and prediction_value[0][0]>0.5) and (r[3]<0.5 and prediction_value[0][1]<0.5)):
            correct=correct+1
    
    t2=datetime.datetime.now()
    print(t2.strftime('%H:%M:%S.%f'))
    print("耗时："+str(millis(t1,t2)))
    print("总数"+str(su))
    print("准确"+str(correct))
    print("准确率"+str(correct*100/su)+"%")



    #print(result)
    c=np.array(train_data1)
    plt.scatter(c[:,0], c[:,1], marker = 'o',color = 'red', s = 40 ,label = 'Second')
    c=np.array(train_data2)
    plt.scatter(c[:,0], c[:,1], marker = 'o',color = 'yellow', s = 40 ,label = 'Second')
    c=np.array(train_data3)
    plt.scatter(c[:,0], c[:,1], marker = 'o',color = 'blue', s = 40 ,label = 'Second')
    #print(result)
    plt.show() 
    t=0