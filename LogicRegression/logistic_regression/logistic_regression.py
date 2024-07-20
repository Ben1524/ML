import numpy as np
from scipy.optimize import minimize
from LogicRegression.utils.features import prepare_for_training
from LogicRegression.utils.hypothesis import sigmoid

class LogisticRegression:
    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=False):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed, # 处理后的数据
         features_mean,  # 特征均值
         # 特征标准差
         features_deviation)  = prepare_for_training(data, polynomial_degree, sinusoid_degree,normalize_data=False)
         
        self.data = data_processed
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        
        num_features = self.data.shape[1]
        num_unique_labels = np.unique(labels).shape[0]
        self.theta = np.zeros((num_unique_labels,num_features)) # 每一个标签与其他标签二分类的参数矩阵
        # 类似于k个二分类器的参数矩阵
        
    def train(self,max_iterations=1000):
        cost_histories = []
        num_features = self.data.shape[1]
        for label_index,unique_label in enumerate(self.unique_labels): # 进行k次二分类器的训练，k为标签的个数
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features,1))
            current_lables = (self.labels == unique_label).astype(float) # 将当前标签与其他标签进行二分类
            # 如果当前标签为unique_label则为1，否则为0，astype(float)将True转换为1，False转换为0
            (current_theta,cost_history) = LogisticRegression.gradient_descent(self.data,current_lables,current_initial_theta,max_iterations)
            self.theta[label_index] = current_theta.T # 更新当前标签的参数矩阵
            cost_histories.append(cost_history)
            
        return self.theta,cost_histories
            
    @staticmethod       # 重要的是这个函数！！！！
    def gradient_descent(data,labels,current_initial_theta,max_iterations):
        cost_history = []
        num_features = data.shape[1]

        result = minimize( # 最小化损失函数，得到最优的参数
            #要优化的目标：
            # 形参接收函数参数类型fun(x, *args) -> float , x为一维数组，args为其他参数
            #lambda current_theta:LogisticRegression.cost_function(data,labels,current_initial_theta.reshape(num_features,1)),
            lambda current_theta:LogisticRegression.cost_function(data,labels,current_theta),
            #初始化的权重参数 转换为一维数组
            current_initial_theta.flatten(),
            #选择优化策略，牛顿法等
            method = 'CG',
            # 梯度下降迭代计算公式
            #jac = lambda current_theta:LogisticRegression.gradient_step(data,labels,current_initial_theta.reshape(num_features,1)),
            jac = lambda current_theta:LogisticRegression.gradient_step(data,labels,current_theta.reshape(num_features,1)),
            # 记录结果
            callback = lambda current_theta:cost_history.append(LogisticRegression.cost_function(data,labels,current_theta.reshape((num_features,1)))),
            # 迭代次数  
            options={'maxiter': max_iterations}                                               
            )
        if not result.success:
            raise ArithmeticError('Can not minimize cost function'+result.message)
        optimized_theta = result.x.reshape(num_features,1)
        return optimized_theta,cost_history
        
    @staticmethod 
    def cost_function(data,labels,theat):
        # 如果theat是一维数组，将其转换为二维数组n*1
        if len(theat.shape) == 1: # shape[0]为行数，shape[1]为列数
            theat = theat.reshape((theat.shape[0],1))

        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data,theat) # 得到预测值
        # np.log(predictions) 为预测值的对数,注意预测值是一个概率值，np.log以e为底的对数
        # 损失是对所有样本的损失求和
        # labels==1 返回真实值为1的样本的索引
        y_is_set_cost = np.dot(labels[labels == 1].T,np.log(predictions[labels == 1]))
        # labels==0 返回真实值为0的样本的索引，1-predictions[labels == 0]为预测值为0的概率
        y_is_not_set_cost = np.dot(1-labels[labels == 0].T,np.log(1-predictions[labels == 0]))
        cost = (-1/num_examples)*(y_is_set_cost+y_is_not_set_cost)
        return cost
    @staticmethod 
    def hypothesis(data,theat): # 一组确定的theta值的预测值
        
        predictions = sigmoid(np.dot(data,theat))
        
        return  predictions
    
    @staticmethod     
    def gradient_step(data,labels,theta):
        num_examples = labels.shape[0]
        predictions = LogisticRegression.hypothesis(data,theta)
        label_diff = predictions- labels
        gradients = (1/num_examples)*np.dot(data.T,label_diff)
        
        return gradients.T.flatten()
    
    def predict(self,data):
        num_examples = data.shape[0]
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree,self.normalize_data)[0]
        prob = LogisticRegression.hypothesis(data_processed,self.theta.T) # 得到每一组theta的预测值
        max_prob_index = np.argmax(prob, axis=1) # 返回每一行最大值的索引
        class_prediction = np.empty(max_prob_index.shape,dtype=object) # 创建一个空的数组
        for index,label in enumerate(self.unique_labels):
            # max_prob_index == index 返回一个布尔数组，为True的位置将label赋值给class_prediction
            class_prediction[max_prob_index == index] = label # 将概率最大的标签赋值给class_prediction
        return class_prediction.reshape((num_examples,1)) # 返回预测的标签的[[label]]形式
        
        
        
        
        
        
        
        
        
        


