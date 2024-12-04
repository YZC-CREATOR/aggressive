#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# 读取训练集和测试集
train_data = pd.read_excel('C:/Users/yzc/Desktop/train_data.xlsx')
test_data = pd.read_excel('C:/Users/yzc/Desktop/test_data.xlsx')

# 选定特征变量
selected_features = ["frequency3", "MVI", "margin", "multiple", "dianeter5", "AFP400", "ALBI", "CA199", "Ki67"]

# 分割训练集的特征和目标变量
X_train = train_data[selected_features]
y_train = train_data['AR']  # 请确保训练集中存在'target'这一列

# 分割测试集的特征和目标变量
X_test = test_data[selected_features]
y_test = test_data['AR']  # 请确保测试集中也存在'target'这一列

# XGBoost模型参数
params_xgb = {
    'learning_rate': 0.02,            
    'booster': 'gbtree',              
    'objective': 'binary:logistic',   
    'max_leaves': 127,                
    'verbosity': 1,                   
    'seed': 42,                       
    'nthread': -1,                    
    'colsample_bytree': 0.6,          
    'subsample': 0.7,                 
    'eval_metric': 'logloss'          
}

# 初始化XGBoost分类模型
model_xgb = xgb.XGBClassifier(**params_xgb)

# 定义参数网格，用于网格搜索
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],  # 树的数量
    'max_depth': [3, 4, 5, 6, 7],               # 树的深度
    'learning_rate': [0.01, 0.02, 0.05, 0.1],   # 学习率
}

# 使用GridSearchCV进行网格搜索和k折交叉验证
grid_search = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_grid,
    scoring='neg_log_loss',  # 评价指标为负对数损失
    cv=5,                    # 5折交叉验证
    n_jobs=-1,               # 并行计算
    verbose=1                # 输出详细进度信息
)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最优参数
print("Best parameters found: ", grid_search.best_params_)
print("Best Log Loss score: ", -grid_search.best_score_)

# 使用最优参数训练模型
best_model = grid_search.best_estimator_

# 预测测试集
y_pred = best_model.predict(X_test)

# 输出模型报告， 查看评价指标
print(classification_report(y_test, y_pred))

# 预测概率
y_score = best_model.predict_proba(X_test)[:, 1]

# 计算ROC曲线
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_score)
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr_logistic, tpr_logistic, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_logistic)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[2]:


import joblib
# 保存模型
joblib.dump(best_model , 'XGBoost.pkl')


# In[3]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost.pkl')

# Define feature names and options (assuming all features are binary)
feature_names = ["frequency3", "MVI", "margin", "multiple", "dianeter5", "AFP400", "ALBI", "CA199", "Ki67"]

# Streamlit user interface
st.title("Heart Disease Predictor")

# Binary features inputs (0 or 1)
frequency3 = st.selectbox("Frequency3 (0 or 1):", options=[0, 1])
MVI = st.selectbox("MVI (0 or 1):", options=[0, 1])
margin = st.selectbox("Margin (0 or 1):", options=[0, 1])
multiple = st.selectbox("Multiple (0 or 1):", options=[0, 1])
dianeter5 = st.selectbox("Dianeter5 (0 or 1):", options=[0, 1])
AFP400 = st.selectbox("AFP400 (0 or 1):", options=[0, 1])
ALBI = st.selectbox("ALBI (0 or 1):", options=[0, 1])
CA199 = st.selectbox("CA199 (0 or 1):", options=[0, 1])
Ki67 = st.selectbox("Ki67 (0 or 1):", options=[0, 1])

# Process inputs and make predictions
feature_values = [frequency3, MVI, margin, multiple, dianeter5, AFP400, ALBI, CA199, Ki67]
features = np.array([feature_values])

# Prediction when the button is pressed
if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    
    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    
    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (f"According to our model, you have a high risk of aggressive recurrence. "
                  f"The model predicts that your probability of having aggressive recurrence  is {probability:.1f}%. "
                  "While this is just an estimate, it suggests that you may be at significant risk. "
                  "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
                  "to ensure you receive an accurate diagnosis and necessary treatment.")
    else:
        advice = (f"According to our model, you have a low risk of aggressive recurrence. "
                  f"The model predicts that your probability of not having aggressive recurrence is {probability:.1f}%. "
                  "However, maintaining a healthy lifestyle is still very important. "
                  "I recommend regular check-ups to monitor your heart health, "
                  "and to seek medical advice promptly if you experience any symptoms.")
    st.write(advice)
    
    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    
    # Save SHAP force plot and display in Streamlit
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")


# In[ ]:




