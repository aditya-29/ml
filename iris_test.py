from sklearn import datasets
from sklearn import tree
clf=tree.DecisionTreeClassifier()
from sklearn.model_selection import train_test_split as tts

iris=datasets.load_iris()
feat=iris.data
lab=iris.target

train_feat,test_feat,train_lab,test_lab=tts(feat,lab,test_size=0.5)

clf.fit(train_feat,train_lab)
predict=clf.predict(test_feat)
print("prediction:\n",predict)
print("real value:\n",test_lab)

s=0
for i in range(len(predict)):
    if(predict[i]==test_lab[i]):
        s=s+1
print("accuracy:{}".format((s/len(predict))*100))
