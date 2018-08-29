from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn import svm
clf=svm.SVC(kernel="linear")
wine=datasets.load_wine()
feat=wine.data
lab=wine.target

train_feat,test_feat,train_lab,test_lab=tts(feat,lab,test_size=0.2)
clf.fit(train_feat,train_lab)

predict=clf.predict(test_feat)

s=0
print("prediction:\n",predict)
print("the real one:\n",test_lab)

for i in range(len(predict)):
    if(predict[i]==test_lab[i]):
        s=s+1
print("accuracy:{}".format((s/len(predict))*100))
