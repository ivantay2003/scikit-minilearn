from sklearn import svm , datasets

digits = datasets.load_digits()

classifier = svm.SVC(gamma=0.001 , C=100)

# train with all but last item in dataset
# test with last item in dataset
classifier.fit(digits.data[:-1], digits.target[:-1])
prediction = classifier.predict(digits.data[-1:])

print ("prediction :" + str (prediction) + "\n")

# reshape data
# transform image into vector (length=64)
import matplotlib.pyplot as plt
print (digits.images.shape)
print (str(digits.images.shape[0]) + " images of")
print (str(digits.images.shape[1]) + " height by")
print (str(digits.images.shape[2]) + " width")

plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)

data = digits.images.reshape((digits.images.shape[0], -1))