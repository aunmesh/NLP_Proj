import numpy as np
import matplotlib.pyplot as plt
import numpy as np

t1 = np.arange(0.0, 1.0, 0.01)
t2 = np.arange(0.0, 1.0, 0.01)
correct_vecs = np.amax(np.loadtxt("../logs/correct_vecs.txt").view(float).reshape(-1,7),axis=1) 
count_correct = np.count_nonzero(correct_vecs > t1.reshape(-1, 1), axis=1)
wrong_vecs = np.amax(np.loadtxt("../logs/err_vecs.txt").view(float).reshape(-1,7),axis=1)
count_wrong = np.count_nonzero(wrong_vecs > t1.reshape(-1, 1), axis=1)

# print(t1)
fig = plt.figure(figsize=(10, 10))

plt.title("Graph of number of questions vs probability of predicted", fontsize = 20)

plt.xlabel('Probability', fontsize=20, color = 'blue')
plt.ylabel('Number of Questions ', fontsize=20, color = 'blue')

plt.xticks(np.arange(0.0, 2.0, 0.1))
plt.xlim(xmax = 1.2)
plt.yticks(np.arange(0.0, count_correct[0]+ 1000, count_correct[0]/20))

# plt.subplot(211)
plt.plot(t1, count_correct, 'k', color = 'green', label = 'correct')
plt.plot(t2, count_wrong, 'k', color = 'red', label = 'wrong')
plt.legend()
plt.xlim(xmax = 1.2)
plt.ylim(ymax = count_correct[0] + 1000)
# plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()