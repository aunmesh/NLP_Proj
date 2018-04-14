import numpy as np
import matplotlib.pyplot as plt
import numpy as np

t1 = np.arange(0.0, 1.0, 0.01)
t2 = np.arange(0.0, 1.0, 0.01)
correct_vecs = np.amax(np.loadtxt("../logs/correct_vecs.txt").view(float).reshape(-1,7),axis=1) 
count_correct = np.count_nonzero(correct_vecs > t1.reshape(-1, 1), axis=1)/(correct_vecs.shape[0]*1.0)
wrong_vecs = np.amax(np.loadtxt("../logs/err_vecs.txt").view(float).reshape(-1,7),axis=1)
count_wrong = np.count_nonzero(wrong_vecs > t1.reshape(-1, 1), axis=1)/(wrong_vecs.shape[0]*1.0)

# print(t1)
plt.figure(figsize=(10, 10))

plt.title("Graph of Probability of predicted vs fraction of questions", fontsize = 20)

plt.xlabel('Fraction of Questions ', fontsize=20, color = 'blue')
plt.ylabel('Probability', fontsize=20, color = 'blue')

plt.xticks(np.arange(0.0, 2.0, 0.1))
plt.yticks(np.arange(0.0, 2.0, 0.1))

# plt.subplot(211)
plt.plot( count_correct, t1, 'k', color = 'green', label = 'correct')
plt.plot(count_wrong, t2, 'k', color = 'red', label = 'wrong')
plt.legend()
plt.xlim(xmax = 1.2)
plt.ylim(ymax = 1.2)
# plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.savefig("ProbvsFracQuesCorrWrong.png")
plt.show()