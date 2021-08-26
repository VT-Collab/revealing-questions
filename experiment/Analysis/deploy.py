import numpy as np
import pickle
import matplotlib.pyplot as plt


def import_data(user, method):
    filename = "../Data/Study2/user" + user + "/plate_" + method + "_qs" + ".pkl"
    data = pickle.load(open(filename, "rb"))
    return data

def method_count(user_n, method):
    question_count = []
    users = [str(i) for i in range(1,user_n)]
    for user in users:
        count = import_data(user, method)
        question_count.append(len(count))
    return question_count


total_users = 11

# counting number of questions for each method (e.g., Informative & Ours)
ig_count = method_count(total_users, 'ig')
tf_count = method_count(total_users, 'tf')
tf_count_avg = np.mean(tf_count)


# plots
users = list(range(1,total_users))
users = [str(i) for i in users]

# plt.bar(users, tf_count)
plt.plot(users, tf_count)
plt.axhline(y = tf_count_avg, linestyle = '-')
plt.plot(users, ig_count)
plt.ylabel("Number of Questions")
plt.xlabel("User")
# plt.xlim([-0.5, 10])
plt.xlim([0, 10])
plt.ylim([0, 13])

plt.savefig('line.svg')
plt.show()
