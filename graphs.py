import numpy as np
import matplotlib.pyplot as plt

# 1
x = np.linspace(-10, 10, 100)
y = pow(x, 2) * np.sin(x)
plt.plot(x, y)
plt.title("y = x^2 * sin(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

# 2
data_hist = np.random.normal(loc=5, scale=2, size=1000)
plt.hist(data_hist, bins=30, alpha=0.6, color='pink', edgecolor='black')
plt.title("Histogram of Normal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.show()

# 3
hobbies = ['Reading', 'Traveling', 'Cooking', 'Gaming']
hobby_counts = [15, 20, 10, 25]
plt.pie(hobby_counts, labels=hobbies, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Hobbies Distribution")
plt.show()

# 4
fruits = ['Apples', 'Bananas', 'Cherries', 'Kiwi']

apple = np.random.normal(130, 10, 100)
banana = np.random.normal(130, 10, 100)
cherries = np.random.normal(130, 10, 100)
kiwi = np.random.normal(130, 10, 100)

data = [apple, banana, cherries, kiwi]

plt.boxplot(data, labels=fruits)
plt.title("Boxplot of Fruit Weights")
plt.ylabel("Weight (grams)")
plt.show()

# 5
x = np.random.uniform(0, 1, 100)
y = np.random.uniform(0, 1, 100)
plt.scatter(x, y, color='green', alpha=0.6)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Uniformly Distributed Random Points")
plt.show()

# 6
x = np.linspace(-2 * np.pi, 2 * np.pi, 500)

f = np.sin(x)
g = np.cos(x)
h = np.sin(x) + np.cos(x)

plt.plot(x, f, label='f(x) = sin(x)', color='blue')
plt.plot(x, g, label='g(x) = cos(x)', color='green')
plt.plot(x, h, label='h(x) = sin(x) + cos(x)', color='red')

plt.grid(True)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graphs of f(x), g(x), and h(x)')
plt.show()