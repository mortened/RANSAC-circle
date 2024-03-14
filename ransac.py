import numpy as np
import matplotlib.pyplot as plt
import random as rand

#Read points from a file and return them as two arrays x and y
def read_points(filename):
    x = np.array([])
    y = np.array([])        
    with open(filename) as f:
        for line in f.readlines():
            x = np.append(x, float(line.split()[0]))
            y = np.append(y, float(line.split()[1]))    
    print(f'Loaded {len(x)} points from {filename}')
    return x,y 

#Draw 3 points uniformly at random
def random_points(x, y):
    indices = rand.sample(range(len(x)), 3)
    arr = np.array([x[indices], y[indices]])
    return arr


class RANSAC:
    def __init__(self, x, y, threshold, iterations):
        self.x = x
        self.y = y
        self.threshold = threshold
        self.iterations = iterations
        self.best_center_x = 0
        self.best_center_y = 0
        self.best_radius = 0
        self.best_inliers = 0
        self.inliers = None
        self.outliers = None
    
    def fit_circle(self, x, y):
        x1, x2, x3 = x
        y1, y2, y3 = y
        A = np.array([[2*(x2 - x1), 2*(y2 - y1)], [2*(x3 - x1), 2*(y3 - y1)]])
        b = np.array([x2**2 - x1**2 + y2**2 - y1**2, x3**2 - x1**2 + y3**2 - y1**2])
        center_x, center_y = np.linalg.solve(A, b)
        radius = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
        return center_x, center_y, radius
    
    def plot_circle(self, center_x, center_y, radius):
        #Plot the circle with center (center_x, center_y) and radius radius
        #The circle is represented as (x - center_x)**2 + (y - center_y)**2 = radius**2
        #The function should plot the circle
        theta = np.linspace(0, 2*np.pi, 100)
        x = center_x + radius*np.cos(theta)
        y = center_y + radius*np.sin(theta)
        plt.plot(x, y, c='yellow', linestyle='--', linewidth=1, label='Best Circle')
        if self.inliers is not None:
            plt.title(f'RANSAC Circle Fit: Radius = {radius:.2f}')
            plt.scatter(self.inliers[:,0], self.inliers[:,1], c='green', label=f'Inliers: {len(self.inliers)}', s=8)
            plt.scatter(self.outliers[:,0], self.outliers[:,1], c='r', label=f'Outliers: {len(self.outliers)}', s=6)
        else:
            plt.scatter(self.x, self.y, c='r', label='Points', s=10)
        plt.scatter(center_x, center_y, c='b', label=f'Center ({center_x:.2f}, {center_y:.2f})', s=20)
        plt.legend()
        plt.show()
    
    def find_inliers(self, center_x, center_y, radius):
        #Return the number of inliers for the given circle
        #The circle is represented as (x - center_x)**2 + (y - center_y)**2 = radius**2
        inliers = []
        outliers = []
        for i in range(len(self.x)):
            if np.abs(np.sqrt((self.x[i] - center_x)**2 + (self.y[i] - center_y)**2) - radius) < self.threshold:
                inliers.append((self.x[i], self.y[i]))
            else:
                outliers.append((self.x[i], self.y[i]))
        return np.array(inliers), np.array(outliers)

    def fit(self):
        for i in range(self.iterations):
            x_rand, y_rand = random_points(self.x, self.y)
            center_x, center_y, radius = self.fit_circle(x_rand, y_rand)
            inliers, outliers = self.find_inliers(center_x, center_y, radius)
            if len(inliers) > self.best_inliers:
                self.best_inliers = len(inliers)
                self.best_center_x = center_x
                self.best_center_y = center_y
                self.best_radius = radius
                self.inliers = inliers
                self.outliers = outliers

        return self.best_center_x, self.best_center_y, self.best_radius

if __name__ == "__main__":
    x, y = read_points('RANSACdata09.txt')
    ransac = RANSAC(x, y, 0.7, 20)
    center_x, center_y, radius = ransac.fit()
    ransac.plot_circle(center_x, center_y, radius)
