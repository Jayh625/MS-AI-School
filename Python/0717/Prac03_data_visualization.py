import os 
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer : 
    def __init__(self, data_dir) :
        self.data_dir = data_dir
        self.total_data = {}
        self.train_data = {}
        self.val_data = {}
        self.test_data = {}
        self.load_data()
        self.visualize_all_data()

    def load_data(self) :
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        test_dir = os.path.join(self.data_dir, 'test')

        for label in os.listdir(train_dir) :
            label_dir = os.path.join(train_dir, label)
            count = len(os.listdir(label_dir))
            self.train_data[label] = count
            self.total_data[label] = count

        for label in os.listdir(val_dir) :
            label_dir = os.path.join(val_dir, label)
            count = len(os.listdir(label_dir))
            self.val_data[label] = count
            if label in self.total_data :
                self.total_data[label] += count
            else :
                self.total_data[label] = count

        for label in os.listdir(test_dir) :
            label_dir = os.path.join(test_dir, label)
            count = len(os.listdir(label_dir))
            self.test_data[label] = count
            if label in self.total_data :
                self.total_data[label] += count
            else :
                self.total_data[label] = count

    def visualize_data(self) :
        labels = list(self.total_data.keys())
        counts = list(self.total_data.values())
        plt.figure(figsize=(12,6))
        plt.bar(range(len(self.total_data)), counts, tick_label=labels)
        plt.title("Label data number")
        plt.xlabel("Labels")
        plt.ylabel("Counts")
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.show()

if __name__ == "__main__" :
    test = DataVisualizer("./data/food_dataset")