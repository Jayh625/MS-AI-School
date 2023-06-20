class CustomDataset:
    def __init__(self, path, mode='train') :
        self.a = [1,2,3,4,5]
        pass

    def __getitem__(self, index):
        return self.a[index] + 2
    
    def __len__(self):
        return len(self.a)
    
    def __add__(self, a):
        return self.a[0] + a

dataset_inst = CustomDataset('./iamges/', 'valid')

for item in dataset_inst:
    print(item)
print(len(dataset_inst))

print(dataset_inst + 3)