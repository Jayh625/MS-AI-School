import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from handwrite_model import CNN
from handwrite_dataset import HandWriteDataSet
import handwrite_utils as handwrite
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    model = CNN().to(device)
    transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
    
    total_dataset = HandWriteDataSet(
        "C:\\Users\\Jay\\Documents\\Dev\\Git\\MS-AI-School\\Python\\0703\\train\\손글씨_데이터",
        transform=transform
    )

    test_len = int(len(total_dataset) * 0.2)
    train_len = len(total_dataset) - test_len

    train_subset, test_subset = random_split(
        total_dataset, 
        [train_len, test_len]
        )
    
    train_dataset = train_subset.dataset
    test_dataset = test_subset.dataset

    # temp_dict = dict.fromkeys(list(range(10)), 0)
    # for _, label in test_dataset:
    #     temp_dict[label] += 1

    # print(temp_dict)

    train_loader = DataLoader(train_dataset, 
                              batch_size = 64, 
                              shuffle=True)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=64)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    handwrite.train(model, len(train_dataset), train_loader,
                    criterion, optimizer, 10, device)
    
    handwrite.eval(model, test_loader, device)