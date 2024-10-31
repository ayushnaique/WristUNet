import cv2
import json
import random
import numpy as np 
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split

SIZE = (512,1024)
image_path = '../Images'
seg_path = '../Segmentation_maps'

color_to_class = {
    (0, 0, 0): 0,      # Class 0 - Black
    (0, 255, 0): 1,    # Class 1 - Green
    (0, 0, 255): 2     # Class 2 - Red
}

def convert_rgb_to_class(seg_map, color_to_class):
    # print(seg_map.shape)
    single_channel_map = np.zeros((seg_map.shape[0], seg_map.shape[1]), dtype=np.uint8)
    for color, class_index in color_to_class.items():
        mask = np.all(seg_map == np.array(color).reshape(1, 1, 3), axis=-1)
        single_channel_map[mask] = class_index
    return single_channel_map

class GrazData(Dataset):
    def __init__(self, path, transform=None):
        self.data = []
        self.transform = transform
        self.path = path
        with open(self.path, 'rt') as f:
            data = json.load(f)

        for object in data:
            src = object['source']
            sgmt = object['segment']
            elem = {
                        "source" : f'{image_path}/{src}.png',
                        "segment" : f'{seg_path}/{sgmt}.png',
                        "condition" : object['condition']
                    }
            self.data.append(elem)

    def __len__(self):
        # return len(self.data)
        return 10
    
    def __getitem__(self, index):
        item = self.data[index]
        source_filename = item['source']
        segment_filename = item['segment']
        fracture = item['condition']

        source = cv2.imread(source_filename)
        source = cv2.resize(source, SIZE)
        segment = cv2.imread(segment_filename, cv2.IMREAD_GRAYSCALE)
        segment = cv2.resize(segment, SIZE)


        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # source = cv2.transpose(source) # reshape to [ch, h, w]
        # source = cv2.flip(source, flipCode=1)

        _, segment = cv2.threshold(segment, 1, 255, cv2.THRESH_BINARY)
        # segment = cv2.transpose(segment) # reshape to [ch, h, w]
        # segment = cv2.flip(segment, flipCode=1)

        # segment = convert_rgb_to_class(segment, color_to_class)

        source = source.astype(np.float32) / 255.0
        segment = segment.astype(np.float32) / 255.0
        # segment = segment.astype(np.float32)
        # print("LOADER: ", source.shape, " ", segment.shape)

        if self.transform:
            source = self.transform(source)
            segment = self.transform(segment)

        return dict(id=source_filename, jpg=source, sgmt=segment, frc=fracture)
    
if __name__ == "__main__":
    dataset = GrazData('./data.json')
    print("Total data elements: ", len(dataset))

    # Example of accessing a single item
    image_number = random.randint(0, len(dataset) - 1)
    sample = dataset[image_number]
    jpg = sample['jpg']
    sgmt = sample['sgmt']
    print(f"Sample JPG shape: {jpg.shape}")
    print(f"Sample SGMT shape: {sgmt.shape}")
    print("Max pixel value in image is: ", jpg.max())
    print("Labels in the mask are : ", np.unique(sgmt))
    print(f"Sample FRC: {sample['frc']}")

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(jpg[:,:,0], cmap='gray')
    plt.subplot(122)
    plt.imshow(sgmt[:,:], cmap='gray')
    plt.show()

    # Create DataLoader
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # # Example of iterating through DataLoader
    # for batch_idx, batch in enumerate(dataloader):
    #     print(f"Batch JPG shape: {batch['jpg'].shape}")
    #     print(f"Batch SGMT shape: {batch['sgmt'].shape}")
    #     # sample = batch['sgmt'][0].cpu().numpy()
    #     # cv2.imwrite(f'./Temp/Mask{batch_idx}.png', sample*255)
    #     print(f"Batch FRC: {batch['frc']}")