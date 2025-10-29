from torch.utils.data import Dataset
from PIL import Image
import os

class Mydataset(Dataset):

    def __init__ (self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path_list = os.listdir(self.path)
    
    def __getitem__(self,index):
        image_name = self.image_path_list[index]
        image_path = os.path.join(self.root_dir,  self.label_dir, image_name)
        image = Image.open(image_path)
        label = self.label_dir
        return image, label
    
    def __len__(self):
        return len(self.image_path_list)
    
root_dir = r"D:\数据\练习数据\hymenoptera_data\train"
label_dir = "ants"
ant1 = Mydataset(root_dir, label_dir)
image, label = ant1[0]
