import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests

import os
# 使用 hf-mirror 镜像 下载clip权重
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# proxy = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = proxy
# os.environ['https_proxy'] = proxy 

cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
device = "cuda:3" if torch.cuda.is_available() else "cpu"
# vlmodel, preprocess = clip.load("ViT-B/32", device=device)
model_type = 'ViT-H-14'
import open_clip

# vlmodel, preprocess_train, feature_extractor =[],[],[]#感觉用不上 直接用空字符代替
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device = device)

import json
# Load the configuration from the JSON file
#config_path = "data_config.json"#配置文件
config_path="/Retrieval/data_config.json"# 配置文件路径
with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Access the paths from the config
data_path = config["data_path"]# 获取数据路径
img_directory_training = config["img_directory_training"]# 获取训练图像目录
img_directory_test = config["img_directory_test"] # 获取测试图像目录


class EEGDataset():
    """
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """
    def __init__(self, data_path, exclude_subject=None, subjects=None, train=True, time_window=[0, 1.0], classes = None, pictures = None, val_size=None):
        self.data_path = data_path  # 存储数据路径
        self.train = train  # 是否为训练模式
        self.subject_list = os.listdir(data_path)  # 获取所有受试者列表
        #self.subjects = self.subject_list if subjects is None else subjects  # 设置受试者列表
        self.subjects = self.subject_list if not subjects  else subjects  # 设置受试者列表
        self.n_sub = len(self.subjects)  # 受试者数量
        self.time_window = time_window  # 时间窗口范围
        self.n_cls = 1654 if train else 200  # 训练/测试类别数
        self.classes = classes  # 指定类别
        self.pictures = pictures  # 指定图片
        self.exclude_subject = exclude_subject  # 排除的受试者
        self.val_size = val_size  # 验证集大小
        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)#  确保指定的受试者存在于列表中
  
        self.data, self.labels, self.text, self.img = self.load_data()
        #d,l,t,i = self.data, self.labels, self.text, self.img
        #单一患者单一模型情况下 ： 
        #d self.data shape =torch.Size([66160, 63, 250]) 
        #l self.labels shape =torch.Size([66160])
        #t self.text shape =数组len ([66160]) 其中每个元素是一个字符串如 "This picture is a cat sitting on a couch."
        #i self.img shape =数组len ([66160]) 其中每个元素是一个字符串如 '/THINGS-EEG_images_set/training_images/00004_acorn/acorn_08s.jpg'

        self.data = self.extract_eeg(self.data, time_window)# 提取EEG数据的时间窗口部分        
        
        if self.classes is None and self.pictures is None:# 如果没有指定特定类别和图片
            # Try to load the saved features if they exist
            features_filename = os.path.join(f'{model_type}_old_features_train.pt') if self.train else os.path.join(f'{model_type}_old_features_test.pt')
            
            if os.path.exists(features_filename) :# 如果特征文件存在
                saved_features = torch.load(features_filename)# 加载保存的特征
                self.text_features = saved_features['text_features'] # 文本特征# 训练 shape =torch.Size([1654, 1024]) 测试shape =torch.Size([200, 1024])
                self.img_features = saved_features['img_features']# 图像特征
            else:
                self.text_features = self.Textencoder(self.text) # 编码文本特征
                self.img_features = self.ImageEncoder(self.img) # 编码图像特征
                torch.save({
                    'text_features': self.text_features.cpu(),
                    'img_features': self.img_features.cpu(),
                }, features_filename) # 保存特征
        else:
            self.text_features = self.Textencoder(self.text)# 编码文本特征
            self.img_features = self.ImageEncoder(self.img)# 编码图像特征
            
    def load_data(self):
        data_list = []
        label_list = []
        texts = []
        images = []
        
        if self.train: # 根据训练/测试模式设置目录
            directory = img_directory_training
        else:
            directory = img_directory_test
        
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort() # 排序
        
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        #获取文本描述放入texts列表中
        for dir in dirnames:# 遍历目录名
            
            try:
                idx = dir.index('_')# 查找下划线位置
                description = dir[idx+1:]   # 提取描述部分
            except ValueError:
                print(f"Skipped: {dir} due to no '_' found.")# 如果没找到下划线则跳过
                continue
                
            new_description = f"An image of  {description}"
            texts.append(new_description)

        if self.train:# 根据训练/测试模式设置图像目录
            img_directory = img_directory_training  
        else:
            img_directory = img_directory_test
        
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()  
        #获取图片路径放入images列表中
        if self.classes is not None and self.pictures is not None: # 如果指定了类别和图片
            images = []  
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                pic_idx = self.pictures[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None: # 如果只指定了类别
            images = []  
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        elif self.classes is None: # 如果没有指定类别
            images = []  
            for folder in all_folders:
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()  
                images.extend(os.path.join(folder_path, img) for img in all_images)
        else:
            
            print("Error")
            
        print("self.subjects", self.subjects)
        print("exclude_subject", self.exclude_subject)

        for subject in self.subjects: # 遍历受试者
            if self.train:# 训练模式
                if subject == self.exclude_subject:  
                    continue            
                # print("subject:", subject)    
                file_name = 'preprocessed_eeg_training.npy' # 训练数据文件名

                file_path = os.path.join(self.data_path, subject, file_name) # 构建文件路径
                data = np.load(file_path, allow_pickle=True)# 加载eeg数据
                
                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()     # 转换为tensor            
                times = torch.from_numpy(data['times']).detach()[50:]# 时间戳数据
                ch_names = data['ch_names']   # 通道名称

                n_classes = 1654   # 类别数
                samples_per_class = 10   # 每类样本数
                
                if self.classes is not None and self.pictures is not None:# 如果指定了类别和图片
                    for c, p in zip(self.classes, self.pictures):
                        start_index = c * 1 + p
                        if start_index < len(preprocessed_eeg_data):  
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]  
                            labels = torch.full((1,), c, dtype=torch.long).detach()  
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)  

                elif self.classes is not None and self.pictures is None: # 如果只指定了类别
                    for c in self.classes:
                        start_index = c * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()  
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                else: # 默认情况
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        # if self.exclude_subject==None:
                        #     preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        # else:
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 1)
                        # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 0)
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # 为当前类别的所有样本创建标签
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                 
            else:# 测试模式
                if subject == self.exclude_subject or self.exclude_subject==None:  
                    file_name = 'preprocessed_eeg_test.npy' # 测试数据文件名
                    file_path = os.path.join(self.data_path, subject, file_name)
                    data = np.load(file_path, allow_pickle=True)
                    preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                    times = torch.from_numpy(data['times']).detach()[50:]
                    ch_names = data['ch_names']  
                    n_classes = 200  # Each class contains 1 images# 每个类别包含1张图片
                    
                    samples_per_class = 1  

                    for i in range(n_classes):
                         # 如果定义了特定类别且当前类别不在列表中，则跳过
                        if self.classes is not None and i not in self.classes:  # If we've defined specific classes and the current class is not in the list, skip
                            continue
                        # 更新每个类别的起始索引
                        start_index = i * samples_per_class  # Update start_index for each class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()# 添加类别标签  # Add class labels
                        
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)  # Add labels to the label list # 将标签添加到标签列表
                else:
                    continue
        # datalist: (subjects * classes) * (10 * 4 * 17 * 100)
        # data_tensor: (subjects * classes * 10 * 4) * 17 * 100
        # data_list = np.mean(data_list, )
        # print("data_list", len(data_list))
        if self.train:
            # print("data_list", *data_list[0].shape[1:])            
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])                 
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[1:])
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
            # print("label_tensor", label_tensor.shape)
            print("data_tensor", data_tensor.shape)
        else:           
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
            # label_tensor = torch.cat(label_list, dim=0)
            # print("label_tensor", label_tensor.shape)
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
        # print("data_tensor", data_tensor.shape)
        # label_list: (subjects * classes) * 10
        # label_tensor: (subjects * classes * 10)
        # print("label_tensor = torch.cat(label_list, dim=0)")
        # print(label_list)
        label_tensor = torch.cat(label_list, dim=0)
        # label_tensor = torch.cat(label_list, dim=0)
        # print(label_tensor[:300])
        if self.train:
            # label_tensor: (subjects * classes * 10 * 4)
            label_tensor = label_tensor.repeat_interleave(4)
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                unique_values = torch.tensor(lis)        
                mapping = {val.item(): index for index, val in enumerate(unique_values)}   
                label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)

        else:
            # label_tensor = label_tensor.repeat_interleave(80)
            # if self.classes is not None:
            #     unique_values = torch.unique(label_tensor, sorted=False)
           
            #     mapping = {val.item(): index for index, val in enumerate(torch.flip(unique_values, [0]))}
            #     label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)
            pass      

                    
        self.times = times
        self.ch_names = ch_names

        print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")
        
        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Get the indices of the times within the specified window
        indices = (self.times >= start) & (self.times <= end)
        # print("self.times", self.times.shape)
        # print("indices", indices)
        # print("indices", indices.shape)
        # print("eeg_data", eeg_data.shape)
        # Use these indices to select the corresponding data
        extracted_data = eeg_data[..., indices]
        # print(f"extracted_data shape: {extracted_data.shape}")

        return extracted_data
    
    def Textencoder(self, text):   
        # 将输入文本转换为token序列并移动到指定设备上
        # 使用 open_clip 的 tokenizer（之前使用 clip.tokenize 会在 open_clip 场景下报错）
        try:
            text_inputs = open_clip.tokenize(text).to(device)
        except Exception:
            # 兼容单个字符串或回退策略
            text_inputs = torch.cat([open_clip.tokenize([t]) for t in text]).to(device)

        # 在不计算梯度的情况下，使用视觉语言模型编码文本
        with torch.no_grad():
            text_features = vlmodel.encode_text(text_inputs)#这里的vlmodel是clip
        
        # 对文本特征进行归一化处理并分离梯度计算图
        text_features = F.normalize(text_features, dim=-1).detach()
   
        return text_features
        
    def ImageEncoder(self,images):
        batch_size = 20  # 设置批处理大小为20
        image_features_list = [] # 创建存储图像特征的列表
      
        # 按批次处理图像，避免内存不足
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size] # 获取当前批次的图像路径
            # 对每张图像进行预处理并堆叠成张量，然后移至指定设备
            image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)

            # 在不计算梯度的情况下，使用视觉语言模型编码图像
            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)#这里的vlmodel是clip
                # 对图像特征进行归一化处理
                batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

            image_features_list.append(batch_image_features) # 将当前批次的特征添加到列表中

        # 将所有批次的图像特征拼接成一个完整的特征张量
        image_features = torch.cat(image_features_list, dim=0)
        
        return image_features
    
    def __getitem__(self, index):
        # 根据索引获取对应的数据和标签
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 10 * 4)
        x = self.data[index] # 获取EEG数据
        label = self.labels[index]# 获取标签
        # 根据是否指定pictures参数计算不同的索引参数
        if self.pictures is None:
            if self.classes is None:# 计算训练和测试时的索引基数
                # 训练模式下每个类别有10个样本，每个样本重复4次 
                index_n_sub_train = self.n_cls * 10 * 4
                # 测试模式下每个类别有1个样本，每个样本重复80次
                index_n_sub_test = self.n_cls * 1 * 80
            else: 
                # 如果指定了特定类别
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 10 * 4
            # text_index: classes # text_index: classes - 计算文本索引
            if self.train:
                # 每个类别有10*4=40个样本，通过整除得到类别索引
                text_index = (index % index_n_sub_train) // (10 * 4)
            else:
                 # 每个类别有1*80=80个样本，通过取模得到类别索引
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10 - 计算图像索引
            if self.train:
                # 每个图像有4个重复样本，通过整除得到图像索引
                img_index = (index % index_n_sub_train) // (4)
            else:
                 # 每个图像有80个重复样本，通过取模得到图像索引
                img_index = (index % index_n_sub_test)
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 1 * 4
            # text_index: classes - 计算文本索引
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10 - 计算图像索引
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)
        # print("text_index", text_index)
        # print("self.text", self.text)
        # print("self.text", len(self.text))
        text = self.text[text_index] # 获取对应的文本描述
        img = self.img[img_index] # 获取对应的图像路径
        
        text_features = self.text_features[text_index] # 获取对应的文本特征
        img_features = self.img_features[img_index] # 获取对应的图像特征
        # 返回EEG数据、标签、文本描述、文本特征、图像路径和图像特征
        return x, label, text, text_features, img, img_features

    def __len__(self):# 返回数据集的长度（样本总数）
        return self.data.shape[0]  # or self.labels.shape[0] which should be the same

if __name__ == "__main__":
    data_path = data_path# 使用之前定义的数据路径
    # 创建训练和测试数据集实例
    train_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=True)    
    test_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=False)
    # train_dataset = EEGDataset(data_path, exclude_subject = 'sub-01', train=True)    
    # test_dataset = EEGDataset(data_path, exclude_subject = 'sub-01', train=False)    
    # train_dataset = EEGDataset(data_path, train=True) 
    # test_dataset = EEGDataset(data_path, train=False) 
    
    
    
    # 创建数据加载器，批大小为1，随机打乱数据
    # 100 Hz
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    # 测试访问数据集中的某个样本
    i = 80*1-1
    x, label, text, text_features, img, img_features  = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")# 打印索引、标签和文本描述
    Image.open(img) # 打开对应的图像
            
    
        

    
