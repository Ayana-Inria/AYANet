import os
from PIL import Image
import numpy as np

def crop_levir():
    '''
    Crop LEVIR-CD dataset in original resolution with this folder structure:
    !!! Make sure the train, test, and val folders are named exactly like what is written below
    -LEVIR-CD/
        ├─train
            ├─A
                ├─train_1.png
                ...
            ├─B
                ├─train_1.png
                ...
            ├─label
                ├─train_1.png
        ├─test
            ├─A
                ├─test_1.png
                ...
            ├─B
                ├─test_1.png
                ...
            ├─label
                ├─test_1.png
        ├─val
            ├─A
                ├─val_1.png
                ...
            ├─B
                ├─val_1.png
                ...
            ├─label
                ├─val_1.png
            ...
    
    
    to 256 x 256 images with this folder structure
    -LEVIR-Cropped
        ├─A
            ├─img1.png
            ...
        ├─B
            ├─img1.png
            ...
        ├─label
            ├─img1.png
            ...
        └─list
            ├─val.txt
            ├─test.txt
            └─train.txt
    '''
    # Path to the root folder of original resolution images
    ori_folder = r"/mnt/c/Users/prisc/OneDrive - unige.it/Documents/Dataset/LEVIR/LEVIR-CD"
    # Path to the root folder of cropped images
    cropped_folder = r"/mnt/c/Users/prisc/OneDrive - unige.it/Documents/Dataset/LEVIR/LEVIR-Cropped"

    ori_size = 1024
    crop_size = 256

    # Prepare lists for each split
    train_list = []
    test_list = []
    val_list = []

    for split in os.listdir(ori_folder):
        # train, test, val
        split_path = os.path.join(ori_folder, split)
        for folder in os.listdir(split_path):
            # A, B, label
            folder_path = os.path.join(split_path, folder)
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                # The labels need to be in grayscale
                if folder == 'label':
                    original_image = np.array(Image.open(img_path), dtype=np.uint8)
                else:
                    original_image = np.asarray(Image.open(img_path).convert('RGB'))

                for i in range(int(ori_size/crop_size)):
                    for j in range (int(ori_size/crop_size)):
                        crop = original_image[crop_size*i:crop_size*(i+1), crop_size*j:crop_size*(j+1)]
                        print(crop.shape)
                        new_name = img.split('.png')[0] + '_' + str(i) + '_' + str(j) + '.png'
                        if folder == 'label':
                            if split == 'train': train_list.append(new_name)
                            elif split == 'test': test_list.append(new_name)
                            elif split == 'val': val_list.append(new_name)
                        new_folder_path = os.path.join(cropped_folder, folder)
                        new_img_path = os.path.join(cropped_folder, folder, new_name)
                        if not os.path.exists(new_folder_path):
                            os.makedirs(new_folder_path)
                        im = Image.fromarray(crop)
                        im.save(new_img_path)
                        print('Write to : ', new_img_path)

    # Make the list
    list_folder = os.path.join(cropped_folder, "list")
    if not os.path.exists(list_folder):
        os.makedirs(list_folder)
    with open(os.path.join(list_folder, "train.txt"), "w") as f:
        for img in train_list:
            f.write(img)
            if img != train_list[-1]:
                f.write("\n")

    with open(os.path.join(list_folder, "test.txt"), "w") as f:
        for img in test_list:
            f.write(img)
            if img != test_list[-1]:
                f.write("\n")

    with open(os.path.join(list_folder, "val.txt"), "w") as f:
        for img in val_list:
            f.write(img)
            if img != val_list[-1]:
                f.write("\n")

def crop_s2looking():
    '''
    Crop S2Looking dataset in original resolution with this folder structure:
    !!! Make sure the train, test, and val folders are named exactly like what is written below
    -S2Looking/
        ├─train
            ├─Image1
                ├─1.png
                ...
            ├─Image2
                ├─1.png
                ...
            ├─label
                ├─1.png
                ...
            ├─label1
                ├─1.png
                ...
            ├─label2
                ├─1.png
                ...
        ├─test
            ├─Image1
                ├─2.png
                ...
            ├─Image2
                ├─2.png
                ...
            ├─label
                ├─2.png
                ...
            ├─label1
                ├─2.png
                ...
            ├─label2
                ├─2.png
                ...
        ├─val
            ├─Image1
                ├─6.png
                ...
            ├─Image2
                ├─6.png
                ...
            ├─label
                ├─6.png
                ...
            ├─label1
                ├─6.png
                ...
            ├─label2
                ├─6.png
                ...
    
    
    to 256 x 256 images with this folder structure
    -S2Looking-Cropped
        ├─A
            ├─img1.png
            ...
        ├─B
            ├─img1.png
            ...
        ├─label
            ├─img1.png
            ...
        └─list
            ├─val.txt
            ├─test.txt
            └─train.txt
    '''

    # Path to the root folder of original resolution images
    ori_folder = r"/mnt/c/Users/prisc/OneDrive - unige.it/Documents/Dataset/S2Looking/S2Looking"
    # Path to the root folder of cropped images
    cropped_folder = r"/mnt/c/Users/prisc/OneDrive - unige.it/Documents/Dataset/S2Looking/S2Looking-Cropped"

    folder_mapping = {'Image1' : 'A', 'Image2' : 'B', 'label' : 'label'}
    ori_size = 1024
    crop_size = 256

    # Prepare lists for each split
    train_list = []
    test_list = []
    val_list = []

    for split in os.listdir(ori_folder):
        # train, test, val
        split_path = os.path.join(ori_folder, split)
        for src_folder, dest_folder in folder_mapping.items():
            # Image1, Image2, label
            folder_path = os.path.join(split_path, src_folder)
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                # The labels need to be in grayscale
                if src_folder == 'label':
                    original_image = np.array(Image.open(img_path), dtype=np.uint8)
                else:
                    original_image = np.asarray(Image.open(img_path).convert('RGB'))

                for i in range(int(ori_size/crop_size)):
                    for j in range (int(ori_size/crop_size)):
                        crop = original_image[crop_size*i:crop_size*(i+1), crop_size*j:crop_size*(j+1)]
                        print(crop.shape)
                        new_name = img.split('.png')[0] + '_' + str(i) + '_' + str(j) + '.png'
                        if src_folder == 'label':
                            if split == 'train': train_list.append(new_name)
                            elif split == 'test': test_list.append(new_name)
                            elif split == 'val': val_list.append(new_name)
                        new_folder_path = os.path.join(cropped_folder, dest_folder)
                        new_img_path = os.path.join(cropped_folder, dest_folder, new_name)
                        if not os.path.exists(new_folder_path):
                            os.makedirs(new_folder_path)
                        im = Image.fromarray(crop)
                        im.save(new_img_path)
                        print('Write to : ', new_img_path)

    # Make the list
    list_folder = os.path.join(cropped_folder, "list")
    if not os.path.exists(list_folder):
        os.makedirs(list_folder)
    with open(os.path.join(list_folder, "train.txt"), "w") as f:
        for img in train_list:
            f.write(img)
            if img != train_list[-1]:
                f.write("\n")

    with open(os.path.join(list_folder, "test.txt"), "w") as f:
        for img in test_list:
            f.write(img)
            if img != test_list[-1]:
                f.write("\n")

    with open(os.path.join(list_folder, "val.txt"), "w") as f:
        for img in val_list:
            f.write(img)
            if img != val_list[-1]:
                f.write("\n")

def crop_whu():
    '''
    Crop WHU-CD dataset in original resolution with this folder structure:
    !!! Make sure the train, test, and val folders are named exactly like what is written below
    -WHU/
        ├─train
            ├─Image1
                ├─1.png
                ...
            ├─Image2
                ├─1.png
                ...
            ├─label
                ├─1.png
                ...
            ├─label1
                ├─1.png
                ...
            ├─label2
                ├─1.png
                ...
        ├─test
            ├─Image1
                ├─2.png
                ...
            ├─Image2
                ├─2.png
                ...
            ├─label
                ├─2.png
                ...
            ├─label1
                ├─2.png
                ...
            ├─label2
                ├─2.png
                ...
        ├─val
            ├─Image1
                ├─6.png
                ...
            ├─Image2
                ├─6.png
                ...
            ├─label
                ├─6.png
                ...
            ├─label1
                ├─6.png
                ...
            ├─label2
                ├─6.png
                ...
    
    
    to 256 x 256 images with this folder structure
    -WHU-Cropped
        ├─A
            ├─img1.png
            ...
        ├─B
            ├─img1.png
            ...
        ├─label
            ├─img1.png
            ...
        └─list
            ├─val.txt
            ├─test.txt
            └─train.txt
    '''

    full_size_pre_folder = './Dataset/WHU/2012/'
    full_size_post_folder = './Dataset/WHU/2016/'
    full_size_label_folder = './Dataset/WHU/change_label/'
    crop_label_folder = './Dataset/WHU/label/'
    crop_pre_folder = './Dataset/WHU/A/'
    crop_post_folder = './Dataset/WHU/B/'
    folder = ['train', 'test']
    label_file = 'change_label.tif'

    cH, cW = 256, 256

    for split in folder:
        pre_file_path = os.path.join(full_size_pre_folder, split, ('2012_' + split + '.tif'))
        post_file_path = os.path.join(full_size_post_folder, split, ('2016_' + split + '.tif'))
        label_file_path = os.path.join(full_size_label_folder, split, label_file)

        label = np.array(Image.open(label_file_path), dtype=np.uint8)
        img_pre = np.asarray(Image.open(pre_file_path).convert('RGB'))
        img_post = np.asarray(Image.open(post_file_path).convert('RGB'))

        H, W = label.shape
        total = 0

        if split == 'train':
            total_W = W // cW + 1
        else:
            total_W = W // cW
        total_H = H // cH + 1

        countwh = 0
        counth = 0
        countw = 0

        for i in range(total_H):
            for j in range(total_W):
                if i == total_H - 1 and j == total_W -1:
                    print('End of H and End of W')
                    countwh += 1
                    if split == 'train':
                        crop_pre = img_pre[cH*i-6:cH*(i+1), cW*j-5:cW*(j+1)]
                        crop_post = img_post[cH*i-6:cH*(i+1), cW*j-5:cW*(j+1)]
                        crop_label = label[cH*i-6:cH*(i+1), cW*j-5:cW*(j+1)]
                    else:
                        crop_pre = img_pre[cH*i-6:cH*(i+1), cW*j:cW*(j+1)]
                        crop_post = img_post[cH*i-6:cH*(i+1), cW*j:cW*(j+1)]
                        crop_label = label[cH*i-6:cH*(i+1), cW*j:cW*(j+1)]
                elif i == total_H - 1:
                    counth += 1
                    print('End of H')
                    crop_pre = img_pre[cH*i-6:cH*(i+1), cW*j:cW*(j+1)]
                    crop_post = img_post[cH*i-6:cH*(i+1), cW*j:cW*(j+1)]
                    crop_label = label[cH*i-6:cH*(i+1), cW*j:cW*(j+1)]
                elif j == total_W -1:
                    countw += 1
                    print('End of W')
                    if split == 'train':
                        crop_pre = img_pre[cH*i:cH*(i+1), cW*j-5:cW*(j+1)]
                        crop_post = img_post[cH*i:cH*(i+1), cW*j-5:cW*(j+1)]
                        crop_label = label[cH*i:cH*(i+1), cW*j-5:cW*(j+1)]
                    else:
                        crop_pre = img_pre[cH*i:cH*(i+1), cW*j:cW*(j+1)]
                        crop_post = img_post[cH*i:cH*(i+1), cW*j:cW*(j+1)]
                        crop_label = label[cH*i:cH*(i+1), cW*j:cW*(j+1)]
                else:
                    print('Normal part')
                    crop_pre = img_pre[cH*i:cH*(i+1), cW*j:cW*(j+1)]
                    crop_post = img_post[cH*i:cH*(i+1), cW*j:cW*(j+1)]
                    crop_label = label[cH*i:cH*(i+1), cW*j:cW*(j+1)]
            
                crop_label[crop_label == 1] = 255

                pre_H, pre_W, _ = crop_pre.shape
                post_H, post_W, _ = crop_post.shape
                label_H, label_W = crop_label.shape

                print('Pre image size : ', crop_pre.shape)
                print('Post image size : ', crop_post.shape)
                print('Label size : ', crop_label.shape)

                if pre_H != 256 or pre_W != 256 or post_H != 256 or post_W != 256 or label_H != 256 or label_W != 256:
                    print('!!!!!!There is an image with a wrong size!!!!!!!')
                    print('End of W and H : ', countwh)
                    print('End of H : ', counth)
                    print('End of W : ', countw)
                    exit(-1)

                new_name = str(total) + '.png'
                new_pre_path = os.path.join(crop_pre_folder, split, new_name)
                new_post_path = os.path.join(crop_post_folder, split, new_name)
                new_label_path = os.path.join(crop_label_folder, split, new_name)

                lbl = Image.fromarray(crop_label)
                lbl.save(new_label_path)
                pre = Image.fromarray(crop_pre)
                pre.save(new_pre_path)
                post = Image.fromarray(crop_post)
                post.save(new_post_path)
                total += 1
                print('Write to : ', new_label_path)
                print('Write to : ', new_pre_path)
                print('Write to : ', new_post_path)

if __name__ == "__main__":
    #### Crop LEVIR-CD dataset to 256 x 256
    # crop_levir()

    #### Crop S2Looking dataset to 256 x 256
    crop_s2looking()

    #### Crop WHU-CD dataset to 256 x 256
    # crop_whu()