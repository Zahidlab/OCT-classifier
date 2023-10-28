import torch 
import cv2

from datetime import datetime
def get_date_time():
  a = datetime.now()
  date_time_now = a.strftime("%d_%b__%H_%M")
  return date_time_now

def about(data):
  typ = type(data)
  print(typ)

  if typ == list:
    print(len(data))
  else:
    print(data.shape)

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

import os, shutil
def del_contents(path):
  folder = path
  for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))

def single_patch(Image, PATCH_SIZE=64):

  c, height, width = Image.shape
  Image = Image.squeeze()
  # print(f"of the image -> Height {height}. Width {width}")
  patches = []
  for h in range(height//PATCH_SIZE):

      for w in range(width//PATCH_SIZE):

        patch = Image[h*PATCH_SIZE:(h+1)*PATCH_SIZE , w*PATCH_SIZE:(w+1)*PATCH_SIZE , ]

        patch = patch.unsqueeze(dim=0)
        # print(patch.shape)
        patches.append(patch)
  # complete_tensor = torch.stack(tensor_list)
  # for i in patches:
  #   print(i.shape)
  return torch.stack(patches)


import numpy as np

def stitch_patches(patches,):
    """
    Stitches together a list of image patches.
    """
    height, width = 512, 960
    p_size = 64
    cols = int(width/p_size )
    rows = int(height/p_size)
    # print(f"Rows{rows} Cols{cols}")
    Image = torch.zeros((height, width))
    # print(Image.shape)
    i=0
    for r in range(rows):
      for c in range(cols):
        rs =  r*p_size
        re = (r+1)*p_size
        cs =  c*p_size
        ce = (c+1) * p_size
        # print(rs, re, cs, ce, i)
        Image[rs:re, cs: ce] = patches[i]
        i+=1



    return Image.unsqueeze(dim=0)


def stitcher(patches, shape, patch_size, step_cut):
    if torch.is_tensor(patches):
      patches = patches.to('cpu')
    image = np.zeros(shape, dtype = np.float32)

    # about(patches)
    # if type(patches) == np.ndarray:
    #     image = np.ones(shape)
    # elif torch.is_tensor(patches):
    #     image = torch.zeros(shape)

#     patch_cont = []
    last_col = patches.shape[1]-1
    last_row = patches.shape[0]-1

    s=patch_size
    row_start = 0
    row_end = 0
    col_start =0
    col_end =0
    for row in range(patches.shape[0]):
        new_row_flag = True
        for col in range(patches.shape[1]):
            this_patch = patches[row][col]
    #         print(this_patch.shape)
            if row==0:
                if col==0:
                    patch = this_patch[:s-step_cut, :s-step_cut]

                elif col==last_col:
                    patch =this_patch[:s-step_cut, step_cut:]

                else:
                    patch = this_patch[:s-step_cut, step_cut:s-step_cut]

            elif row == last_row:
                if col==0:
                    patch = this_patch[step_cut:, :s-step_cut]

                elif col==last_col:
                    patch = this_patch[step_cut:, step_cut:]

                else:
                    patch = this_patch[step_cut:, step_cut:s-step_cut]
            else:
                if col == 0:
                    patch = this_patch[step_cut:s-step_cut, :s-step_cut]
                elif col==last_col:
                    patch = this_patch[step_cut:s-step_cut, step_cut:]
                else:
                    patch = this_patch[step_cut:s-step_cut, step_cut:s-step_cut]

#             print(f"[{row}, {col}] {patch.shape}")
#             print(f"dfd {row_start}{row_end}")
            row_end = row_start+patch.shape[0]


            col_end = col_start+patch.shape[1]



            # print(f"rs, re, cs, ce: {row_start}, {row_end}, {col_start}, {col_end} patch- {patch.shape}")
            image[row_start: row_end, col_start: col_end] = patch

#             plt.imshow(patch, "gray")
            col_start = col_end




        col_start = 0
        row_start = row_end

    return image
        
    
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_images(images):

# create a figure with X rows and 4 columns
  cols = 4
  rows = len(images)//cols
  fig, axs = plt.subplots(rows, cols, figsize=(12, 3*rows))
  idx = 0
  for i in range(rows):
    for j in range(cols):
        ax = axs[i, j]
        # plot your image using ax.imshow() or any other plotting function
        ax.imshow(images[idx], cmap = 'gray')
        # add titles, labels, etc.
        ax.set_title(f"Image {i*4 + j + 1}")
        plt.axis('off')
        idx+=1
  fig.tight_layout()
  plt.show()


def show_sample():
  test_noisy_image = n

  gen_clean_image = generator(test_noisy_image.unsqueeze(dim=1)).squeeze()
  real_clean_image = c
  print()
  fig = plt.figure(figsize = (10,10))

  with torch.inference_mode():
    fig.add_subplot(1,3,1)
    plt.imshow(test_noisy_image.squeeze().to('cpu'), cmap='gray')
    plt.title("Real Noisy Image")
    plt.axis('off')

    fig.add_subplot(1,3,2)
    plt.imshow(gen_clean_image.to('cpu'), cmap='gray')
    plt.title("Fake Clean Image")
    plt.axis('off')

    fig.add_subplot(1,3,3)
    plt.imshow(real_clean_image.squeeze(), cmap='gray')
    plt.title("Real Clean Image")
    plt.axis('off')

    plt.show()

def single_image_saver(generator,save_path, noisy, clean):
  np_image = noisy.squeeze().numpy()
  patches = patchify(np_image, (64,64), step = 32)
  patches = torch.from_numpy(patches.reshape(-1, 64, 64))
  patches = patches.unsqueeze(dim=1)
  # about(patches)

  #Output from the generator
  with torch.inference_mode():
    result_patches = generator(patches)
    
  stitched_image = stitcher(torch.reshape(result_patches, (15,29,64,64)))
#   print(about(noisy), about(clean), about(stitched_image))
#   show_sample_with_param(noisy.squeeze(),clean.squeeze(), torch.from_numpy(stitched_image))
  cv2.imwrite(save_path, stitched_image.squeeze() *255.0)

def show_sample_with_param(n, c, gen_clean_image):
  """
  N, c , gen
  """
  test_noisy_image = n
  gen_clean_image = gen_clean_image.squeeze()
  real_clean_image = c

  fig = plt.figure(figsize = (10,10))

  with torch.inference_mode():
    fig.add_subplot(1,3,1)
    plt.imshow(test_noisy_image.squeeze().to('cpu'), cmap='gray')
    plt.title("Real Noisy Image")
    plt.axis('off')

    fig.add_subplot(1,3,2)
    plt.imshow(gen_clean_image.to('cpu'), cmap='gray')
    plt.title("Fake Clean Image")
    plt.axis('off')

    fig.add_subplot(1,3,3)
    plt.imshow(real_clean_image.squeeze(), cmap='gray')
    plt.title("Real Clean Image")
    plt.axis('off')

    plt.show()


from math import log10,sqrt

def PSNR(original, compressed):
    mse = torch.mean((torch.sub(original, compressed)) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = torch.max(original)
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
from skimage.metrics import structural_similarity 
def ssim_calculator_new(generator, clean_images, fake_images):
    total_ssim = 0
    for n,c,f in zip(clean_images, fake_images):
        ssim_value, _ = structural_similarity(f.squeeze().cpu().numpy(), c.squeeze().cpu().numpy(), full = True) #ssim after
        total_ssim += ssim_value
    
    return total_ssim/len(clean_images)
def ssim_calculator(generator):
    for noisy_images, clean_images in train_dl:
        for i, image in enumerate(noisy_images):
            np_image = image.squeeze().numpy()
            patches = patchify(np_image, (64,64), step = 32)
            patches = torch.from_numpy(patches.reshape(-1, 64, 64)).to(DEVICE)
            patches = patches.unsqueeze(dim=1)
     
            with torch.inference_mode():
                clean_image = clean_images[i].to(DEVICE)
                result_patches = generator(patches)
                stitched_image = stitcher(torch.reshape(result_patches, (15,29,64,64)), step_cut=16)
                ssim = SSIM(stitched_image.squeeze(), clean_images[i].squeeze().cpu().numpy())
                plt.imshow(stitched_image.squeeze(), 'gray')
                plt.axis('off')
                return ssim
