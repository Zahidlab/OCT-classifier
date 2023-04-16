# Creating Image Paths

PATCH_SIZE = 64

patch_root_dir = "/content/drive/MyDrive/Research/OCT datasets/Patch OCT Dataset"
def create_patch(patch_size = 64, dest_dir = patch_root_dir):

  #Check if the dest_dir is already availabe, if so, delete it otherwise create
  if isdir(patch_root_dir):
    del_contents(patch_root_dir)
  else:
    os.mkdir(patch_root_dir)


  #setting the different paths for clean and Noisy patch
  clean_patch_dir = join(patch_root_dir, 'Clean')
  noisy_patch_dir = join(patch_root_dir, 'Noisy')

  if not isdir(clean_patch_dir):
    makedirs(clean_patch_dir)
    makedirs(noisy_patch_dir)


  patch_list = []

  #setting the different paths for raw images
  clean_raw_image_dir = '/content/drive/MyDrive/Research/OCT datasets/Organised Dataset/Clean'
  noisy_raw_image_dir = '/content/drive/MyDrive/Research/OCT datasets/Organised Dataset/Noisy'


  for folder in sorted(os.listdir(clean_raw_image_dir)):
    clean_image_path = join(clean_raw_image_dir, folder, folder+'.jpg')
    noisy_image_path = join(noisy_raw_image_dir, folder, folder+'.jpg')

    clean_image = cv2.imread(clean_image_path,0)
    noisy_image = cv2.imread(noisy_image_path, 0)

    if(clean_image.shape != noisy_image.shape):
      print(f"Shape mismatch {clean_image.shape} - {noisy_image.shape} - {folder}")

    height, width = clean_image.shape

    clean_patch_folder_path = join(clean_patch_dir, folder)
    noisy_patch_folder_path = join(noisy_patch_dir, folder)
    mkdir(clean_patch_folder_path)
    mkdir(noisy_patch_folder_path)
    
    index = 1
    for h in range(height//PATCH_SIZE):
      # print(i)
      for w in range(width//PATCH_SIZE):
        # print(f"j = {j}")
        clean_patch = clean_image[h*PATCH_SIZE:(h+1)*PATCH_SIZE , w*PATCH_SIZE:(w+1)*PATCH_SIZE , ]
        clean_patch_image_path = join(clean_patch_folder_path, f"{index:03d}.jpg")
        cv2.imwrite(clean_patch_image_path, clean_patch)
        ######################
        noisy_patch = noisy_image[h*PATCH_SIZE:(h+1)*PATCH_SIZE , w*PATCH_SIZE:(w+1)*PATCH_SIZE , ]
        noisy_patch_image_path = join(noisy_patch_folder_path, f"{index:03d}.jpg")
        cv2.imwrite(noisy_patch_image_path, noisy_patch)

        # print(clean_patch_image_path, noisy_patch_image_path)


      patch_list.append([noisy_patch, clean_patch])
      index+=1

      # plt.subplot(1,2,1)
      # plt.imshow(noisy_patch, cmap='gray')
      # plt.subplot(1,2,2)
      # plt.imshow(clean_patch, cmap='gray')
  #     break
  #   break
  # break

 
