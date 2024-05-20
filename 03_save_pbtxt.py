import os
import csv
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function to extract the vector
def get_vector(input_image):
    image = input_image.convert("RGB")  # Convert the image to RGB format if not already
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)
    my_embedding = torch.zeros([1, 512, 1, 1])

    # Define the hook function to copy the data
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    
    # Register the forward hook
    h = layer.register_forward_hook(copy_data)
    model(batch_t)
    h.remove()

    return my_embedding.squeeze().cpu().numpy()

# Initialize the model and transformation
model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

# Define the content of the .pbtxt file
pbtxt_content = """embeddings {
  tensor_path: "feature_vecs.tsv"
  metadata_path: "meta.tsv"
  sprite {
    image_path: "sprite.jpg"
    single_image_dim: 300
    single_image_dim: 300
  }
}"""

# Loop through each sample directory and process the images
for pid in os.listdir('sample'):
    # Create pbtxt file
    pbtxt_file_path = f'sample/{pid}/projector_config.pbtxt'

    with open(pbtxt_file_path, 'w') as f:
        f.write(pbtxt_content)

    print(f".pbtxt file saved at: {pbtxt_file_path}")

    # List of all image paths
    cgi_im_list = [os.path.join(f'sample/{pid}/cgi', impath) for impath in os.listdir(f'sample/{pid}/cgi')]
    fgi_im_list = [os.path.join(f'sample/{pid}/fgi', impath) for impath in os.listdir(f'sample/{pid}/fgi')]
    
    # Extract vectors for all images
    all_im_list = cgi_im_list + fgi_im_list
    all_vecs = [list(get_vector(Image.open(img))) for img in all_im_list]
    
    # Save all vectors (for TensorBoard visualization)
    with open(f'sample/{pid}/feature_vecs.tsv', 'w') as fw:
        csv_writer = csv.writer(fw, delimiter='\t')
        csv_writer.writerows(all_vecs)
    
    print(f'All img vecs saved at: sample/{pid}/feature_vecs.tsv')

    # Extract vectors for cgi images only for clustering
    cgi_vecs = [list(get_vector(Image.open(img))) for img in cgi_im_list]
    
    # Save cgi vectors
    with open(f'sample/{pid}/cgi_feature_vecs.tsv', 'w') as fw:
        csv_writer = csv.writer(fw, delimiter='\t')
        csv_writer.writerows(cgi_vecs)
    
    print(f'CGI img vecs saved at: sample/{pid}/cgi_feature_vecs.tsv')

    # Create sprite.jpg
    images = [Image.open(filename).resize((300, 300)) for filename in all_im_list]

    image_width, image_height = images[0].size
    one_square_size = int(np.ceil(np.sqrt(len(images))))
    master_width = (image_width * one_square_size)
    master_height = image_height * one_square_size

    spriteimage = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0, 0, 0, 0))  # fully transparent

    for count, image in enumerate(images):
        div, mod = divmod(count, one_square_size)
        h_loc = image_width * div
        w_loc = image_width * mod
        spriteimage.paste(image, (w_loc, h_loc))

    spriteimage.convert("RGB").save(f'sample/{pid}/sprite.jpg', transparency=0)

    print(f'Sprite image saved at: sample/{pid}/sprite.jpg')

    # Perform K-means clustering on the cgi feature vectors
    cgi_vecs_array = np.array(cgi_vecs)
    n_clusters = 5  # Number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(cgi_vecs_array)
    labels = kmeans.labels_

    print(f'K-means clustering performed with {n_clusters} clusters.')

    # Optionally, visualize the clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(cgi_vecs_array[:, 0], cgi_vecs_array[:, 1], cgi_vecs_array[:, 2], c=labels, cmap='viridis')

    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    plt.show()
