from einops import rearrange
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from Demo_Simplified import *
import segmentation_models_pytorch as smp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from collections import OrderedDict
def remove_module(ckpt):
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    return new_state_dict
class SmokeDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data_img = data
        self.transform = transform
    def __len__(self):
        return int(self.data_img.shape[0])
    def __getitem__(self, idx):
        rgb = self.data_img
        data_tensor = self.transform(rgb)#.unsqueeze_(0)
        data_tensor = torch.nan_to_num(data_tensor)
        return data_tensor

def get_data_loader(data):
    data_transforms = transforms.Compose([transforms.ToTensor()])
    test_set = SmokeDataset(data, data_transforms)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, drop_last=True)
    return test_loader

def get_model():
    model_path = "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz/deep_learning/PLDR_models"
    PSPNet = smp.create_model(
        arch="PSPNet",
        encoder_name="timm-efficientnet-b2",
        encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization                                                          
        in_channels=3, # model input channels                                                                                                                
        classes=3, # model output channels                                                                                                                   
    )
    chkpt_pth = f"{model_path}/PSPNet_timm-efficientnet-b2_exp0_1761192426.pth"
    chkpt = torch.load(chkpt_pth, map_location=torch.device(device))
    PSPNet = PSPNet.to(device)
    PSPNet.load_state_dict(remove_module(chkpt['model_state_dict']))
    PSPNet.eval()
    return PSPNet

def get_model_ensemble():
    model_path = "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz/deep_learning/PLDR_models"
    PSPNet = smp.create_model(
        arch="PSPNet",
        encoder_name="timm-efficientnet-b2",
        encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization                                                          
        in_channels=3, # model input channels                                                                                                                
        classes=3, # model output channels                                                                                                                   
    )
    chkpt_pth = f"{model_path}/PSPNet_timm-efficientnet-b2_exp0_1761192426.pth"
    chkpt = torch.load(chkpt_pth, map_location=torch.device(device))
    PSPNet = PSPNet.to(device)
    PSPNet.load_state_dict(remove_module(chkpt['model_state_dict']))
    PSPNet.eval()

    DPT = smp.create_model(
        arch="DPT",
        encoder_name="tu-maxvit_nano_rw_256",
        encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization                                                          
        in_channels=3, # model input channels                                                                                                                
        classes=3, # model output channels                                                                                                                   
    )
    chkpt_pth = f"{model_path}/DPT_tu-maxvit_nano_rw_256_exp0_1761202022.pth"
    chkpt = torch.load(chkpt_pth, map_location=torch.device(device))
    DPT = DPT.to(device)
    DPT.load_state_dict(remove_module(chkpt['model_state_dict']))
    DPT.eval()

    UPerNet = smp.create_model(
        arch="UPerNet",
        encoder_name="tu-efficientvit_b0",
        encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization                                                          
        in_channels=3, # model input channels                                                                                                                
        classes=3, # model output channels                                                                                                                   
    )
    chkpt_pth = f"{model_path}/UPerNet_tu-efficientvit_b0_exp0_1761196360.pth"
    chkpt = torch.load(chkpt_pth, map_location=torch.device(device))
    UPerNet = UPerNet.to(device)
    UPerNet.load_state_dict(remove_module(chkpt['model_state_dict']))
    UPerNet.eval()

    return PSPNet, DPT, UPerNet

def get_pred(data_loader, model):
    torch.set_grad_enabled(False)

    for batch_data in data_loader:
        batch_data = batch_data.to(device, dtype=torch.float)
        pred = model(batch_data)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
        pred = np.einsum('ijk->jki', pred.cpu().numpy().squeeze(0))
        return pred

def get_mesh(num_pixels):
    x = np.linspace(0,num_pixels-1,num_pixels)
    y = np.linspace(0,num_pixels-1,num_pixels)
    X, Y = np.meshgrid(x,y)
    return X,Y

def plot_data_preds(lat, lon, res, img_size, data, dt_str, preds):
    lats, lons = coords_from_lat_lon(lat, lon, res, img_size)
    # Plot the composite image with geolocated axes
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(data)

    X, Y = get_mesh(img_size)
    colors = ['red', 'orange', 'yellow']
    for idx in reversed(range(3)):
        plt.contour(X,Y,preds[:,:,idx],levels =[.99],colors=[colors[idx]])

    # Set Y-axis ticks and labels using latitude values
    plt.yticks(np.linspace(0, data.shape[0] - 1, len(lats)), np.round(lats, 2), fontsize=12)
    plt.ylabel('latitude (degrees)', fontsize=16)
    # Set X-axis ticks and labels using longitude values
    plt.xticks(np.linspace(0, data.shape[1] - 1, len(lons)), np.round(lons, 2), fontsize=12)
    plt.xlabel('longitude (degrees)', fontsize=16)
    # Add a title and adjust layout
    plt.title(dt_str, fontsize=20)
    plt.tight_layout(pad=0)
    plt.show()
    print(f"{dt_str} ({lat}, {lon})")


