import torch
from torch.nn.parameter import Parameter

def main():
    file = torch.load("models/LDD/model.ckpt", map_location='cpu')
    
    new_state_dict = {}
    
    for name, param in file["state_dict"].items():
      if name.startswith("diffusion_ema.ema_model."):
          new_name = name.replace("diffusion_ema.ema_model.", "")
          if isinstance(param, Parameter):
              # backwards compatibility for serialized parameters
              param = param.data
          new_state_dict[new_name] = param
          
    for item in new_state_dict:
        print(item)
        
    model = dict(
        state_dict = new_state_dict
    )
    
    torch.save(model,"models/LDD/model.pruned.ckpt")

if __name__ == '__main__':
    main()