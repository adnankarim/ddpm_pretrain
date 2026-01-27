
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

try:
    fid = FrechetInceptionDistance()
    print("FID initialized")
    if hasattr(fid, 'inception'):
        print("Has 'inception' attribute")
        print(f"Type: {type(fid.inception)}")
    else:
        print("Does NOT have 'inception' attribute")
        
    print(f"Attributes: {dir(fid)}")
    
except Exception as e:
    print(f"Error: {e}")
