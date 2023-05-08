import torch
image_path = 'images/video1-00150.jpg' 
model, results, plt = torch.hub.load('AlanSavio25/DeepSingleImageCalibration',
                            'calib', image_path=image_path, force_reload=False)
if image_path is not None:
    print(results)
    plt.savefig('image.png')
    print("Saved plot to image.png")
else:
    print(model)
