from PIL import Image
import torchvision.transforms as transforms
import torch


def predict(model, image_path, device, genre_map, in_channel=1):

    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  
        transforms.Resize((400, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * in_channel, std=[0.5] * in_channel)
    ])

    image = Image.open(image_path).convert("L")

    input_tensor = transform(image) 
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    index_to_genre = {v: k for k, v in genre_map.items()}
    return index_to_genre[predicted_class]
