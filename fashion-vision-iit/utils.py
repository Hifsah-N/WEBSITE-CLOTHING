from PIL import Image
import numpy as np
from colorthief import ColorThief
import io
import cv2

# Preprocess image for model input (resize to 224x224, normalize)
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Helper: Convert RGB to HEX
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

# Helper: Get color name (simple mapping for demo)
def get_color_name(rgb):
    # Basic color mapping
    colors = {
        (255,255,255): 'White', (0,0,0): 'Black', (255,0,0): 'Red', (0,255,0): 'Green', (0,0,255): 'Blue',
        (255,255,0): 'Yellow', (255,165,0): 'Orange', (128,0,128): 'Purple', (0,255,255): 'Cyan', (128,128,128): 'Gray',
        (165,42,42): 'Brown', (255,192,203): 'Pink', (245,245,220): 'Beige', (0,128,0): 'Dark Green', (0,0,128): 'Navy'
    }
    min_dist = float('inf')
    closest = (0,0,0)
    for c in colors:
        dist = sum((a-b)**2 for a,b in zip(rgb, c))
        if dist < min_dist:
            min_dist = dist
            closest = c
    return colors[closest]

# Extract dominant color using ColorThief
def extract_dominant_color(image: Image.Image):
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        output.seek(0)
        ct = ColorThief(output)
        rgb = ct.get_color(quality=1)
        hex_code = rgb_to_hex(rgb)
        name = get_color_name(rgb)
        return {'name': name, 'hex': hex_code, 'rgb': list(rgb)}

# Estimate pattern using edge detection (simple heuristic)
def estimate_pattern(image: Image.Image):
    img = np.array(image.convert('L'))  # Grayscale
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.mean(edges > 0)
    if edge_density < 0.02:
        return 'Solid'
    elif edge_density < 0.07:
        return 'Striped'
    elif edge_density < 0.15:
        return 'Floral'
    else:
        return 'Complex/Patterned'

# Estimate material using color/texture heuristics (demo only)
def estimate_material(image: Image.Image):
    img = np.array(image)
    std = np.std(img)
    if std < 20:
        return 'Cotton'
    elif std < 40:
        return 'Denim'
    elif std < 60:
        return 'Silk'
    else:
        return 'Leather'

# Estimate style using brightness (demo only)
def estimate_style(image: Image.Image):
    img = np.array(image.convert('L'))
    mean_brightness = np.mean(img)
    if mean_brightness > 200:
        return 'Formal'
    elif mean_brightness > 120:
        return 'Casual'
    elif mean_brightness > 80:
        return 'Party'
    else:
        return 'Streetwear'
