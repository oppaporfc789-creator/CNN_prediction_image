import os
import shutil

# Define project directories and class labels
CLASSES = ['Cat', 'Cow', 'Dog', 'Elephant', 'Sheep']
BASE_DIR = 'Dataset'
SPLITS = ['train', 'validation', 'test']

def verify_dataset_structure():
    """Verify directory structure and print image counts per class."""
    print("Checking dataset directory structure...\n")
    
    for split in SPLITS:
        split_path = os.path.join(BASE_DIR, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path, exist_ok=True)
            print(f"Created missing directory: {split_path}")
            
        if split in ['train', 'validation']:
            for cls in CLASSES:
                cls_path = os.path.join(split_path, cls)
                os.makedirs(cls_path, exist_ok=True)
                count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"[{split.upper()}] Class '{cls}': {count} images found.")
        print("-" * 40)

if __name__ == '__main__':
    verify_dataset_structure()