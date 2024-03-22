import matplotlib.pyplot as plt

def show_sequence_task1_MNIST(sequence, class_labels, repeat_labels):
    # sequence shape: [sequence_length, channels, height, width]
    # class_labels shape: [sequence_length]
    # repeat_labels shape: [sequence_length]

    fig, axs = plt.subplots(1, len(sequence), figsize=(15, 6))  
    for i, img in enumerate(sequence):
        img = img.squeeze().numpy() 
        axs[i].imshow(img, cmap='gray')  
        axs[i].set_title(f'Class: {class_labels[i]}, Repeat: {repeat_labels[i]}', fontsize=8)  # Show class and repeat labels
        axs[i].axis('off')  

    plt.tight_layout()
    plt.show()
