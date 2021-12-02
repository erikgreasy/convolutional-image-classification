from matplotlib import pyplot as plt


def preview_images(train_ds, class_names, show_plot=False):
    """Generate plot with one image from every class with its class label and save it. Show if set True."""

    plt.figure(figsize=(30, 30))
    for images, labels in train_ds.take(1):
        for i in range(30):
            ax = plt.subplot(5, 6, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

        plt.savefig('overview.png')

        if show_plot:
            plt.show()
