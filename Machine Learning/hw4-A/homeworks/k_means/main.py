if __name__ == "__main__":
    from k_means import lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code might take a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. Make sure to change it back before submission!
    """
    (x_train, _), _ = load_dataset("mnist")
    train = x_train[:10000]
    result = lloyd_algorithm(x_train, 10)
    fig, axs = plt.subplots(1, 10, figsize=(20, 4))

    centers = result[0]
    for i in range(len(centers)):
        image = np.reshape(centers[i], (28, 28))  # Reshape to 28x28 image
        axs[i].imshow(image, cmap='gray')
        axs[i].axis('on')
        axs[i].set_title(f'Cluster {i}')

    plt.tight_layout()
    plt.show()
    




if __name__ == "__main__":
    main()
