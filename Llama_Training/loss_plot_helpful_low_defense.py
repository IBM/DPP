import torch
#from torchvision.utils import save_image
import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss(loss_buffer, counter, batch):
    sns.set_theme()
    save_dir = 'YOUR_LOCAL_PATH/plot_losses_helpful_low_defense'
    num_iters = len(loss_buffer)

    x_ticks = list(range(0, num_iters))

    # Plot and label the training and validation loss values
    plt.plot(x_ticks, loss_buffer, label='Target Loss')

    # Add in a title and axes labels
    plt.title('Loss Plot')
    plt.xlabel('Iters')
    plt.ylabel('Loss')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(f'{save_dir}/Batch_{batch}_{counter}_loss_curve.png')
    plt.clf()

    torch.save(loss_buffer, f'{save_dir}/Batch_{batch}_{counter}_loss' )
