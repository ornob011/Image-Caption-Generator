import matplotlib.pyplot as plt


def show_results(history):

    plt.figure(figsize=(7, 6))
    plt.plot(history.history['loss'], '-o', label='Training Loss')
    plt.plot(history.history['val_loss'], '-o', label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    # plt.show()
    plt.savefig('training_vs_validation_loss.png', dpi=600)
