import matplotlib.pyplot as plt
import seaborn as sns


def save_metric_plot(output_file, arr_train, arr_validation, x_label, y_label, loc="lower right"):
    sns.set(style='darkgrid')
    plt.xticks([ i+1 for i in range(arr_train.shape[0])])
    plt.plot(arr_train, 'b-o', label="train")
    plt.plot(arr_validation, 'r-o', label="validation")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=loc)
    plt.savefig(f"{output_file}.png")
    
    plt.close()
