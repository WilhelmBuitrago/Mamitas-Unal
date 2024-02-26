from gcpds.DataSet.infrared_thermal_feet import ModifiedInfraredThermalFeet
import gcpds.DataSet.convRFFds.data as data
from utils import resizen, get_episodenk
from FSL.Metrics import compute_dice
from FSL.models.FSLModel import FSLmodel
import matplotlib.pyplot as plt
import tensorflow as tf
import os


Valid_Dice = []
loss_entro = []
accur = []
Best_performance = 0
filepath = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(filepath, "FSL", "models",
                        "Models_saved", "BestModelEpoc.h5")


def train(model, traind, vald, parameters):
    global loss_entro
    global accur
    epochs = parameters['epochs']
    tr_iterations = parameters['tr_iterations']
    it_eval = parameters['it_eval']
    nway = parameters['nway']
    kshot = parameters['kshot']

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        device = '/gpu:0'
    else:
        device = '/cpu:0'
    with tf.device(device):
        for ep in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            for idx in range(tr_iterations):
                support, smask, query, qmask = get_episodenk(
                    traind, nway, kshot)
                acc_loss = model.train_on_batch([support, smask, query], qmask)
                # Losses
                epoch_loss += acc_loss[0]
                epoch_acc += acc_loss[1]

                if (idx % 10) == 0:
                    print('Base_Model:::Epoch > ', (ep+1), ' --- Iteration > ', (idx+1), '/',
                          tr_iterations, ' --- BM_Loss:', epoch_loss/(idx+1), ' --- Acc: ', epoch_acc/(idx+1))
                if (idx == tr_iterations-1):
                    print('Base_Model:::Epoch > ', (ep+1), ' --- Iteration > ', (idx+1), '/',
                          tr_iterations, ' --- BM_Loss:', epoch_loss/(idx+1), ' --- Acc: ', epoch_acc/(idx+1))
                    accur.append(epoch_acc/(idx+1))
                    loss_entro.append(epoch_loss/(idx+1))
            evaluate(ep, model, vald, nway, kshot, it_eval)
        return Valid_Dice, loss_entro, accur


def evaluate(ep, model, vald, nway, kshot, it_eval):
    global Best_performance
    global Valid_Dice
    overall_Dice = 0.0

    for idx in range(it_eval):
        support, smask, query, qmask = get_episodenk(vald, nway, kshot)
        Es_mask = model.predict([support, smask, query])
        Dice_score = compute_dice(Es_mask, qmask)
        overall_Dice += Dice_score

    print('Epoch>>>', ep+1, 'Dice score on Ph2 set>> ', overall_Dice / it_eval)
    Valid_Dice.append(overall_Dice / it_eval)
    if Best_performance < (overall_Dice / it_eval):
        Best_performance = (overall_Dice / it_eval)
        model.save(filepath, save_format="tf")
        print("New best model saved")


if __name__ == "__main__":
    resize = False
    if resize:
        resizen()
    kwargs_data_augmentation = dict(repeat=1,
                                    batch_size=2,
                                    shape=224,
                                    split=[0.4, 0.3]
                                    )
    dataset = ModifiedInfraredThermalFeet
    train_dataset, val_dataset, test_dataset = data.get_data(
        dataset_class=dataset, data_augmentation=False, return_label_info=True, **kwargs_data_augmentation)
    """    
    LR = 0.0001
    input_size = (224, 224, 1)
    encoder = 'VGG'
    k_shot = 5
    Model10way5shot = FSLmodel(
        encoder=encoder, input_size=input_size, k_shot=k_shot, learning_rate=LR)
    parameters = {'epochs': 1,
                  'tr_iterations': 10,
                  'it_eval': 1,
                  'nway': 10,
                  'kshot': 5}
    train(Model10way5shot, train_dataset, val_dataset, parameters)

    plt.subplot(1, 3, 1)
    plt.plot(Valid_Dice)
    plt.title("Dice for Validation")
    plt.subplot(1, 3, 2)
    plt.plot(loss_entro)
    plt.title("Loss")
    plt.subplot(1, 3, 3)
    plt.plot(accur)
    plt.title("Accuracy")
    plt.show()
    """
