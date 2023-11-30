import matplotlib.pyplot as plt

# 初始化列表来存储解析的数据
epochs = []
train_acc = []
test_acc = []
train_loss = []


with open('/data/home/xiezicheng/clip_MIA/plot.txt', 'r') as file:
    for line in file:
        parts = line.split()

        if len(parts)>0:
            if parts[0] == 'epoch:':

                epoch_number = parts[1].rstrip(',')
                epochs.append(int(epoch_number))
            elif parts[0] == 'train' and parts[1] == 'acc:':
                train_acc.append(float(parts[2].rstrip(',')))
            elif parts[0] == 'test' and parts[1] == 'acc:':
                test_acc.append(float(parts[2].rstrip(',')))
            elif parts[0] == 'train' and parts[1] == 'loss:':
                train_loss.append(float(parts[2].rstrip(',')))

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, test_acc, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy per Epoch')
plt.legend()
plt.savefig('/data/home/xiezicheng/clip_MIA/save_figure/accuracy_per_epoch.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.savefig('/data/home/xiezicheng/clip_MIA/save_figure/loss_per_epoch.png')
plt.close()
