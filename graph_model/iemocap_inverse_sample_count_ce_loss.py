import torch
import torch.nn as nn


class IEMOCAPInverseSampleCountCELoss(nn.Module):

    def __init__(self):
        super(IEMOCAPInverseSampleCountCELoss, self).__init__()

        # There are four types of emotions
        # Out of 2717 traning examples, there are
        # 954 ones for emotion 0
        # 338 ones for emotion 1
        # 690 ones for emotion 2
        # 735 ones for emotion 3
        total = 2717
        self.loss_for_emotion_0 = nn.CrossEntropyLoss(weight=torch.tensor([954./total, (total-954.)/total]))
        self.loss_for_emotion_1 = nn.CrossEntropyLoss(weight=torch.tensor([338./total, (total-338.)/total]))
        self.loss_for_emotion_2 = nn.CrossEntropyLoss(weight=torch.tensor([690./total, (total-690.)/total]))
        self.loss_for_emotion_3 = nn.CrossEntropyLoss(weight=torch.tensor([735./total, (total-735.)/total]))


    def forward(self, outputs, labels):
        """

        :param outputs: shape (B * 4, 2)
        :param labels:  shape (B * 4,)
        :return:
        """
        assert outputs.shape[0] % 4 == 0
        assert outputs.shape[0] == labels.shape[0]
        assert outputs.shape[1] == 2




        outputs = outputs.reshape(-1, 4, 2)
        labels = labels.reshape(-1, 4)

        loss_0 = self.loss_for_emotion_0(outputs[:, 0, :], labels[:, 0])
        loss_1 = self.loss_for_emotion_1(outputs[:, 1, :], labels[:, 1])
        loss_2 = self.loss_for_emotion_2(outputs[:, 2, :], labels[:, 2])
        loss_3 = self.loss_for_emotion_3(outputs[:, 3, :], labels[:, 3])

        loss = (loss_0 + loss_1 + loss_2 + loss_3) / 4.

        return loss


