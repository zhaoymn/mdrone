import torch.nn as nn
import torch.nn.functional as F
import torch

class FeatureExtractor_azimuth(nn.Module):
    def __init__(self):
        super(FeatureExtractor_azimuth, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, 3)
        #self.bn1 = nn.BatchNorm2d(8)
        #torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(8, 16, 3)
        #self.bn2 = nn.BatchNorm2d(16)
        #torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(16, 32, 3)
        #self.bn3 = nn.BatchNorm2d(32)
        #torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.maxpool2d = nn.MaxPool2d(2, 1)
        self.fc2d = nn.Linear(161024, 256)
        self.dropout1 = nn.Dropout(0.3)
    def forward(self, x):
        #x = self.bn1(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = self.maxpool2d(x)
        #x = self.bn2(F.relu(self.conv2(x))）
        x = F.relu(self.conv2(x))
        x = self.maxpool2d(x)
        #x = self.bn3(F.relu(self.conv3(x)))
        x = F.relu(self.conv3(x))
        x = self.maxpool2d(x)
        x = x.view(-1, 161024)
        x = self.fc2d(x)
        #x = self.dropout1(x)
        return x

class FeatureExtractor_elevation(nn.Module):
    def __init__(self):
        super(FeatureExtractor_elevation, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, 3)
        #self.bn1 = nn.BatchNorm2d(8)
        #torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(8, 16, 3)
        #self.bn2 = nn.BatchNorm2d(16)
        #torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(16, 32, 3)
        #self.bn3 = nn.BatchNorm2d(32)
        #torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.maxpool2d = nn.MaxPool2d(2, 1)
        self.fc2d = nn.Linear(161024, 256)
        self.dropout1 = nn.Dropout(0.3)
    def forward(self, x):
        #x = self.bn1(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = self.maxpool2d(x)
        #x = self.bn2(F.relu(self.conv2(x))）
        x = F.relu(self.conv2(x))
        x = self.maxpool2d(x)
        #x = F.relu(self.conv3(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2d(x)

        x = x.view(-1, 161024)
        x = self.fc2d(x)
        #x = self.dropout1(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.fe_azimuth = FeatureExtractor_azimuth()
        self.fe_elevation = FeatureExtractor_elevation()
        self.fc = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.3)
        self.mask = nn.Linear(512,512)
    
    def forward(self, x_azimuth, x_elevation):
        f_azimuth = self.fe_azimuth(x_azimuth)
        f_elevation = self.fe_elevation(x_elevation)
        x = torch.cat((f_azimuth, f_elevation), 1)
        x = self.fc(x)
        att = F.sigmoid(self.mask(x))
        x = x.mul(att)
        #x = self.dropout(x)
        return x



class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.feature_num = 512
        #self.rnn = nn.RNN(input_size = self.feature_num, hidden_size = 1024, num_layers =1, batch_first = True)
        self.lstm = nn.LSTM(input_size = self.feature_num, hidden_size = 512, num_layers =1, batch_first = True)
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2_d = nn.Linear(256, 128)
        self.dropout2_d = nn.Dropout(0.2)
        self.fc3_d = nn.Linear(128, 3)

    def forward(self, x_azimuth, x_elevation):
        #print(x.shape)
        #output = torch.tensor([]).cuda()
        x_feature = torch.tensor([]).cuda()
        for i in range(8):
            output_t = self.feature_extractor(x_azimuth[:,i,:,:,:], x_elevation[:,i,:,:,:])
            x_feature = torch.cat((x_feature, output_t.unsqueeze(1)), 1)
        output, (h_n, c_n) = self.lstm(x_feature)
        #output, h_n = self.rnn(x_feature)
        x = output[:, -1, :]
        #x = self.feature_extractor(x_azimuth[:,0,:,:,:], x_elevation[:,0,:,:,:])
        x = F.relu(self.fc1(x))
        #x = self.dropout1(x)
        x_d = F.relu(self.fc2_d(x))
        #x_d = self.dropout2_d(x_d)
        x = self.fc3_d(x_d)
        #print(x.shape)
        #print(x.size())
        return x
